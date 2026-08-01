"""Covalent docking strategy built on :mod:`anchor_dock.core`."""

from __future__ import annotations

import os
import time
from typing import Literal

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

from anchor_dock.core import (
    PocketBundle,
    compute_intramolecular_mask,
    compute_vina_features,
    extract_pocket_around_residue,
    generate_conformers_and_cluster,
    optimize_torsions_vina,
    precompute_interaction_matrices,
    process_query_ligand,
    vina_scoring,
    write_ranked_poses,
)

from .adduct import create_adduct_template, get_protein_exclusion_atom_indices
from .anchor import (
    REACTIVE_RESIDUES,
    AnchorPoint,
    WarheadHit,
    check_warhead_residue_compatibility,
    create_covalent_coordmap,
    detect_warheads,
    find_reactive_residues,
)


def _device(value: str | torch.device | None) -> torch.device:
    if value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return result


def _residue_id(anchor: AnchorPoint) -> str:
    base = f"{anchor.residue_name}{anchor.residue_num}"
    return f"{base}:{anchor.chain_id}" if anchor.chain_id else base


def load_pocket_for_caching(
    protein_pdb: str,
    reactive_residue: str | None = None,
    pocket_cutoff: float = 12.0,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    """Load a receptor, choose an anchor, and cache its extracted pocket features."""
    target_device = _device(device)
    protein = Chem.MolFromPDBFile(protein_pdb, sanitize=False, removeHs=True)
    if protein is None:
        raise ValueError(f"Failed to load protein from {protein_pdb}")
    anchors = find_reactive_residues(protein, reactive_residue)
    if not anchors:
        supported = ", ".join(f"{name}({config.atom_name})" for name, config in REACTIVE_RESIDUES.items())
        raise ValueError(f"No reactive residue found for {reactive_residue or 'auto-detect'}; supported: {supported}")
    anchor = anchors[0]
    residue_id = _residue_id(anchor)
    pocket = extract_pocket_around_residue(protein, residue_id, cutoff=pocket_cutoff)
    relocated = find_reactive_residues(pocket, residue_id)
    if not relocated:
        raise RuntimeError(f"Reactive residue {residue_id} was lost during pocket extraction")
    anchor = relocated[0]
    coords = torch.tensor(pocket.GetConformer().GetPositions(), dtype=torch.float32, device=target_device)
    features = compute_vina_features(pocket, target_device)
    bundle = PocketBundle(mol=pocket, coords=coords, features=features)
    if verbose:
        print(
            f"Anchor: {residue_id} {anchor.atom_name}; pocket={pocket.GetNumAtoms()} atoms; "
            f"device={target_device}"
        )
    return {
        "pocket_bundle": bundle,
        "anchor": anchor,
        "residue_spec_str": residue_id,
        "device": target_device,
        "source_path": os.path.abspath(protein_pdb),
        "pocket_cutoff": float(pocket_cutoff),
    }


def _align_anchor_atoms(
    mol: Chem.Mol,
    conformer_ids: list[int],
    anchor_indices: list[int],
    target_positions: np.ndarray,
) -> None:
    """Rigidly align each conformer anchor frame to receptor coordinates."""
    for conformer_id in conformer_ids:
        conformer = mol.GetConformer(conformer_id)
        current = np.asarray([conformer.GetAtomPosition(idx) for idx in anchor_indices], dtype=float)
        if len(anchor_indices) == 1:
            aligned = np.asarray(conformer.GetPositions(), dtype=float) + (target_positions[0] - current[0])
        else:
            current_center = current.mean(axis=0)
            target_center = target_positions.mean(axis=0)
            covariance = (current - current_center).T @ (target_positions - target_center)
            u, _, vt = np.linalg.svd(covariance)
            correction = np.eye(3)
            if np.linalg.det(vt.T @ u.T) < 0:
                correction[-1, -1] = -1
            rotation = vt.T @ correction @ u.T
            aligned = (np.asarray(conformer.GetPositions(), dtype=float) - current_center) @ rotation.T + target_center
        for atom_idx, xyz in enumerate(aligned):
            conformer.SetAtomPosition(atom_idx, Point3D(*map(float, xyz)))


def _axis_rotation_scan(
    coords: torch.Tensor,
    cb_position: np.ndarray,
    nucleophile_position: np.ndarray,
    step_degrees: int,
) -> torch.Tensor:
    """Rotate all poses around the fixed Cβ→nucleophile axis."""
    if step_degrees <= 0:
        return coords
    device, dtype = coords.device, coords.dtype
    cb = torch.as_tensor(cb_position, dtype=dtype, device=device)
    origin = torch.as_tensor(nucleophile_position, dtype=dtype, device=device)
    axis = (origin - cb) / torch.linalg.vector_norm(origin - cb).clamp_min(1e-12)
    angles = torch.arange(0, 360, step_degrees, dtype=dtype, device=device) * torch.pi / 180.0
    x, y, z = axis
    zeros = torch.zeros_like(x)
    skew = torch.stack((zeros, -z, y, z, zeros, -x, -y, x, zeros)).reshape(3, 3)
    identity = torch.eye(3, dtype=dtype, device=device)
    rotations = identity + torch.sin(angles)[:, None, None] * skew + (1.0 - torch.cos(angles))[:, None, None] * (skew @ skew)
    shifted = coords[None, :, :, :] - origin
    return torch.einsum("rpnc,rcd->rpnd", shifted, rotations.transpose(1, 2)).add(origin).reshape(-1, *coords.shape[1:])


def _score_in_chunks(
    coords: torch.Tensor,
    pocket: PocketBundle,
    query_features: dict[str, torch.Tensor],
    preset: str,
    precomputed: dict[str, torch.Tensor],
    chunk_size: int = 256,
) -> torch.Tensor:
    scores = []
    for start in range(0, coords.shape[0], chunk_size):
        scores.append(
            vina_scoring(
                coords[start : start + chunk_size],
                pocket.coords,
                query_features,
                pocket.features,
                weight_preset=preset,
                precomputed_matrices=precomputed,
            )
        )
    return torch.cat(scores)


def dock_covalent(
    protein_pdb: str,
    query_ligand: str,
    reactive_residue: str | None = None,
    output_dir: str = "output_predictions",
    pocket_cutoff: float = 12.0,
    _cached_pocket: dict[str, object] | None = None,
    num_confs: int = 1000,
    rmsd_threshold: float = 1.0,
    rotation_scan_step: int = 30,
    rotation_top_k: int = 50,
    optimize: bool = False,
    optimizer: Literal["adam", "adamw", "lbfgs"] = "adam",
    opt_steps: int = 100,
    opt_lr: float = 0.05,
    opt_batch_size: int = 128,
    weight_preset: Literal["vina", "vina_lp", "vinardo"] = "vina",
    torsion_penalty: bool = True,
    save_all_poses: bool | None = None,
    top_k: int | None = None,
    device: str | torch.device | None = None,
    verbose: bool = True,
    warhead_index: int = 0,
    strict_compatibility: bool = False,
) -> dict[str, object]:
    """Dock a reactive ligand by constructing and optimizing a residue-linked adduct."""
    started = time.perf_counter()
    if not verbose:
        RDLogger.DisableLog("rdApp.warning")
    cache = _cached_pocket or load_pocket_for_caching(
        protein_pdb, reactive_residue, pocket_cutoff, device, verbose=verbose
    )
    pocket = cache["pocket_bundle"]
    anchor = cache["anchor"]
    target_device = cache["device"]
    if not isinstance(pocket, PocketBundle) or not isinstance(anchor, AnchorPoint) or not isinstance(target_device, torch.device):
        raise TypeError("Invalid cached pocket object")

    ligand, canonical_smiles = process_query_ligand(query_ligand)
    hits = detect_warheads(ligand)
    if not hits:
        raise ValueError("No supported reactive warhead detected")
    if not 0 <= warhead_index < len(hits):
        raise IndexError(f"warhead_index={warhead_index} outside 0..{len(hits)-1}")
    hit = hits[warhead_index]
    compatible, message = check_warhead_residue_compatibility(
        hit.warhead_type, anchor.residue_name, strict=strict_compatibility
    )
    if not compatible:
        raise ValueError(message)
    if verbose:
        print(f"Warhead: {hit.warhead_type} at atom {hit.reactive_atom_idx}; {message}")

    original_rotors = None
    if torsion_penalty:
        from rdkit.Chem import rdMolDescriptors
        original_rotors = rdMolDescriptors.CalcNumRotatableBonds(ligand)

    adduct, cb_idx, nuc_idx, reactive_idx = create_adduct_template(ligand, hit, anchor)
    coord_map = create_covalent_coordmap(cb_idx, nuc_idx, anchor)
    adduct, conformer_ids = generate_conformers_and_cluster(
        adduct,
        target_device,
        num_confs=num_confs,
        rmsd_threshold=rmsd_threshold,
        coordMap=coord_map,
        exact_coordmap_before_clustering=True,
    )

    anchor_indices = ([cb_idx] if cb_idx is not None else []) + [nuc_idx]
    target_positions = np.asarray(
        ([anchor.cb_coord] if cb_idx is not None and anchor.cb_coord is not None else []) + [anchor.coord],
        dtype=float,
    )
    _align_anchor_atoms(adduct, conformer_ids, anchor_indices, target_positions)
    coords = torch.stack(
        [torch.tensor(adduct.GetConformer(cid).GetPositions(), dtype=torch.float32) for cid in conformer_ids]
    ).to(target_device)

    query_features = compute_vina_features(adduct, target_device)
    precomputed = precompute_interaction_matrices(query_features, pocket.features, target_device)
    if rotation_scan_step > 0 and cb_idx is not None and anchor.cb_coord is not None:
        if rotation_top_k <= 0:
            raise ValueError("rotation_top_k must be positive when rotation scanning is enabled")
        rotations_per_conformer = len(range(0, 360, rotation_scan_step))
        rotated = _axis_rotation_scan(coords, anchor.cb_coord, anchor.coord, rotation_scan_step)
        quick_scores = _score_in_chunks(rotated, pocket, query_features, weight_preset, precomputed)
        matrix = quick_scores.reshape(rotations_per_conformer, coords.shape[0])
        best_rotation = matrix.argmin(dim=0)
        flat_indices = best_rotation * coords.shape[0] + torch.arange(coords.shape[0], device=target_device)
        best_per_conformer = rotated[flat_indices]
        best_scores = quick_scores[flat_indices]
        selected = torch.argsort(best_scores)[: min(rotation_top_k, coords.shape[0])]
        coords = best_per_conformer[selected]

    # Exclude pseudo protein atoms against the receptor and the duplicated receptor nucleophile column.
    pseudo_atoms = {idx for idx in (cb_idx, nuc_idx) if idx is not None}
    protein_nuc = get_protein_exclusion_atom_indices(pocket.mol, anchor, n_hop_exclude=0)
    exclusion = torch.zeros(
        1, adduct.GetNumAtoms(), pocket.mol.GetNumAtoms(), dtype=torch.bool, device=target_device
    )
    for idx in pseudo_atoms:
        exclusion[:, idx, :] = True
    for protein_idx in protein_nuc:
        exclusion[:, :, protein_idx] = True
        exclusion[:, reactive_idx, protein_idx] = True

    intra_mask = compute_intramolecular_mask(adduct, target_device, exclude_atom_indices=pseudo_atoms)
    scores = vina_scoring(
        coords,
        pocket.coords,
        query_features,
        pocket.features,
        original_rotors,
        weight_preset,
        intramolecular_mask=intra_mask,
        precomputed_matrices=precomputed,
        intermolecular_exclusion_mask=exclusion,
    )
    initial_scores = scores.clone()

    if optimize:
        fixed = anchor_indices
        coords = optimize_torsions_vina(
            adduct,
            fixed,
            coords,
            pocket.coords,
            query_features,
            pocket.features,
            target_device,
            num_steps=opt_steps,
            lr=opt_lr,
            freeze_anchor=True,
            num_rotatable_bonds=original_rotors,
            weight_preset=weight_preset,
            batch_size=opt_batch_size,
            optimizer=optimizer,
            intermolecular_exclusion_mask=exclusion,
            precomputed_matrices=precomputed,
            intramolecular_exclude_indices=pseudo_atoms,
        )
        scores = vina_scoring(
            coords,
            pocket.coords,
            query_features,
            pocket.features,
            original_rotors,
            weight_preset,
            intramolecular_mask=intra_mask,
            precomputed_matrices=precomputed,
            intermolecular_exclusion_mask=exclusion,
        )

    adduct.SetProp("AnchorDock_Mode", "covalent")
    adduct.SetProp("AnchorDock_Score_Semantics", "nonbonded_pose_score_conditioned_on_adduct")
    adduct.SetProp("AnchorDock_Warhead_Type", hit.warhead_type)
    adduct.SetProp("AnchorDock_Reactive_Atom_Idx", str(reactive_idx))
    adduct.SetProp("AnchorDock_Anchor_Residue", _residue_id(anchor))
    adduct.SetProp("AnchorDock_Anchor_Atom", anchor.atom_name)
    adduct.SetProp("AnchorDock_Gradient_Optimized", str(bool(optimize)))
    # Legacy SDF properties.
    adduct.SetProp("CovVina_Warhead_Type", hit.warhead_type)
    adduct.SetProp("CovVina_Reactive_Atom_Idx", str(reactive_idx))
    adduct.SetProp("CovVina_Anchor_Residue", _residue_id(anchor))
    adduct.SetProp("CovVina_Bond_Length", f"{anchor.bond_length:.2f}")

    if save_all_poses is False and top_k is None:
        top_k = 3
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "covalent_poses_all.sdf" if top_k is None else f"covalent_pose_top{top_k}.sdf")
    selected = write_ranked_poses(
        adduct,
        coords,
        scores,
        output_file,
        initial_scores=initial_scores,
        pose_ids=list(range(coords.shape[0])),
        top_k=top_k,
    )
    runtime = time.perf_counter() - started
    result = {
        "mode": "covalent",
        "output_file": output_file,
        "num_poses": int(selected.numel()),
        "best_score": float(scores.min().detach().cpu()),
        "score_semantics": "nonbonded_pose_score_conditioned_on_adduct",
        "runtime": runtime,
        "num_conformers": int(num_confs),
        "num_representatives": int(coords.shape[0]),
        "warhead_type": hit.warhead_type,
        "anchor_residue": _residue_id(anchor),
        "anchor_atom": anchor.atom_name,
        "canonical_smiles": canonical_smiles,
        "device": str(target_device),
    }
    if verbose:
        print(f"Covalent docking complete: {result['num_poses']} poses, best={result['best_score']:.3f}, {runtime:.2f}s")
    return result


# Compatibility name used by CovVina.
run_covalent_pipeline = dock_covalent
