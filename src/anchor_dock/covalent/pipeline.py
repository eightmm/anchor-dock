"""Covalent residue-warhead docking on the shared AnchorDock engine."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from ..core.conformers import generate_conformers_and_cluster
from ..core.engine import DockingEngine
from ..core.features import ATOM_TYPING_VERSION
from ..core.io import (
    ReceptorContext,
    choose_device,
    extract_pocket_around_residue,
    load_ligand,
    receptor_context_from_mol,
)
from ..core.kinematics import get_batched_rotation_matrix
from ..core.output import write_ranked_poses
from ..core.scoring import ScorerLike
from .adduct import (
    create_adduct_template,
    create_covalent_exclusion_mask,
    find_receptor_nucleophile_index,
)
from .anchor import (
    AnchorPoint,
    check_warhead_residue_compatibility,
    create_covalent_coordmap,
    detect_warheads,
    select_reactive_anchor,
)


@dataclass(frozen=True)
class CovalentReceptorContext:
    """Cached reactive anchor and extracted receptor pocket."""

    anchor: AnchorPoint
    receptor: ReceptorContext


_COVALENT_CONTEXT_CACHE: dict[tuple[object, ...], CovalentReceptorContext] = {}


def clear_covalent_context_cache() -> None:
    _COVALENT_CONTEXT_CACHE.clear()


def _prepare_covalent_receptor(
    protein_pdb: str | os.PathLike[str],
    reactive_residue: str | None,
    pocket_cutoff: float,
    include_heteroatoms: bool,
    device: str | torch.device | None,
) -> CovalentReceptorContext:
    target_device = choose_device(device)
    path = os.path.abspath(os.fspath(protein_pdb))
    stat = os.stat(path)
    key = (
        path,
        stat.st_mtime_ns,
        stat.st_size,
        reactive_residue.strip() if reactive_residue is not None else None,
        float(pocket_cutoff),
        bool(include_heteroatoms),
        str(target_device),
        ATOM_TYPING_VERSION,
    )
    cached = _COVALENT_CONTEXT_CACHE.get(key)
    if cached is not None:
        return cached
    protein = Chem.MolFromPDBFile(path, sanitize=False, removeHs=True)
    if protein is None:
        raise ValueError(f"failed to load protein from {protein_pdb}")
    anchor = select_reactive_anchor(protein, reactive_residue)
    pocket = extract_pocket_around_residue(
        protein,
        anchor.residue_id,
        cutoff=pocket_cutoff,
        include_heteroatoms=include_heteroatoms,
    )
    anchor = select_reactive_anchor(pocket, anchor.residue_id)
    receptor = receptor_context_from_mol(pocket, target_device, source_path=path)
    context = CovalentReceptorContext(anchor, receptor)
    _COVALENT_CONTEXT_CACHE[key] = context
    return context


def _rotation_scan(
    coords: torch.Tensor,
    support_coord: torch.Tensor,
    nucleophile_coord: torch.Tensor,
    step_degrees: int,
) -> torch.Tensor:
    if step_degrees <= 0:
        return coords.unsqueeze(0)
    if step_degrees > 360:
        raise ValueError("rotation_scan_step must be in 1..360 or 0 to disable")
    angles = torch.arange(0, 360, step_degrees, dtype=coords.dtype, device=coords.device)
    angles = angles * torch.pi / 180.0
    axis = nucleophile_coord - support_coord
    axes = axis.unsqueeze(0).expand(angles.shape[0], -1)
    rotations = get_batched_rotation_matrix(axes, angles)
    shifted = coords.unsqueeze(0) - nucleophile_coord
    return torch.matmul(shifted, rotations.transpose(1, 2)[:, None, :, :]) + nucleophile_coord


def dock_covalent(
    protein_pdb: str | os.PathLike[str],
    query_ligand: str | os.PathLike[str] | Chem.Mol,
    reactive_residue: str | None = None,
    output_dir: str | os.PathLike[str] = "anchor_dock_covalent",
    *,
    pocket_cutoff: float = 12.0,
    include_heteroatoms: bool = True,
    num_confs: int = 1000,
    rmsd_threshold: float = 1.0,
    rotation_scan_step: int = 30,
    rotation_top_k: int = 50,
    optimize: bool = True,
    optimizer: Literal["adam", "lbfgs"] = "adam",
    opt_steps: int = 100,
    opt_lr: float = 0.05,
    opt_batch_size: int = 128,
    scorer: ScorerLike = "vina",
    torsion_penalty: bool = True,
    top_k: int | None = None,
    warhead_index: int = 0,
    strict_compatibility: bool = False,
    random_seed: int = 42,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    """Dock a reactive ligand against one explicitly resolved protein anchor.

    When ``reactive_residue`` is omitted, automatic selection is accepted only
    when the protein contains exactly one supported nucleophile.
    """
    started = time.perf_counter()
    context = _prepare_covalent_receptor(
        protein_pdb, reactive_residue, pocket_cutoff, include_heteroatoms, device
    )
    anchor = context.anchor
    receptor = context.receptor
    target_device = receptor.device

    ligand, canonical_smiles = load_ligand(query_ligand, add_hydrogens=False)
    warheads = detect_warheads(ligand)
    if not warheads:
        raise ValueError("no supported reactive warhead detected")
    if not 0 <= warhead_index < len(warheads):
        raise IndexError(f"warhead_index={warhead_index} outside 0..{len(warheads) - 1}")
    warhead = warheads[warhead_index]
    compatible, compatibility_message = check_warhead_residue_compatibility(
        warhead.warhead_type,
        anchor.residue_name,
        strict=strict_compatibility,
    )
    if not compatible:
        raise ValueError(compatibility_message)

    num_rotatable_bonds = (
        int(rdMolDescriptors.CalcNumRotatableBonds(ligand)) if torsion_penalty else 0
    )
    adduct, support_idx, nucleophile_idx, reactive_idx = create_adduct_template(ligand, warhead, anchor)
    coord_map = create_covalent_coordmap(support_idx, nucleophile_idx, reactive_idx, anchor)
    adduct, representative_ids = generate_conformers_and_cluster(
        adduct,
        target_device,
        num_confs=num_confs,
        rmsd_threshold=rmsd_threshold,
        coord_map=coord_map,
        exact_constraints_before_clustering=True,
        add_hydrogens=False,
        random_seed=random_seed,
    )
    if not representative_ids:
        raise RuntimeError("covalent conformer generation produced no representative poses")
    coords = torch.stack(
        [torch.tensor(adduct.GetConformer(conf_id).GetPositions(), dtype=torch.float32) for conf_id in representative_ids]
    ).to(target_device)

    receptor_nucleophile_idx = find_receptor_nucleophile_index(receptor.mol, anchor)
    pseudo_indices = {support_idx, nucleophile_idx}
    exclusion = create_covalent_exclusion_mask(
        adduct,
        receptor.mol,
        pseudo_atom_indices=pseudo_indices,
        reactive_atom_idx=reactive_idx,
        receptor_nucleophile_idx=receptor_nucleophile_idx,
        device=target_device,
    )
    engine = DockingEngine(
        scorer,
        device=target_device,
        optimizer=optimizer,
        num_steps=opt_steps,
        learning_rate=opt_lr,
        batch_size=opt_batch_size,
    )
    fixed_indices = (support_idx, nucleophile_idx, reactive_idx)
    problem = engine.prepare(
        adduct,
        receptor,
        anchor_indices=fixed_indices,
        num_rotatable_bonds=num_rotatable_bonds,
        exclude_intramolecular_atoms=pseudo_indices,
        intermolecular_exclusion_mask=exclusion,
    )

    if rotation_scan_step > 0:
        rotated = _rotation_scan(
            coords,
            torch.as_tensor(anchor.support_coord, dtype=coords.dtype, device=target_device),
            torch.as_tensor(anchor.coord, dtype=coords.dtype, device=target_device),
            rotation_scan_step,
        )
        rotation_count, conformer_count = rotated.shape[:2]
        scan_scores = problem.scorer.search_energy(rotated.reshape(-1, *coords.shape[1:])).reshape(
            rotation_count, conformer_count
        )
        best_rotation = scan_scores.argmin(dim=0)
        conformer_indices = torch.arange(conformer_count, device=target_device)
        coords = rotated[best_rotation, conformer_indices]
        if rotation_top_k <= 0:
            raise ValueError("rotation_top_k must be positive when rotation scanning is enabled")
        best_scores = scan_scores[best_rotation, conformer_indices]
        keep = torch.argsort(best_scores)[: min(rotation_top_k, conformer_count)]
        coords = coords[keep]

    initial_coords = coords
    if optimize:
        final_coords, optimization_stats = engine.optimize_anchored(problem, initial_coords, freeze_anchor=True)
    else:
        final_coords = initial_coords
        optimization_stats = None
    initial_components, final_components = engine.report_scores(problem, initial_coords, final_coords)

    bond_lengths = torch.linalg.vector_norm(
        final_coords[:, reactive_idx] - final_coords[:, nucleophile_idx],
        dim=1,
    )
    target_length = float(anchor.bond_length)
    if not torch.allclose(bond_lengths, torch.full_like(bond_lengths, target_length), atol=1e-4, rtol=0.0):
        raise RuntimeError("covalent bond-length invariant was violated during optimization")

    output_path = Path(output_dir) / "covalent_poses.sdf"
    pose_metadata = [
        {
            "Warhead_Type": warhead.warhead_type,
            "Reactive_Atom_Index": reactive_idx,
            "Covalent_Bond_Length": f"{float(length):.6f}",
        }
        for length in bond_lengths.detach().cpu()
    ]
    selected = write_ranked_poses(
        adduct,
        final_coords,
        final_components.score,
        str(output_path),
        scorer_name=problem.scorer.name,
        score_units=problem.scorer.units,
        search_energies=final_components.search_energy,
        initial_scores=initial_components.score,
        pose_ids=[f"p{index:04d}" for index in range(final_coords.shape[0])],
        top_k=top_k,
        molecule_metadata={
            "Mode": "covalent",
            "Anchor_Strategy": "residue_warhead",
            "Anchor_Residue": anchor.residue_id,
            "Anchor_Atom": anchor.atom_name,
            "Support_Atom": anchor.support_atom_name,
            "Warhead_Type": warhead.warhead_type,
            "Compatibility": compatibility_message,
            "Canonical_SMILES": canonical_smiles,
            "Atom_Typing": receptor.atom_typing_version,
            "Gradient_Optimized": optimize,
            "Random_Seed": random_seed,
        },
        per_pose_metadata=pose_metadata,
    )
    runtime = time.perf_counter() - started
    best_idx = int(torch.argmin(final_components.score).item())
    result: dict[str, object] = {
        "mode": "covalent",
        "output_file": str(output_path),
        "num_poses": int(selected.numel()),
        "num_representatives": int(final_coords.shape[0]),
        "best_score": float(final_components.score[best_idx].detach().cpu()),
        "best_search_energy": float(final_components.search_energy[best_idx].detach().cpu()),
        "score_units": problem.scorer.units,
        "scorer": problem.scorer.name,
        "score_semantics": "adduct_conditioned_pose_ranking",
        "warhead_type": warhead.warhead_type,
        "anchor_residue": anchor.residue_id,
        "anchor_atom": anchor.atom_name,
        "canonical_smiles": canonical_smiles,
        "covalent_bond_length": target_length,
        "optimized": optimize,
        "optimization": optimization_stats.as_dict() if optimization_stats is not None else None,
        "runtime": runtime,
        "device": str(target_device),
    }
    if verbose:
        print(
            f"covalent docking complete: {result['num_poses']} poses, "
            f"best={result['best_score']:.4f}, {runtime:.2f}s"
        )
    return result
