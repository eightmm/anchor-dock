"""Covalent residue-warhead docking on the shared AnchorDock engine."""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from .._version import __version__
from ..core.conformers import generate_conformers_and_cluster
from ..core.engine import DockingEngine
from ..core.features import ATOM_TYPING_VERSION
from ..core.io import (
    ReceptorContext,
    choose_device,
    extract_pocket_around_residue,
    file_content_fingerprint,
    load_ligand,
    receptor_context_from_mol,
    receptor_structure_fingerprint,
)
from ..core.kinematics import get_batched_rotation_matrix
from ..core.output import write_ranked_poses
from ..core.scoring import ScorerLike
from .adduct import (
    covalent_one_three_bounds,
    create_adduct_template,
    create_covalent_exclusion_mask,
    find_receptor_nucleophile_index,
    normalize_covalent_conformer,
    select_formed_bond_geometry,
)
from .anchor import (
    AnchorPoint,
    check_warhead_residue_compatibility,
    create_covalent_coordmap,
    detect_warheads,
    select_reactive_anchor,
)

SCORE_SEMANTICS = "adduct_conditioned_pose_ranking"
COVALENT_TRANSFORM_VERSION = "3"
COVALENT_RECEPTOR_TYPING_VERSION = "covalent-product-v1"


@dataclass(frozen=True)
class CovalentReceptorContext:
    """Cached reactive anchor and extracted receptor pocket."""

    anchor: AnchorPoint
    receptor: ReceptorContext


_COVALENT_CONTEXT_CACHE_MAX_SIZE = max(
    0,
    int(os.environ.get("ANCHOR_DOCK_COVALENT_CACHE_SIZE", "8")),
)
_COVALENT_CONTEXT_CACHE: OrderedDict[tuple[object, ...], CovalentReceptorContext] = OrderedDict()


def clear_covalent_context_cache() -> None:
    _COVALENT_CONTEXT_CACHE.clear()


def _product_state_receptor_context(
    receptor: ReceptorContext,
    anchor: AnchorPoint,
    receptor_nucleophile_idx: int,
    adduct: Chem.Mol,
    reactive_idx: int,
) -> tuple[ReceptorContext, dict[str, object]]:
    """Retype the bonded receptor nucleophile without mutating cached input.

    The receptor is represented separately from the ligand-side adduct, so its
    PDB-derived features otherwise describe the reactant.  This copy-on-write
    override makes the one receptor atom participating in the formed bond use
    product-state donor/acceptor semantics during scoring.
    """
    if not 0 <= receptor_nucleophile_idx < receptor.mol.GetNumAtoms():
        raise ValueError("covalent receptor nucleophile index is outside the receptor")
    if not 0 <= reactive_idx < adduct.GetNumAtoms():
        raise ValueError("covalent electrophile index is outside the adduct")

    receptor_atom = receptor.mol.GetAtomWithIdx(receptor_nucleophile_idx)
    residue = anchor.residue_name.upper()
    if receptor_atom.GetAtomicNum() != anchor.atomic_number:
        raise ValueError(
            f"covalent receptor atom for {anchor.residue_id} has element "
            f"{receptor_atom.GetAtomicNum()}, expected {anchor.atomic_number}"
        )

    try:
        xs_types = list(receptor.features["xs_types"])
        donor = receptor.features["donor"]
        acceptor = receptor.features["acceptor"]
    except KeyError as exc:
        raise ValueError(f"receptor features are missing product-state typing input: {exc.args[0]}") from exc
    if not isinstance(donor, torch.Tensor) or not isinstance(acceptor, torch.Tensor):
        raise TypeError("receptor donor and acceptor features must be tensors")
    if (
        len(xs_types) != receptor.mol.GetNumAtoms()
        or donor.numel() != len(xs_types)
        or acceptor.numel() != len(xs_types)
    ):
        raise ValueError("receptor atom features do not match receptor atom count")

    before_type = str(xs_types[receptor_nucleophile_idx])
    expected_prefix = {"CYS": "S_", "SER": "O_", "THR": "O_", "TYR": "O_", "LYS": "N_", "HIS": "N_"}
    prefix = expected_prefix.get(residue)
    if prefix is None:
        raise ValueError(f"no covalent product-state typing rule for residue {residue}")
    if not before_type.startswith(prefix):
        raise ValueError(
            f"reactive {residue} atom has unsupported pre-adduct XS type {before_type!r}; expected {prefix}*"
        )

    if residue == "CYS":
        after_type, after_donor, after_acceptor = "S_P", 0.0, 0.0
        rule = "cys_thioether_non_hbonding"
    elif residue in {"SER", "THR", "TYR"}:
        after_type, after_donor, after_acceptor = "O_A", 0.0, 1.0
        rule = "oxygen_ether_or_ester_acceptor"
    elif residue == "HIS":
        after_type, after_donor, after_acceptor = "N_P", 0.0, 0.0
        rule = "histidine_bonded_nitrogen_non_hbonding"
    else:
        electrophile = adduct.GetAtomWithIdx(reactive_idx)
        conjugated_heteroatom = any(
            bond.GetBondTypeAsDouble() >= 2.0 and bond.GetOtherAtom(electrophile).GetAtomicNum() in {7, 8, 16}
            for bond in electrophile.GetBonds()
        )
        if electrophile.GetAtomicNum() in {15, 16} or conjugated_heteroatom:
            after_type, after_donor, after_acceptor = "N_D", 1.0, 0.0
            rule = "lys_conjugated_or_heteroatom_electrophile"
        elif electrophile.GetAtomicNum() == 6:
            after_type, after_donor, after_acceptor = "N_DA", 1.0, 1.0
            rule = "lys_saturated_carbon_electrophile"
        else:
            raise ValueError(
                f"no validated LYS product-state rule for electrophile element {electrophile.GetAtomicNum()}"
            )

    features = {
        key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in receptor.features.items()
    }
    product_donor = features["donor"]
    product_acceptor = features["acceptor"]
    assert isinstance(product_donor, torch.Tensor) and isinstance(product_acceptor, torch.Tensor)
    product_donor[receptor_nucleophile_idx] = after_donor
    product_acceptor[receptor_nucleophile_idx] = after_acceptor
    xs_types[receptor_nucleophile_idx] = after_type
    features["xs_types"] = tuple(xs_types)
    product_typing = f"{receptor.atom_typing_version}+{COVALENT_RECEPTOR_TYPING_VERSION}"
    features["typing_version"] = product_typing

    change: dict[str, object] = {
        "atom_index": receptor_nucleophile_idx,
        "residue": anchor.residue_id,
        "atom_name": anchor.atom_name,
        "before": {
            "xs_type": before_type,
            "donor": bool(float(donor[receptor_nucleophile_idx].detach().cpu()) >= 0.5),
            "acceptor": bool(float(acceptor[receptor_nucleophile_idx].detach().cpu()) >= 0.5),
        },
        "after": {
            "xs_type": after_type,
            "donor": bool(after_donor),
            "acceptor": bool(after_acceptor),
        },
        "rule": rule,
    }
    product_context = replace(
        receptor,
        features=features,
        structure_fingerprint=receptor_structure_fingerprint(receptor.coords, features),
        atom_typing_version=product_typing,
    )
    return product_context, change


def _prepare_covalent_receptor(
    protein_pdb: str | os.PathLike[str],
    reactive_residue: str | None,
    pocket_cutoff: float,
    include_heteroatoms: bool,
    device: str | torch.device | None,
) -> CovalentReceptorContext:
    target_device = choose_device(device)
    path = os.path.abspath(os.fspath(protein_pdb))
    source_fingerprint = file_content_fingerprint(path)
    key = (
        path,
        source_fingerprint,
        reactive_residue.strip() if reactive_residue is not None else None,
        float(pocket_cutoff),
        bool(include_heteroatoms),
        str(target_device),
        ATOM_TYPING_VERSION,
    )
    cached = _COVALENT_CONTEXT_CACHE.get(key)
    if cached is not None:
        _COVALENT_CONTEXT_CACHE.move_to_end(key)
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
    receptor = receptor_context_from_mol(
        pocket,
        target_device,
        source_path=path,
        source_fingerprint=source_fingerprint,
    )
    context = CovalentReceptorContext(anchor, receptor)
    if _COVALENT_CONTEXT_CACHE_MAX_SIZE:
        for stale_key in list(_COVALENT_CONTEXT_CACHE):
            if stale_key[0] == path and stale_key[5] == str(target_device) and stale_key != key:
                _COVALENT_CONTEXT_CACHE.pop(stale_key)
        _COVALENT_CONTEXT_CACHE[key] = context
        while len(_COVALENT_CONTEXT_CACHE) > _COVALENT_CONTEXT_CACHE_MAX_SIZE:
            _COVALENT_CONTEXT_CACHE.popitem(last=False)
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
    output_dir: str | os.PathLike[str] = "output_predictions",
    *,
    pocket_cutoff: float = 12.0,
    include_heteroatoms: bool = True,
    num_confs: int = 1000,
    rmsd_threshold: float = 1.0,
    rotation_scan_step: int = 30,
    rotation_top_k: int = 50,
    optimize: bool = False,
    optimizer: Literal["adam", "adamw", "lbfgs"] = "adam",
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
    context = _prepare_covalent_receptor(protein_pdb, reactive_residue, pocket_cutoff, include_heteroatoms, device)
    anchor = context.anchor
    receptor = context.receptor
    target_device = receptor.device

    ligand, canonical_smiles = load_ligand(query_ligand, add_hydrogens=False)
    canonical_ligand_atoms = ligand.GetNumAtoms()
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

    input_rotatable_bonds = int(rdMolDescriptors.CalcNumRotatableBonds(ligand))
    num_rotatable_bonds = input_rotatable_bonds if torsion_penalty else 0
    adduct, support_idx, nucleophile_idx, reactive_idx = create_adduct_template(ligand, warhead, anchor)
    formed_bond = select_formed_bond_geometry(
        adduct,
        nucleophile_idx,
        reactive_idx,
        preferred_carbon_length=anchor.bond_length,
    )
    adduct_smiles = Chem.MolToSmiles(Chem.RemoveHs(adduct), canonical=True, isomericSmiles=True)
    coord_map = create_covalent_coordmap(support_idx, nucleophile_idx, anchor)
    geometry_rejected = 0

    def normalize_before_clustering(molecule: Chem.Mol, conformer_id: int) -> bool:
        nonlocal geometry_rejected
        if formed_bond.electrophile_atomic_number != 6:
            if not AllChem.UFFHasAllMoleculeParams(molecule):
                geometry_rejected += 1
                return False
            force_field = AllChem.UFFGetMoleculeForceField(molecule, confId=conformer_id)
            force_field.AddFixedPoint(support_idx)
            force_field.AddFixedPoint(nucleophile_idx)
            force_field.Initialize()
            force_field.Minimize(maxIts=500)
        valid = normalize_covalent_conformer(
            molecule,
            conformer_id,
            support_idx=support_idx,
            nucleophile_idx=nucleophile_idx,
            reactive_idx=reactive_idx,
            bond_length=formed_bond.target,
        )
        if not valid:
            geometry_rejected += 1
        return valid

    adduct, representative_ids = generate_conformers_and_cluster(
        adduct,
        target_device,
        num_confs=num_confs,
        rmsd_threshold=rmsd_threshold,
        coord_map=coord_map,
        exact_constraints_before_clustering=True,
        add_hydrogens=False,
        random_seed=random_seed,
        distance_bound_exempt_pairs={(support_idx, nucleophile_idx)},
        conformer_preprocessor=normalize_before_clustering,
    )
    if not representative_ids:
        raise RuntimeError("covalent conformer generation produced no representative poses")
    coords = torch.stack(
        [
            torch.tensor(adduct.GetConformer(conf_id).GetPositions(), dtype=torch.float32)
            for conf_id in representative_ids
        ]
    ).to(target_device)

    receptor_nucleophile_idx = find_receptor_nucleophile_index(receptor.mol, anchor)
    reactant_receptor_structure_fingerprint = receptor.structure_fingerprint
    reactant_receptor_atom_typing_version = receptor.atom_typing_version
    receptor, receptor_typing_change = _product_state_receptor_context(
        receptor,
        anchor,
        receptor_nucleophile_idx,
        adduct,
        reactive_idx,
    )
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
    score_rotatable_bonds = problem.scorer.effective_rotatable_bonds
    torsion_penalty_applied = problem.scorer.torsion_penalty_applied

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
    intramolecular_reference = float(final_components.intramolecular_reference.detach().cpu())
    optimization_applied = optimization_stats is not None and optimization_stats.maximum_steps > 0
    optimization_improved = (
        optimization_stats is not None
        and optimization_stats.final_best_energy < optimization_stats.initial_best_energy - 1e-12
    )
    search_parameters = {
        "pocket_cutoff": pocket_cutoff,
        "include_heteroatoms": include_heteroatoms,
        "num_confs": num_confs,
        "rmsd_threshold": rmsd_threshold,
        "rotation_scan_step": rotation_scan_step,
        "rotation_top_k": rotation_top_k,
        "optimize": optimize,
        "optimizer": optimizer,
        "opt_steps": opt_steps,
        "opt_lr": opt_lr,
        "opt_batch_size": opt_batch_size,
        "torsion_penalty_requested": torsion_penalty,
        "top_k": top_k,
        "warhead_index": warhead_index,
        "strict_compatibility": strict_compatibility,
        "random_seed": random_seed,
    }

    bond_lengths = torch.linalg.vector_norm(
        final_coords[:, reactive_idx] - final_coords[:, nucleophile_idx],
        dim=1,
    )
    target_length = formed_bond.target
    if not torch.allclose(bond_lengths, torch.full_like(bond_lengths, target_length), atol=1e-4, rtol=0.0):
        raise RuntimeError("covalent bond-length invariant was violated during optimization")
    one_three_distances = torch.linalg.vector_norm(
        final_coords[:, reactive_idx] - final_coords[:, support_idx],
        dim=1,
    )
    one_three_lower, one_three_upper = covalent_one_three_bounds(adduct, support_idx, reactive_idx)
    if not torch.all((one_three_distances >= one_three_lower - 0.05) & (one_three_distances <= one_three_upper + 0.05)):
        raise RuntimeError("covalent support-reactive 1-3 geometry invariant was violated")

    output_name = "covalent_poses_all.sdf" if top_k is None else f"covalent_pose_top{top_k}.sdf"
    output_path = Path(output_dir) / output_name
    pose_metadata = [
        {
            "Warhead_Type": warhead.warhead_type,
            "Reactive_Atom_Index": reactive_idx,
            "Covalent_Bond_Length": f"{float(length):.6f}",
            "Support_Reactive_Distance": f"{float(one_three):.6f}",
        }
        for length, one_three in zip(
            bond_lengths.detach().cpu(),
            one_three_distances.detach().cpu(),
            strict=True,
        )
    ]
    selected = write_ranked_poses(
        adduct,
        final_coords,
        final_components.score,
        str(output_path),
        scorer_name=problem.scorer.name,
        score_units=problem.scorer.units,
        score_semantics=SCORE_SEMANTICS,
        scorer_fingerprint=problem.scorer.fingerprint,
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
            "Covalent_Transform_Version": COVALENT_TRANSFORM_VERSION,
            "Covalent_Bond_Target_Source": formed_bond.source,
            "Covalent_Bond_Lower_Bound": f"{formed_bond.lower_bound:.6f}",
            "Covalent_Bond_Upper_Bound": f"{formed_bond.upper_bound:.6f}",
            "Nucleophile_Atomic_Number": formed_bond.nucleophile_atomic_number,
            "Electrophile_Atomic_Number": formed_bond.electrophile_atomic_number,
            "Covalent_1_3_Lower_Bound": f"{one_three_lower:.6f}",
            "Covalent_1_3_Upper_Bound": f"{one_three_upper:.6f}",
            "Canonical_Ligand_Atoms": canonical_ligand_atoms,
            "Canonical_Ligand_Reactive_Atom_Index": warhead.reactive_atom_idx,
            "Adduct_Reactive_Atom_Index": reactive_idx,
            "Ligand_Index_Space": "canonical_smiles",
            "Pseudo_Support_Atom_Index": support_idx,
            "Pseudo_Nucleophile_Atom_Index": nucleophile_idx,
            "Adduct_SMILES": adduct_smiles,
            "Compatibility": compatibility_message,
            "Canonical_SMILES": canonical_smiles,
            "Receptor_Structure_Fingerprint": receptor.structure_fingerprint,
            "Receptor_Reactant_Structure_Fingerprint": reactant_receptor_structure_fingerprint,
            "Receptor_Structure_Scope": "extracted_pocket",
            "Receptor_Source_Fingerprint": receptor.source_fingerprint,
            "Torsion_Penalty_Requested": torsion_penalty,
            "Torsion_Penalty_Applied": torsion_penalty_applied,
            "Input_Ligand_Rotatable_Bonds": input_rotatable_bonds,
            "Score_Rotatable_Bonds": score_rotatable_bonds,
            "Intramolecular_Reference": intramolecular_reference,
            "Atom_Typing": receptor.atom_typing_version,
            "Receptor_Atom_Typing_Version": receptor.atom_typing_version,
            "Receptor_Reactant_Atom_Typing_Version": reactant_receptor_atom_typing_version,
            "Covalent_Receptor_Typing_State": "product",
            "Covalent_Receptor_Typing_Version": COVALENT_RECEPTOR_TYPING_VERSION,
            "Covalent_Receptor_Typing_Changes": json.dumps(
                [receptor_typing_change], sort_keys=True, separators=(",", ":")
            ),
            "Optimization_Requested": optimize,
            "Optimization_Applied": optimization_applied,
            "Optimization_Improved": optimization_improved,
            "Optimizer": optimizer,
            "Optimization_Steps_Requested": opt_steps,
            "Optimization_Learning_Rate": opt_lr,
            "Search_Parameters": json.dumps(search_parameters, sort_keys=True, separators=(",", ":")),
            "Random_Seed": random_seed,
        },
        per_pose_metadata=pose_metadata,
    )
    runtime = time.perf_counter() - started
    best_idx = int(torch.argmin(final_components.score).item())
    result: dict[str, object] = {
        "mode": "covalent",
        "anchor_dock_version": __version__,
        "output_file": str(output_path),
        "num_poses": int(selected.numel()),
        "num_conformers": int(num_confs),
        "num_representatives": int(final_coords.shape[0]),
        "best_score": float(final_components.score[best_idx].detach().cpu()),
        "best_search_energy": float(final_components.search_energy[best_idx].detach().cpu()),
        "score_units": problem.scorer.units,
        "scorer": problem.scorer.name,
        "scorer_fingerprint": problem.scorer.fingerprint,
        "score_semantics": SCORE_SEMANTICS,
        "warhead_type": warhead.warhead_type,
        "covalent_transform_version": COVALENT_TRANSFORM_VERSION,
        "covalent_bond_target_source": formed_bond.source,
        "covalent_bond_distance_bounds": [formed_bond.lower_bound, formed_bond.upper_bound],
        "nucleophile_atomic_number": formed_bond.nucleophile_atomic_number,
        "electrophile_atomic_number": formed_bond.electrophile_atomic_number,
        "canonical_ligand_atoms": canonical_ligand_atoms,
        "canonical_ligand_reactive_atom_index": warhead.reactive_atom_idx,
        "adduct_reactive_atom_index": reactive_idx,
        "pseudo_support_atom_index": support_idx,
        "pseudo_nucleophile_atom_index": nucleophile_idx,
        "adduct_smiles": adduct_smiles,
        "anchor_residue": anchor.residue_id,
        "anchor_atom": anchor.atom_name,
        "canonical_smiles": canonical_smiles,
        "receptor_structure_fingerprint": receptor.structure_fingerprint,
        "receptor_reactant_structure_fingerprint": reactant_receptor_structure_fingerprint,
        "receptor_structure_scope": "extracted_pocket",
        "receptor_source_fingerprint": receptor.source_fingerprint,
        "receptor_atom_typing_version": receptor.atom_typing_version,
        "receptor_reactant_atom_typing_version": reactant_receptor_atom_typing_version,
        "covalent_receptor_typing_state": "product",
        "covalent_receptor_typing_version": COVALENT_RECEPTOR_TYPING_VERSION,
        "covalent_receptor_typing_changes": [receptor_typing_change],
        "torsion_penalty_requested": torsion_penalty,
        "torsion_penalty_applied": torsion_penalty_applied,
        "input_ligand_rotatable_bonds": input_rotatable_bonds,
        "score_rotatable_bonds": score_rotatable_bonds,
        "intramolecular_reference": intramolecular_reference,
        "covalent_bond_length": target_length,
        "support_reactive_distance_bounds": [one_three_lower, one_three_upper],
        "geometry_rejected_conformers": geometry_rejected,
        "optimization_requested": optimize,
        "optimization_applied": optimization_applied,
        "optimization_improved": optimization_improved,
        "optimization_config": {
            "optimizer": optimizer,
            "steps": opt_steps,
            "learning_rate": opt_lr,
            "batch_size": opt_batch_size,
        },
        "optimized": optimization_applied,
        "optimization": optimization_stats.as_dict() if optimization_stats is not None else None,
        "search_parameters": search_parameters,
        "runtime": runtime,
        "device": str(target_device),
    }
    if verbose:
        print(
            f"covalent docking complete: {result['num_poses']} poses, best={result['best_score']:.4f}, {runtime:.2f}s"
        )
    return result
