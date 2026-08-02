"""Reference-ligand MCS docking on the shared AnchorDock engine."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Geometry import Point3D

from .._version import __version__
from ..core.conformers import generate_conformers_and_cluster
from ..core.engine import DockingEngine, PreparedDockingProblem
from ..core.io import load_ligand, load_receptor_context, load_reference_ligand
from ..core.output import write_ranked_poses
from ..core.scoring import RawScoreComponents, ScorerLike
from .mcs import MCSSelection, select_mcs_mappings
from .relax import RelaxationResult, relax_pose_with_fixed_core

SCORE_SEMANTICS = "anchor-conditioned_pose_ranking"


@dataclass
class _MappingRun:
    selection_index: int
    mol: Chem.Mol
    mapping: tuple[tuple[int, int], ...]
    initial_coords: torch.Tensor
    final_coords: torch.Tensor
    initial_raw: RawScoreComponents
    final_raw: RawScoreComponents
    problem: PreparedDockingProblem
    relaxation: list[RelaxationResult]
    optimization_stats: dict[str, object] | None


def _coordinate_map(reference: Chem.Mol, mapping: tuple[tuple[int, int], ...]) -> dict[int, Point3D]:
    conformer = reference.GetConformer()
    result: dict[int, Point3D] = {}
    for reference_idx, query_idx in mapping:
        position = conformer.GetAtomPosition(reference_idx)
        result[query_idx] = Point3D(position.x, position.y, position.z)
    return result


def _restore_anchor_coordinates(
    mol: Chem.Mol,
    conformer_id: int,
    coord_map: dict[int, Point3D],
) -> None:
    conformer = mol.GetConformer(conformer_id)
    for atom_idx, position in coord_map.items():
        conformer.SetAtomPosition(atom_idx, Point3D(position.x, position.y, position.z))


def _relaxation_summary(results: list[RelaxationResult]) -> tuple[str, str]:
    methods: dict[str, int] = {}
    messages: dict[str, int] = {}
    for result in results:
        methods[result.method] = methods.get(result.method, 0) + 1
        messages[result.message] = messages.get(result.message, 0) + 1
    method_summary = ",".join(f"{name}:{count}" for name, count in sorted(methods.items()))
    message_summary = "; ".join(f"{name} ({count})" for name, count in sorted(messages.items()))
    return method_summary or "none", message_summary or "none"


def _cat_raw(values: list[RawScoreComponents]) -> RawScoreComponents:
    return RawScoreComponents(
        intermolecular=torch.cat([value.intermolecular for value in values]),
        intramolecular=torch.cat([value.intramolecular for value in values]),
        search_energy=torch.cat([value.search_energy for value in values]),
    )


def dock_reference(
    protein_pdb: str | os.PathLike[str],
    reference_ligand: str | os.PathLike[str],
    query_ligand: str | os.PathLike[str] | Chem.Mol,
    output_dir: str | os.PathLike[str] = "output_predictions",
    *,
    num_confs: int = 1000,
    rmsd_threshold: float = 1.0,
    mcs_mode: Literal["auto", "single", "multi", "cross"] = "auto",
    min_mcs_atoms: int = 3,
    min_fragment_size: int = 5,
    max_fragments: int = 3,
    max_mappings: int = 64,
    mcs_timeout: int = 10,
    match_chirality: bool = False,
    relax: bool = True,
    relax_max_iters: int = 500,
    optimize: bool = False,
    optimizer: Literal["adam", "adamw", "lbfgs"] = "adam",
    opt_steps: int = 100,
    opt_lr: float = 0.05,
    opt_batch_size: int = 128,
    freeze_anchor: bool = True,
    scorer: ScorerLike = "vina",
    torsion_penalty: bool = True,
    top_k: int | None = None,
    random_seed: int = 42,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    """Dock one query by transferring all selected MCS anchors from a reference.

    Every selected mapping is independently embedded and refined, then all poses
    are pooled and reported against one intramolecular reference. This makes
    scores comparable across symmetry-related and cross-fragment mappings.
    """
    started = time.perf_counter()
    receptor = load_receptor_context(protein_pdb, device)
    reference = Chem.RemoveHs(load_reference_ligand(reference_ligand))
    reference_canonical_smiles = Chem.MolToSmiles(reference, canonical=True, isomericSmiles=True)
    reference_fingerprint = "sha256:" + hashlib.sha256(Chem.MolToMolBlock(reference).encode("utf-8")).hexdigest()
    query, canonical_smiles = load_ligand(query_ligand, add_hydrogens=False)
    selection: MCSSelection = select_mcs_mappings(
        reference,
        query,
        mode=mcs_mode,
        min_atoms=min_mcs_atoms,
        min_fragment_size=min_fragment_size,
        max_fragments=max_fragments,
        timeout=mcs_timeout,
        max_mappings=max_mappings,
        match_chirality=match_chirality,
    )
    input_rotatable_bonds = int(rdMolDescriptors.CalcNumRotatableBonds(query))
    num_rotatable_bonds = input_rotatable_bonds if torsion_penalty else 0
    engine = DockingEngine(
        scorer,
        device=receptor.device,
        optimizer=optimizer,
        num_steps=opt_steps,
        learning_rate=opt_lr,
        batch_size=opt_batch_size,
    )

    runs: list[_MappingRun] = []
    failed_mappings: list[dict[str, object]] = []
    for mapping_index, mapping in enumerate(selection.mappings, start=1):
        coord_map = _coordinate_map(reference, mapping)
        try:
            mapped_mol, representative_ids = generate_conformers_and_cluster(
                query,
                receptor.device,
                num_confs=num_confs,
                rmsd_threshold=rmsd_threshold,
                coord_map=coord_map,
                exact_constraints_before_clustering=True,
                add_hydrogens=True,
                random_seed=random_seed + mapping_index - 1,
            )
        except RuntimeError as exc:
            failed_mappings.append(
                {
                    "selection_index": mapping_index,
                    "random_seed": random_seed + mapping_index - 1,
                    "reason": str(exc),
                    "mapping": [list(pair) for pair in mapping],
                }
            )
            if verbose:
                print(f"reference mapping {mapping_index}: conformer generation failed: {exc}")
            continue
        if not representative_ids:
            failed_mappings.append(
                {
                    "selection_index": mapping_index,
                    "random_seed": random_seed + mapping_index - 1,
                    "reason": "conformer clustering produced no representatives",
                    "mapping": [list(pair) for pair in mapping],
                }
            )
            continue

        fixed_indices = {query_idx for _, query_idx in mapping}
        mmff_properties = None
        if relax:
            try:
                mmff_properties = AllChem.MMFFGetMoleculeProperties(mapped_mol)
            except Exception:
                mmff_properties = None
        relaxation_results: list[RelaxationResult] = []
        pose_coords: list[torch.Tensor] = []
        for conformer_id in representative_ids:
            _restore_anchor_coordinates(mapped_mol, conformer_id, coord_map)
            relaxation_result = relax_pose_with_fixed_core(
                mapped_mol,
                conformer_id,
                fixed_indices,
                requested=relax,
                max_iters=relax_max_iters,
                mmff_props=mmff_properties,
            )
            # Fixed points should remain unchanged, but reassert exact coordinates
            # so exported anchors are independent of force-field tolerance.
            _restore_anchor_coordinates(mapped_mol, conformer_id, coord_map)
            relaxation_results.append(relaxation_result)
            pose_coords.append(torch.tensor(mapped_mol.GetConformer(conformer_id).GetPositions(), dtype=torch.float32))
        initial_coords = torch.stack(pose_coords).to(receptor.device)
        problem = engine.prepare(
            mapped_mol,
            receptor,
            anchor_indices=tuple(sorted(fixed_indices)),
            num_rotatable_bonds=num_rotatable_bonds,
        )
        initial_raw = problem.scorer.raw_components(initial_coords)
        if optimize:
            final_coords, stats = engine.optimize_anchored(
                problem,
                initial_coords,
                freeze_anchor=freeze_anchor,
            )
            optimization_stats: dict[str, object] | None = stats.as_dict()
        else:
            final_coords = initial_coords
            optimization_stats = None
        final_raw = problem.scorer.raw_components(final_coords)
        runs.append(
            _MappingRun(
                mapping_index,
                mapped_mol,
                mapping,
                initial_coords,
                final_coords,
                initial_raw,
                final_raw,
                problem,
                relaxation_results,
                optimization_stats,
            )
        )
        if verbose:
            print(
                f"reference mapping {mapping_index}/{len(selection.mappings)}: "
                f"{len(representative_ids)} representative poses"
            )

    if not runs:
        details = json.dumps(failed_mappings, sort_keys=True, separators=(",", ":"))
        raise RuntimeError(f"no conformers were generated for any MCS mapping: {details}")

    combined_initial_raw = _cat_raw([run.initial_raw for run in runs])
    combined_final_raw = _cat_raw([run.final_raw for run in runs])
    best_search_index = torch.argmin(combined_final_raw.search_energy)
    intramolecular_reference = combined_final_raw.intramolecular[best_search_index].detach()
    reporting_scorer = runs[0].problem.scorer
    score_rotatable_bonds = reporting_scorer.effective_rotatable_bonds
    torsion_penalty_applied = reporting_scorer.torsion_penalty_applied
    initial_components = reporting_scorer.report(combined_initial_raw, intramolecular_reference)
    final_components = reporting_scorer.report(combined_final_raw, intramolecular_reference)
    intramolecular_reference_value = float(final_components.intramolecular_reference.detach().cpu())
    optimization_details = {
        str(run.selection_index): run.optimization_stats for run in runs if run.optimization_stats is not None
    }
    optimization_applied = any(int(stats["maximum_steps"]) > 0 for stats in optimization_details.values())
    optimization_improved = any(
        float(stats["final_best_energy"]) < float(stats["initial_best_energy"]) - 1e-12
        for stats in optimization_details.values()
    )
    search_parameters = {
        "num_confs": num_confs,
        "rmsd_threshold": rmsd_threshold,
        "mcs_mode_requested": mcs_mode,
        "min_mcs_atoms": min_mcs_atoms,
        "min_fragment_size": min_fragment_size,
        "max_fragments": max_fragments,
        "max_mappings": max_mappings,
        "mcs_timeout": mcs_timeout,
        "match_chirality": match_chirality,
        "relax": relax,
        "relax_max_iters": relax_max_iters,
        "optimize": optimize,
        "optimizer": optimizer,
        "opt_steps": opt_steps,
        "opt_lr": opt_lr,
        "opt_batch_size": opt_batch_size,
        "freeze_anchor": freeze_anchor,
        "torsion_penalty_requested": torsion_penalty,
        "top_k": top_k,
        "random_seed": random_seed,
    }

    combined_coords = torch.cat([run.final_coords for run in runs])
    pose_ids: list[str] = []
    per_pose_metadata: list[dict[str, object]] = []
    all_relaxation: list[RelaxationResult] = []
    for run in runs:
        for pose_index, relaxation_result in enumerate(run.relaxation, start=1):
            pose_ids.append(f"m{run.selection_index:03d}_p{pose_index:04d}")
            per_pose_metadata.append(
                {
                    "MCS_Position": run.selection_index,
                    "MCS_Size": len(run.mapping),
                    "MCS_Mapping": json.dumps(run.mapping, separators=(",", ":")),
                    "MCS_Query_Index_Space": "canonical_ligand",
                    "MCS_Reference_Index_Space": "reference_heavy_atom_after_remove_hs",
                    "Mapping_Random_Seed": random_seed + run.selection_index - 1,
                    "Relaxation_Applied": relaxation_result.applied,
                    "Relaxation_Method": relaxation_result.method,
                    "Relaxation_Message": relaxation_result.message,
                }
            )
            all_relaxation.append(relaxation_result)
    relaxation_methods, relaxation_messages = _relaxation_summary(all_relaxation)

    output_path = Path(output_dir) / "predicted_poses.sdf"
    selected = write_ranked_poses(
        runs[0].mol,
        combined_coords,
        final_components.score,
        str(output_path),
        scorer_name=reporting_scorer.name,
        score_units=reporting_scorer.units,
        score_semantics=SCORE_SEMANTICS,
        scorer_fingerprint=reporting_scorer.fingerprint,
        search_energies=final_components.search_energy,
        initial_scores=initial_components.score,
        pose_ids=pose_ids,
        top_k=top_k,
        molecule_metadata={
            "Mode": "reference",
            "Anchor_Strategy": "reference_mcs",
            "MCS_Mode": selection.mode,
            "MCS_Mode_Requested": mcs_mode,
            "MCS_Reason": selection.reason,
            "MCS_Candidate_Complete": selection.candidate_complete,
            "MCS_Max_Size_Proven": selection.max_size_proven,
            "MCS_Candidate_Limit": selection.candidate_limit,
            "MCS_Positions": len(runs),
            "MCS_Positions_Attempted": len(selection.mappings),
            "MCS_Positions_Selected": ",".join(str(run.selection_index) for run in runs),
            "MCS_Failed": len(failed_mappings),
            "MCS_Failure_Details": json.dumps(failed_mappings, sort_keys=True),
            "Reference_Canonical_SMILES": reference_canonical_smiles,
            "Reference_Structure_Fingerprint": reference_fingerprint,
            "Receptor_Structure_Fingerprint": receptor.structure_fingerprint,
            "Receptor_Structure_Scope": "input_receptor",
            "Receptor_Source_Fingerprint": receptor.source_fingerprint,
            "MCS_Simple_Size": selection.simple_size,
            "MCS_Cross_Size": selection.cross_size,
            "Canonical_SMILES": canonical_smiles,
            "Torsion_Penalty_Requested": torsion_penalty,
            "Torsion_Penalty_Applied": torsion_penalty_applied,
            "Input_Ligand_Rotatable_Bonds": input_rotatable_bonds,
            "Score_Rotatable_Bonds": score_rotatable_bonds,
            "Intramolecular_Reference": intramolecular_reference_value,
            "Atom_Typing": receptor.atom_typing_version,
            "Relaxation_Requested": relax,
            "Relaxation_Methods": relaxation_methods,
            "Relaxation_Summary": relaxation_messages,
            "Optimization_Requested": optimize,
            "Optimization_Applied": optimization_applied,
            "Optimization_Improved": optimization_improved,
            "Optimizer": optimizer,
            "Optimization_Steps_Requested": opt_steps,
            "Optimization_Learning_Rate": opt_lr,
            "Search_Parameters": json.dumps(search_parameters, sort_keys=True, separators=(",", ":")),
            "Freeze_Anchor": freeze_anchor,
            "Random_Seed": random_seed,
        },
        per_pose_metadata=per_pose_metadata,
    )
    runtime = time.perf_counter() - started
    best_idx = int(torch.argmin(final_components.score).item())
    best_position = runs[0].selection_index
    offset = 0
    for run in runs:
        next_offset = offset + run.final_coords.shape[0]
        if offset <= best_idx < next_offset:
            best_position = run.selection_index
            break
        offset = next_offset
    result: dict[str, object] = {
        "mode": "reference",
        "anchor_dock_version": __version__,
        "output_file": str(output_path),
        "num_poses": int(selected.numel()),
        "num_conformers": int(num_confs),
        "num_representatives": int(combined_coords.shape[0]),
        "best_score": float(final_components.score[best_idx].detach().cpu()),
        "best_search_energy": float(final_components.search_energy[best_idx].detach().cpu()),
        "score_units": reporting_scorer.units,
        "scorer": reporting_scorer.name,
        "scorer_fingerprint": reporting_scorer.fingerprint,
        "score_semantics": SCORE_SEMANTICS,
        "mcs_mode": selection.mode,
        "mcs_candidate_complete": selection.candidate_complete,
        "mcs_max_size_proven": selection.max_size_proven,
        "mcs_candidate_limit": selection.candidate_limit,
        "mcs_positions": len(runs),
        "mcs_positions_attempted": len(selection.mappings),
        "mcs_positions_selected": [run.selection_index for run in runs],
        "failed_mappings": failed_mappings,
        "mcs_mappings_selected": {str(run.selection_index): [list(pair) for pair in run.mapping] for run in runs},
        "mcs_size": max(len(run.mapping) for run in runs),
        "best_position": best_position,
        "canonical_smiles": canonical_smiles,
        "reference_canonical_smiles": reference_canonical_smiles,
        "reference_structure_fingerprint": reference_fingerprint,
        "receptor_structure_fingerprint": receptor.structure_fingerprint,
        "receptor_structure_scope": "input_receptor",
        "receptor_source_fingerprint": receptor.source_fingerprint,
        "torsion_penalty_requested": torsion_penalty,
        "torsion_penalty_applied": torsion_penalty_applied,
        "input_ligand_rotatable_bonds": input_rotatable_bonds,
        "score_rotatable_bonds": score_rotatable_bonds,
        "intramolecular_reference": intramolecular_reference_value,
        "runtime": runtime,
        "device": str(receptor.device),
        "relaxation_methods": relaxation_methods,
        "optimization_requested": optimize,
        "optimization_applied": optimization_applied,
        "optimization_improved": optimization_improved,
        "optimization_config": {
            "optimizer": optimizer,
            "steps": opt_steps,
            "learning_rate": opt_lr,
            "batch_size": opt_batch_size,
        },
        "optimization": optimization_details or None,
        "search_parameters": search_parameters,
        "optimized": optimization_applied,
    }
    if verbose:
        print(
            f"reference docking complete: {result['num_poses']} poses, best={result['best_score']:.4f}, {runtime:.2f}s"
        )
    return result
