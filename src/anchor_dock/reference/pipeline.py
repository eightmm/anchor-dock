"""Reference-ligand MCS docking on the shared AnchorDock engine."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Geometry import Point3D

from ..core.conformers import generate_conformers_and_cluster
from ..core.engine import DockingEngine, PreparedDockingProblem
from ..core.io import load_ligand, load_receptor_context, load_reference_ligand
from ..core.output import write_ranked_poses
from ..core.scoring import RawScoreComponents, ScorerLike
from .mcs import MCSSelection, select_mcs_mappings
from .relax import RelaxationResult, relax_pose_with_fixed_core


@dataclass
class _MappingRun:
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
    output_dir: str | os.PathLike[str] = "anchor_dock_reference",
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
    optimizer: Literal["adam", "lbfgs"] = "adam",
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
    num_rotatable_bonds = (
        int(rdMolDescriptors.CalcNumRotatableBonds(query)) if torsion_penalty else 0
    )
    engine = DockingEngine(
        scorer,
        device=receptor.device,
        optimizer=optimizer,
        num_steps=opt_steps,
        learning_rate=opt_lr,
        batch_size=opt_batch_size,
    )

    runs: list[_MappingRun] = []
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
        except RuntimeError:
            if verbose:
                print(f"reference mapping {mapping_index}: conformer generation failed")
            continue
        if not representative_ids:
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
            pose_coords.append(
                torch.tensor(mapped_mol.GetConformer(conformer_id).GetPositions(), dtype=torch.float32)
            )
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
        raise RuntimeError("no conformers were generated for any MCS mapping")

    combined_initial_raw = _cat_raw([run.initial_raw for run in runs])
    combined_final_raw = _cat_raw([run.final_raw for run in runs])
    best_search_index = torch.argmin(combined_final_raw.search_energy)
    intramolecular_reference = combined_final_raw.intramolecular[best_search_index].detach()
    reporting_scorer = runs[0].problem.scorer
    initial_components = reporting_scorer.report(combined_initial_raw, intramolecular_reference)
    final_components = reporting_scorer.report(combined_final_raw, intramolecular_reference)

    combined_coords = torch.cat([run.final_coords for run in runs])
    pose_ids: list[str] = []
    per_pose_metadata: list[dict[str, object]] = []
    all_relaxation: list[RelaxationResult] = []
    for mapping_index, run in enumerate(runs, start=1):
        for pose_index, relaxation_result in enumerate(run.relaxation, start=1):
            pose_ids.append(f"m{mapping_index:03d}_p{pose_index:04d}")
            per_pose_metadata.append(
                {
                    "MCS_Position": mapping_index,
                    "MCS_Size": len(run.mapping),
                    "Relaxation_Applied": relaxation_result.applied,
                    "Relaxation_Method": relaxation_result.method,
                    "Relaxation_Message": relaxation_result.message,
                }
            )
            all_relaxation.append(relaxation_result)
    relaxation_methods, relaxation_messages = _relaxation_summary(all_relaxation)

    output_path = Path(output_dir) / "reference_poses.sdf"
    selected = write_ranked_poses(
        runs[0].mol,
        combined_coords,
        final_components.score,
        str(output_path),
        scorer_name=reporting_scorer.name,
        score_units=reporting_scorer.units,
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
            "MCS_Positions": len(runs),
            "MCS_Simple_Size": selection.simple_size,
            "MCS_Cross_Size": selection.cross_size,
            "Canonical_SMILES": canonical_smiles,
            "Atom_Typing": receptor.atom_typing_version,
            "Relaxation_Requested": relax,
            "Relaxation_Methods": relaxation_methods,
            "Relaxation_Summary": relaxation_messages,
            "Gradient_Optimized": optimize,
            "Freeze_Anchor": freeze_anchor,
            "Random_Seed": random_seed,
        },
        per_pose_metadata=per_pose_metadata,
    )
    runtime = time.perf_counter() - started
    order = torch.argsort(final_components.score)
    best_idx = int(order[0].item())
    result: dict[str, object] = {
        "mode": "reference",
        "output_file": str(output_path),
        "num_poses": int(selected.numel()),
        "num_representatives": int(combined_coords.shape[0]),
        "best_score": float(final_components.score[best_idx].detach().cpu()),
        "best_search_energy": float(final_components.search_energy[best_idx].detach().cpu()),
        "score_units": reporting_scorer.units,
        "scorer": reporting_scorer.name,
        "score_semantics": "anchor-conditioned_pose_ranking",
        "mcs_mode": selection.mode,
        "mcs_positions": len(runs),
        "mcs_size": max(len(run.mapping) for run in runs),
        "canonical_smiles": canonical_smiles,
        "runtime": runtime,
        "device": str(receptor.device),
        "relaxation_methods": relaxation_methods,
        "optimized": optimize,
    }
    if verbose:
        print(
            f"reference docking complete: {result['num_poses']} poses, "
            f"best={result['best_score']:.4f}, {runtime:.2f}s"
        )
    return result
