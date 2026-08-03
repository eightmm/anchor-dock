"""Interaction-guided local docking pipeline."""

from __future__ import annotations

import json
import math
import os
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from .._version import __version__
from ..core.conformers import generate_conformers_and_cluster
from ..core.engine import DockingEngine
from ..core.features import ATOM_TYPING_VERSION
from ..core.geometry import sample_uniform_rotation_vectors
from ..core.io import (
    ReceptorContext,
    choose_device,
    extract_pocket_around_residue,
    extract_pocket_around_residues,
    file_content_fingerprint,
    load_ligand,
    receptor_context_from_mol,
)
from ..core.kinematics import get_batched_rotation_matrix
from ..core.optimization import OptimizationStats
from ..core.output import write_ranked_poses
from ..core.scoring import ScorerLike
from .hypotheses import (
    JointLigandHypothesis,
    enumerate_joint_hypotheses,
    pairwise_shell_feasible,
)
from .restraint import (
    flat_bottom_distance_restraint,
    flat_bottom_distance_restraint_matrix,
    interaction_distance_matrix,
    interaction_distances,
    mean_flat_bottom_distance_restraint,
)
from .selectors import (
    InteractionSelectionError,
    MatchLimitExceededError,
    select_ligand_anchors,
    select_receptor_atom,
)
from .spec import InteractionConstraint, InteractionInput, interaction_dicts, normalize_interactions

SCORE_SEMANTICS = "interaction_conditioned_local_pose_ranking"
OUTPUT_COORDINATE_DECIMALS = 4
RESTRAINT_FORMULA = "weight * relu(abs(distance-target)-tolerance)^2"
PROTONATION_LIMITATION = "exact input ligand state, receptor hydrogens removed, no protomer/tautomer enumeration"
INTERACTION_CONTEXT_CACHE_MAXSIZE = 8

# Bounded LRU cache keyed by source contents, resolved selector, pocket settings,
# device, and atom-typing version.
_INTERACTION_CONTEXT_CACHE: OrderedDict[tuple[object, ...], ReceptorContext] = OrderedDict()


def clear_interaction_context_cache() -> None:
    """Clear the cached interaction pocket contexts."""
    _INTERACTION_CONTEXT_CACHE.clear()


def preselect_candidates(
    energies: torch.Tensor,
    match_indices: torch.Tensor,
    conformer_ordinals: torch.Tensor,
    preselect_k: int,
    match_count: int,
    conformer_count: int,
) -> list[int]:
    """Stratified deterministic candidate preselector.

    Allocates preselect_k quotas round-robin across match groups; within each match,
    takes the best orientation per conformer before a second orientation of any
    conformer, using stable energy/candidate-id ties.
    """
    if energies.ndim != 1 or match_indices.ndim != 1 or conformer_ordinals.ndim != 1:
        raise ValueError("energies, match_indices, and conformer_ordinals must be one-dimensional")
    num_candidates = len(energies)
    if len(match_indices) != num_candidates or len(conformer_ordinals) != num_candidates:
        raise ValueError("preselection tensors must have the same length")
    if not torch.is_floating_point(energies) or not torch.isfinite(energies).all():
        raise ValueError("energies must be finite floating-point values")
    for name, value in (
        ("preselect_k", preselect_k),
        ("match_count", match_count),
        ("conformer_count", conformer_count),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if preselect_k > num_candidates:
        raise ValueError("preselect_k cannot exceed the candidate count")

    group_candidates: list[list[int]] = [[] for _ in range(match_count)]
    for i in range(num_candidates):
        m_idx = int(match_indices[i].item())
        c_ord = int(conformer_ordinals[i].item())
        if not 0 <= m_idx < match_count:
            raise ValueError("match_indices contains an out-of-range value")
        if not 0 <= c_ord < conformer_count:
            raise ValueError("conformer_ordinals contains an out-of-range value")
        group_candidates[m_idx].append(i)

    ordered_groups: list[list[int]] = []
    for m in range(match_count):
        candidates_in_group = group_candidates[m]
        if not candidates_in_group:
            raise ValueError(f"match group {m} has no candidates")
        conf_to_candidates: dict[int, list[int]] = {c: [] for c in range(conformer_count)}
        for cid in candidates_in_group:
            c_ord = int(conformer_ordinals[cid].item())
            conf_to_candidates[c_ord].append(cid)

        def energy_key(candidate_id: int) -> tuple[float, int]:
            return float(energies[candidate_id].item()), candidate_id

        for values in conf_to_candidates.values():
            values.sort(key=energy_key)
        winners = [values[0] for values in conf_to_candidates.values() if values]
        winners.sort(key=energy_key)
        winner_set = set(winners)
        remaining = sorted(
            (cid for cid in candidates_in_group if cid not in winner_set),
            key=energy_key,
        )
        ordered_groups.append(winners + remaining)

    selected_candidates: list[int] = []
    group_ptrs = [0] * match_count

    while len(selected_candidates) < preselect_k:
        any_added = False
        for m in range(match_count):
            if len(selected_candidates) >= preselect_k:
                break
            ptr = group_ptrs[m]
            if ptr < len(ordered_groups[m]):
                selected_candidates.append(ordered_groups[m][ptr])
                group_ptrs[m] += 1
                any_added = True
        if not any_added:
            break

    return selected_candidates


def _positive_int(name: str, value: object, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _positive_finite(name: str, value: object, *, allow_zero: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    numeric = float(value)
    outside_domain = numeric < 0.0 if allow_zero else numeric <= 0.0
    if not math.isfinite(numeric) or outside_domain:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} finite number")
    return numeric


def _aggregate_stats(stats_list: list[OptimizationStats], total_poses: int) -> OptimizationStats:
    all_steps = [value.average_steps for value in stats_list for _ in range(value.num_poses)]
    return OptimizationStats(
        average_steps=float(sum(all_steps) / len(all_steps)) if all_steps else 0.0,
        minimum_steps=min((value.minimum_steps for value in stats_list), default=0),
        maximum_steps=max((value.maximum_steps for value in stats_list), default=0),
        num_poses=total_poses,
        initial_best_energy=min((value.initial_best_energy for value in stats_list), default=0.0),
        final_best_energy=min((value.final_best_energy for value in stats_list), default=0.0),
    )


def _reject_pocket_altlocs(pocket: Chem.Mol) -> None:
    ambiguous: list[str] = []
    for atom in pocket.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None or not info.GetAltLoc().strip():
            continue
        ambiguous.append(
            f"{info.GetResidueName().strip()}{info.GetResidueNumber()}:{info.GetChainId().strip()}"
            f"/{info.GetName().strip()}:{info.GetAltLoc().strip()}"
        )
    if ambiguous:
        raise InteractionSelectionError(
            "alternate locations are not supported in the scored receptor pocket: " + ", ".join(sorted(ambiguous))
        )


def _candidate_state_cycle(
    feasible_conformers: Sequence[Sequence[int]],
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Round-robin viable hypotheses and their conformers deterministically."""
    active_hypotheses = [index for index, values in enumerate(feasible_conformers) if values]
    if not active_hypotheses:
        raise RuntimeError("no conformer and joint interaction hypothesis passed geometry preflight")
    positions = [0] * len(feasible_conformers)
    states: list[tuple[int, int, int]] = []
    while True:
        added = False
        for group_index, hypothesis_index in enumerate(active_hypotheses):
            position = positions[hypothesis_index]
            values = feasible_conformers[hypothesis_index]
            if position < len(values):
                states.append((group_index, hypothesis_index, int(values[position])))
                positions[hypothesis_index] += 1
                added = True
        if not added:
            break
    return active_hypotheses, states


def _allocate_candidate_states(
    feasible_conformers: Sequence[Sequence[int]],
    num_candidates: int,
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Allocate equal orientation quotas, cycling conformers within each hypothesis."""
    num_candidates = _positive_int("num_candidates", num_candidates)
    active_hypotheses, _ = _candidate_state_cycle(feasible_conformers)
    states: list[tuple[int, int, int]] = []
    for candidate_index in range(num_candidates):
        group_index = candidate_index % len(active_hypotheses)
        hypothesis_index = active_hypotheses[group_index]
        occurrence = candidate_index // len(active_hypotheses)
        conformers = feasible_conformers[hypothesis_index]
        conformer_ordinal = int(conformers[occurrence % len(conformers)])
        states.append((group_index, hypothesis_index, conformer_ordinal))
    return active_hypotheses, states


def _distance_matrix_by_hypothesis(
    coords: torch.Tensor,
    hypothesis_indices: Sequence[int],
    hypotheses: Sequence[JointLigandHypothesis],
    receptor_coords: torch.Tensor,
) -> torch.Tensor:
    """Calculate a ``[poses, interactions]`` matrix for heterogeneous assignments."""
    if coords.ndim != 3 or coords.shape[0] != len(hypothesis_indices):
        raise ValueError("coords and hypothesis_indices must describe the same poses")
    result = torch.empty(
        (coords.shape[0], receptor_coords.shape[0]),
        dtype=coords.dtype,
        device=coords.device,
    )
    grouped: dict[int, list[int]] = {}
    for position, hypothesis_index in enumerate(hypothesis_indices):
        if not 0 <= hypothesis_index < len(hypotheses):
            raise ValueError("hypothesis_indices contains an out-of-range value")
        grouped.setdefault(hypothesis_index, []).append(position)
    for hypothesis_index, positions in grouped.items():
        hypothesis = hypotheses[hypothesis_index]
        result[positions] = interaction_distance_matrix(
            coords[positions],
            hypothesis.ligand_atom_indices,
            receptor_coords,
        )
    return result


def _coarse_selection_energies(
    search_energies: torch.Tensor,
    restraint_energies: torch.Tensor,
) -> torch.Tensor:
    """Combine physical coarse energy with the mean multi-restraint violation."""
    if search_energies.ndim != 1 or restraint_energies.ndim != 2:
        raise ValueError("coarse search and restraint energies must have shapes [B] and [B,K]")
    if restraint_energies.shape[0] != search_energies.shape[0] or restraint_energies.shape[1] == 0:
        raise ValueError("coarse search and restraint energies must describe the same poses")
    if not torch.isfinite(search_energies).all() or not torch.isfinite(restraint_energies).all():
        raise ValueError("coarse selection energies must be finite")
    return search_energies + restraint_energies.mean(dim=1)


def _inclusive_distance_window_mask(
    distances: torch.Tensor,
    target_distances: torch.Tensor,
    distance_tolerances: torch.Tensor,
) -> torch.Tensor:
    """Apply inclusive windows without rejecting decimal boundary values."""
    if not torch.is_floating_point(distances) or distances.ndim != 2:
        raise ValueError("distances must be a floating-point matrix")
    count = distances.shape[1]
    if target_distances.shape != (count,) or distance_tolerances.shape != (count,):
        raise ValueError("target distances and tolerances must have shape [K]")
    if not torch.isfinite(distances).all():
        raise ValueError("distances must be finite")
    targets = target_distances.to(device=distances.device, dtype=distances.dtype)
    tolerances = distance_tolerances.to(device=distances.device, dtype=distances.dtype)
    if not torch.isfinite(targets).all() or not torch.isfinite(tolerances).all():
        raise ValueError("target distances and tolerances must be finite")
    lower_bounds = targets - tolerances
    upper_bounds = targets + tolerances
    magnitude = torch.maximum(
        torch.ones_like(distances),
        torch.maximum(
            distances.abs(),
            torch.maximum(lower_bounds.abs(), upper_bounds.abs())[None, :],
        ),
    )
    guard = 16.0 * torch.finfo(distances.dtype).eps * magnitude
    return (distances >= lower_bounds[None, :] - guard) & (distances <= upper_bounds[None, :] + guard)


def dock_interaction(
    protein_pdb: str | os.PathLike[str],
    query_ligand: str | os.PathLike[str] | Chem.Mol,
    output_dir: str | os.PathLike[str] = "anchor_dock_interaction",
    *,
    receptor_residue: str | None = None,
    receptor_atom: str | None = None,
    ligand_smarts: str | None = None,
    target_distance: float | None = None,
    distance_tolerance: float | None = None,
    interactions: Sequence[InteractionInput] | None = None,
    pocket_cutoff: float = 12.0,
    include_heteroatoms: bool = True,
    num_confs: int = 32,
    rmsd_threshold: float = 1.0,
    num_candidates: int = 128,
    preselect_k: int = 16,
    max_matches: int = 16,
    max_joint_matches: int = 64,
    optimize: bool = True,
    optimizer: Literal["adam", "adamw", "lbfgs"] = "adam",
    opt_steps: int = 50,
    release_steps: int = 25,
    opt_lr: float = 0.05,
    opt_batch_size: int = 32,
    scorer: ScorerLike = "softdock",
    torsion_penalty: bool = True,
    restraint_weight: float = 10.0,
    top_k: int | None = 10,
    random_seed: int = 42,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    """Run bounded single- or simultaneous multi-interaction local docking."""
    normalized = normalize_interactions(
        interactions=interactions,
        receptor_residue=receptor_residue,
        receptor_atom=receptor_atom,
        ligand_smarts=ligand_smarts,
        target_distance=target_distance,
        distance_tolerance=distance_tolerance,
        default_restraint_weight=restraint_weight,
    )
    max_joint_matches = _positive_int("max_joint_matches", max_joint_matches)
    if len(normalized) == 1:
        item = normalized[0]
        return _dock_single_interaction(
            protein_pdb,
            query_ligand,
            output_dir,
            receptor_residue=item.receptor_residue,
            receptor_atom=item.receptor_atom,
            ligand_smarts=item.ligand_smarts,
            target_distance=item.target_distance,
            distance_tolerance=item.distance_tolerance,
            pocket_cutoff=pocket_cutoff,
            include_heteroatoms=include_heteroatoms,
            num_confs=num_confs,
            rmsd_threshold=rmsd_threshold,
            num_candidates=num_candidates,
            preselect_k=preselect_k,
            max_matches=max_matches,
            max_joint_matches=max_joint_matches,
            optimize=optimize,
            optimizer=optimizer,
            opt_steps=opt_steps,
            release_steps=release_steps,
            opt_lr=opt_lr,
            opt_batch_size=opt_batch_size,
            scorer=scorer,
            torsion_penalty=torsion_penalty,
            restraint_weight=item.restraint_weight,
            top_k=top_k,
            random_seed=random_seed,
            device=device,
            verbose=verbose,
        )
    return _dock_multi_interaction(
        protein_pdb,
        query_ligand,
        output_dir,
        interactions=normalized,
        pocket_cutoff=pocket_cutoff,
        include_heteroatoms=include_heteroatoms,
        num_confs=num_confs,
        rmsd_threshold=rmsd_threshold,
        num_candidates=num_candidates,
        preselect_k=preselect_k,
        max_matches=max_matches,
        max_joint_matches=max_joint_matches,
        optimize=optimize,
        optimizer=optimizer,
        opt_steps=opt_steps,
        release_steps=release_steps,
        opt_lr=opt_lr,
        opt_batch_size=opt_batch_size,
        scorer=scorer,
        torsion_penalty=torsion_penalty,
        top_k=top_k,
        random_seed=random_seed,
        device=device,
        verbose=verbose,
    )


def _dock_multi_interaction(
    protein_pdb: str | os.PathLike[str],
    query_ligand: str | os.PathLike[str] | Chem.Mol,
    output_dir: str | os.PathLike[str],
    *,
    interactions: tuple[InteractionConstraint, ...],
    pocket_cutoff: float,
    include_heteroatoms: bool,
    num_confs: int,
    rmsd_threshold: float,
    num_candidates: int,
    preselect_k: int,
    max_matches: int,
    max_joint_matches: int,
    optimize: bool,
    optimizer: Literal["adam", "adamw", "lbfgs"],
    opt_steps: int,
    release_steps: int,
    opt_lr: float,
    opt_batch_size: int,
    scorer: ScorerLike,
    torsion_penalty: bool,
    top_k: int | None,
    random_seed: int,
    device: str | torch.device | None,
    verbose: bool,
) -> dict[str, object]:
    """Dock against two or more simultaneous atom-pair distance hypotheses."""
    started = time.perf_counter()
    if len(interactions) < 2:
        raise ValueError("multi-interaction execution requires at least two interactions")

    pocket_cutoff = _positive_finite("pocket_cutoff", pocket_cutoff)
    rmsd_threshold = _positive_finite("rmsd_threshold", rmsd_threshold)
    opt_lr = _positive_finite("opt_lr", opt_lr)
    num_confs = _positive_int("num_confs", num_confs)
    num_candidates = _positive_int("num_candidates", num_candidates)
    preselect_k = _positive_int("preselect_k", preselect_k)
    max_matches = _positive_int("max_matches", max_matches)
    max_joint_matches = _positive_int("max_joint_matches", max_joint_matches)
    opt_steps = _positive_int("opt_steps", opt_steps, allow_zero=True)
    release_steps = _positive_int("release_steps", release_steps, allow_zero=True)
    opt_batch_size = _positive_int("opt_batch_size", opt_batch_size)
    random_seed = _positive_int("random_seed", random_seed, allow_zero=True)
    if random_seed > 2**31 - 1:
        raise ValueError("random_seed must not exceed 2147483647")
    if not isinstance(include_heteroatoms, bool):
        raise TypeError("include_heteroatoms must be a bool")
    if not isinstance(optimize, bool):
        raise TypeError("optimize must be a bool")
    if not isinstance(torsion_penalty, bool):
        raise TypeError("torsion_penalty must be a bool")
    if optimizer not in {"adam", "adamw", "lbfgs"}:
        raise ValueError("optimizer must be adam, adamw, or lbfgs")
    if optimize and opt_steps > 0 and release_steps == 0:
        raise ValueError("release_steps must be positive when guided optimization steps are requested")
    if preselect_k > num_candidates:
        raise ValueError("preselect_k must be a positive integer less than or equal to num_candidates")
    if top_k is not None:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer or None")

    protein_path = os.path.abspath(os.fspath(protein_pdb))
    source_fingerprint = file_content_fingerprint(protein_path)
    raw_for_selection = Chem.MolFromPDBFile(protein_path, sanitize=False, removeHs=False)
    if raw_for_selection is None:
        raise ValueError(f"failed to load receptor PDB from {protein_pdb}")
    selections = tuple(
        select_receptor_atom(
            protein_path,
            raw_for_selection,
            item.receptor_residue,
            item.receptor_atom,
        )
        for item in interactions
    )
    target_device = choose_device(device)
    selector_key = tuple((selection.residue_id, selection.atom_name) for selection in selections)
    cache_key = (
        source_fingerprint,
        selector_key,
        float(pocket_cutoff),
        bool(include_heteroatoms),
        str(target_device),
        ATOM_TYPING_VERSION,
    )
    pocket_context = _INTERACTION_CONTEXT_CACHE.get(cache_key)
    if pocket_context is not None:
        _INTERACTION_CONTEXT_CACHE.move_to_end(cache_key)
    else:
        receptor_no_hs = Chem.MolFromPDBFile(protein_path, sanitize=False, removeHs=True)
        if receptor_no_hs is None:
            raise ValueError(f"failed to load receptor PDB from {protein_pdb}")
        pocket_mol = extract_pocket_around_residues(
            receptor_no_hs,
            [selection.residue_id for selection in selections],
            cutoff=pocket_cutoff,
            include_heteroatoms=include_heteroatoms,
        )
        _reject_pocket_altlocs(pocket_mol)
        pocket_context = receptor_context_from_mol(
            pocket_mol,
            target_device,
            source_path=protein_path,
            source_fingerprint=source_fingerprint,
        )
        _INTERACTION_CONTEXT_CACHE[cache_key] = pocket_context
        _INTERACTION_CONTEXT_CACHE.move_to_end(cache_key)
        while len(_INTERACTION_CONTEXT_CACHE) > INTERACTION_CONTEXT_CACHE_MAXSIZE:
            _INTERACTION_CONTEXT_CACHE.popitem(last=False)

    ligand, canonical_smiles = load_ligand(query_ligand, add_hydrogens=False)
    ligand, representative_ids = generate_conformers_and_cluster(
        ligand,
        target_device,
        num_confs=num_confs,
        rmsd_threshold=rmsd_threshold,
        add_hydrogens=True,
        random_seed=random_seed,
    )
    if not representative_ids:
        raise RuntimeError("interaction docking conformer generation produced no representatives")
    anchor_groups = tuple(
        tuple(select_ligand_anchors(ligand, item.ligand_smarts, max_matches=max_matches)) for item in interactions
    )
    hypotheses = enumerate_joint_hypotheses(
        anchor_groups,
        max_joint_matches=max_joint_matches,
    )
    primary_interaction_index = min(
        range(len(interactions)),
        key=lambda index: (len(anchor_groups[index]), index),
    )

    # Keep an exact float64 copy for conservative geometry preflight and the
    # inclusive final hard gate. Converting float32 values back to float64 after
    # rounding can otherwise reject poses exactly on a requested boundary.
    representative_coords_exact_cpu = torch.stack(
        [
            torch.tensor(ligand.GetConformer(conf_id).GetPositions(), dtype=torch.float64)
            for conf_id in representative_ids
        ]
    )
    representative_coords = representative_coords_exact_cpu.to(
        device=target_device,
        dtype=torch.float32,
    )
    conformer_count = len(representative_ids)
    receptor_coords_exact_cpu = torch.tensor(
        [selection.coordinate for selection in selections],
        dtype=torch.float64,
    )
    target_distances_exact_cpu = torch.tensor(
        [item.target_distance for item in interactions],
        dtype=torch.float64,
    )
    distance_tolerances_exact_cpu = torch.tensor(
        [item.distance_tolerance for item in interactions],
        dtype=torch.float64,
    )
    restraint_weights_exact_cpu = torch.tensor(
        [item.restraint_weight for item in interactions],
        dtype=torch.float64,
    )
    receptor_coords_exact = receptor_coords_exact_cpu.to(target_device)
    target_distances_exact = target_distances_exact_cpu.to(target_device)
    distance_tolerances_exact = distance_tolerances_exact_cpu.to(target_device)
    restraint_weights_exact = restraint_weights_exact_cpu.to(target_device)
    receptor_coords = receptor_coords_exact.to(dtype=torch.float32)
    target_distances = target_distances_exact.to(dtype=torch.float32)
    distance_tolerances = distance_tolerances_exact.to(dtype=torch.float32)
    restraint_weights = restraint_weights_exact.to(dtype=torch.float32)

    feasible_conformers: list[list[int]] = []
    for hypothesis in hypotheses:
        feasible_conformers.append(
            [
                conformer_ordinal
                for conformer_ordinal in range(conformer_count)
                if pairwise_shell_feasible(
                    representative_coords_exact_cpu[conformer_ordinal],
                    hypothesis.ligand_atom_indices,
                    receptor_coords_exact_cpu,
                    target_distances_exact_cpu,
                    distance_tolerances_exact_cpu,
                )
            ]
        )
    active_hypotheses, state_cycle = _candidate_state_cycle(feasible_conformers)
    active_hypothesis_count = len(active_hypotheses)
    if num_candidates < active_hypothesis_count:
        raise ValueError(
            f"num_candidates ({num_candidates}) is less than the number of viable joint "
            f"hypotheses ({active_hypothesis_count})"
        )
    if preselect_k < active_hypothesis_count:
        raise ValueError(
            f"preselect_k ({preselect_k}) is less than the number of viable joint "
            f"hypotheses ({active_hypothesis_count})"
        )

    _, candidate_states = _allocate_candidate_states(feasible_conformers, num_candidates)
    candidate_group_indices = torch.tensor(
        [state[0] for state in candidate_states], dtype=torch.long, device=target_device
    )
    candidate_hypothesis_indices = [state[1] for state in candidate_states]
    conformer_ordinals = torch.tensor([state[2] for state in candidate_states], dtype=torch.long, device=target_device)
    base_coords = representative_coords[conformer_ordinals]
    pivot_atom_indices = torch.tensor(
        [
            hypotheses[hypothesis_index].anchors[primary_interaction_index].ligand_atom_index
            for hypothesis_index in candidate_hypothesis_indices
        ],
        dtype=torch.long,
        device=target_device,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(random_seed))
    normal = torch.randn((num_candidates, 3), generator=generator)
    directions = normal / torch.linalg.vector_norm(normal, dim=1, keepdim=True).clamp_min(1e-12)
    directions = directions.to(target_device)
    rotation_vectors = sample_uniform_rotation_vectors(num_candidates, generator).to(target_device)
    primary_receptor_coord = receptor_coords[primary_interaction_index]
    primary_target_distance = target_distances[primary_interaction_index]
    centers = primary_receptor_coord[None, :] + primary_target_distance * directions

    input_rotatable_bonds = int(rdMolDescriptors.CalcNumRotatableBonds(ligand))
    num_rotatable_bonds = input_rotatable_bonds if torsion_penalty else 0
    engine = DockingEngine(
        scorer=scorer,
        device=target_device,
        optimizer=optimizer,
        num_steps=opt_steps,
        learning_rate=opt_lr,
        batch_size=opt_batch_size,
    )
    problem = engine.prepare(
        ligand,
        pocket_context,
        num_rotatable_bonds=num_rotatable_bonds,
    )

    initial_search_energy_chunks: list[torch.Tensor] = []
    initial_coord_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, num_candidates, opt_batch_size):
            stop = min(start + opt_batch_size, num_candidates)
            chunk_base = base_coords[start:stop]
            chunk_pivots = pivot_atom_indices[start:stop]
            chunk_rotations = rotation_vectors[start:stop]
            chunk_angles = torch.linalg.vector_norm(chunk_rotations, dim=1)
            chunk_matrices = get_batched_rotation_matrix(chunk_rotations, chunk_angles)
            chunk_rows = torch.arange(stop - start, device=target_device)
            chunk_pivot_coords = chunk_base[chunk_rows, chunk_pivots, :]
            chunk_coords = (
                torch.matmul(
                    chunk_base - chunk_pivot_coords[:, None, :],
                    chunk_matrices.transpose(1, 2),
                )
                + centers[start:stop, None, :]
            )
            initial_search_energy_chunks.append(problem.scorer.search_energy(chunk_coords))
            initial_coord_chunks.append(chunk_coords)
    initial_search_energies = torch.cat(initial_search_energy_chunks)
    initial_coords = torch.cat(initial_coord_chunks)
    initial_distance_matrix = _distance_matrix_by_hypothesis(
        initial_coords,
        candidate_hypothesis_indices,
        hypotheses,
        receptor_coords,
    )
    initial_restraint_matrix = flat_bottom_distance_restraint_matrix(
        initial_distance_matrix,
        target_distances,
        distance_tolerances,
        restraint_weights,
    )
    initial_selection_energies = _coarse_selection_energies(
        initial_search_energies,
        initial_restraint_matrix,
    )

    preselected_indices = preselect_candidates(
        energies=initial_selection_energies.cpu(),
        match_indices=candidate_group_indices.cpu(),
        conformer_ordinals=conformer_ordinals.cpu(),
        preselect_k=preselect_k,
        match_count=active_hypothesis_count,
        conformer_count=conformer_count,
    )
    preselected_initial_coords = initial_coords[preselected_indices]
    preselected_base_coords = base_coords[preselected_indices]
    preselected_centers = centers[preselected_indices]
    preselected_rotation_vectors = rotation_vectors[preselected_indices]
    preselected_hypothesis_indices = [candidate_hypothesis_indices[index] for index in preselected_indices]

    num_atoms = len(ligand.GetAtoms())
    guided_coords = torch.empty((preselect_k, num_atoms, 3), dtype=torch.float32, device=target_device)
    final_coords = torch.empty_like(guided_coords)
    optimization_improved = False
    if optimize:
        positions_by_hypothesis: dict[int, list[int]] = {}
        for position, hypothesis_index in enumerate(preselected_hypothesis_indices):
            positions_by_hypothesis.setdefault(hypothesis_index, []).append(position)
        guide_stats_parts: list[OptimizationStats] = []
        release_stats_parts: list[OptimizationStats] = []
        for hypothesis_index, positions in positions_by_hypothesis.items():
            hypothesis = hypotheses[hypothesis_index]
            pivot = hypothesis.anchors[primary_interaction_index].ligand_atom_index

            def additional_energy_fn(
                values: torch.Tensor,
                ligand_atom_indices: tuple[int, ...] = hypothesis.ligand_atom_indices,
            ) -> torch.Tensor:
                distances = interaction_distance_matrix(
                    values,
                    ligand_atom_indices,
                    receptor_coords,
                )
                return mean_flat_bottom_distance_restraint(
                    distances,
                    target_distances,
                    distance_tolerances,
                    restraint_weights,
                )

            guided_chunk, final_chunk, guide_part, release_part = engine.optimize_se3(
                problem,
                preselected_base_coords[positions],
                pivot,
                centers=preselected_centers[positions],
                rotation_vectors=preselected_rotation_vectors[positions],
                additional_energy_fn=additional_energy_fn,
                release_steps=release_steps,
            )
            guided_coords[positions] = guided_chunk
            final_coords[positions] = final_chunk
            guide_stats_parts.append(guide_part)
            release_stats_parts.append(release_part)
        guide_stats = _aggregate_stats(guide_stats_parts, preselect_k)
        release_stats = _aggregate_stats(release_stats_parts, preselect_k)
        optimization_applied = opt_steps > 0 or release_steps > 0
    else:
        guided_coords = preselected_initial_coords.clone()
        final_coords = preselected_initial_coords.clone()
        initial_best = float(initial_search_energies[preselected_indices].min().item())
        zero_stats = OptimizationStats(
            average_steps=0.0,
            minimum_steps=0,
            maximum_steps=0,
            num_poses=preselect_k,
            initial_best_energy=initial_best,
            final_best_energy=initial_best,
        )
        guide_stats = zero_stats
        release_stats = zero_stats
        optimization_applied = False

    coordinate_scale = float(10**OUTPUT_COORDINATE_DECIMALS)
    export_final_coords = torch.round(final_coords.to(torch.float64) * coordinate_scale) / coordinate_scale
    initial_distances = _distance_matrix_by_hypothesis(
        preselected_initial_coords,
        preselected_hypothesis_indices,
        hypotheses,
        receptor_coords,
    )
    guided_distances = _distance_matrix_by_hypothesis(
        guided_coords,
        preselected_hypothesis_indices,
        hypotheses,
        receptor_coords,
    )
    final_distances = _distance_matrix_by_hypothesis(
        export_final_coords,
        preselected_hypothesis_indices,
        hypotheses,
        receptor_coords_exact,
    )
    initial_restraints = flat_bottom_distance_restraint_matrix(
        initial_distances,
        target_distances,
        distance_tolerances,
        restraint_weights,
    )
    guided_restraints = flat_bottom_distance_restraint_matrix(
        guided_distances,
        target_distances,
        distance_tolerances,
        restraint_weights,
    )
    final_restraints = flat_bottom_distance_restraint_matrix(
        final_distances,
        target_distances_exact,
        distance_tolerances_exact,
        restraint_weights_exact,
    )
    satisfaction = _inclusive_distance_window_mask(
        final_distances,
        target_distances_exact,
        distance_tolerances_exact,
    )
    valid_mask = satisfaction.all(dim=1)
    if not valid_mask.any():
        raise RuntimeError("no poses satisfied all interaction distance restraints after release")
    surviving_indices = torch.nonzero(valid_mask).flatten()
    final_surviving_coords = export_final_coords[surviving_indices]
    initial_surviving_coords = preselected_initial_coords[surviving_indices]
    initial_components, final_components = engine.report_scores(
        problem,
        initial_surviving_coords,
        final_surviving_coords.to(dtype=initial_surviving_coords.dtype),
    )
    if optimization_applied:
        optimization_improved = bool(
            final_components.score.min().item() < initial_components.score.min().item() - 1e-12
        )

    interaction_records: list[dict[str, object]] = []
    for index, (item, selection, anchors) in enumerate(zip(interactions, selections, anchor_groups, strict=True)):
        interaction_records.append(
            {
                "interaction_index": index,
                "requested_receptor_residue": item.receptor_residue,
                "receptor_residue": selection.residue_id,
                "receptor_atom": selection.atom_name,
                "receptor_atom_element": selection.element,
                "receptor_atom_coordinate": list(selection.coordinate),
                "receptor_atom_rdkit_index": selection.rdkit_index,
                "receptor_atom_pdb_serial": selection.pdb_serial,
                "receptor_atom_occupancy": selection.occupancy,
                "ligand_smarts": item.ligand_smarts,
                "ligand_matches": [asdict(anchor) for anchor in anchors],
                "target_distance": item.target_distance,
                "distance_tolerance": item.distance_tolerance,
                "distance_lower": item.target_distance - item.distance_tolerance,
                "distance_upper": item.target_distance + item.distance_tolerance,
                "restraint_weight": item.restraint_weight,
            }
        )
    joint_hypothesis_records = [
        {
            "hypothesis_index": hypothesis.hypothesis_index,
            "ligand_atom_indices": list(hypothesis.ligand_atom_indices),
            "ligand_match_indices": list(hypothesis.ligand_match_indices),
            "representative_matches": [list(anchor.representative_match) for anchor in hypothesis.anchors],
        }
        for hypothesis in hypotheses
    ]
    search_parameters = {
        "interaction_logic": "all",
        "interactions": interaction_dicts(interactions),
        "primary_interaction_index": primary_interaction_index,
        "pocket_cutoff": float(pocket_cutoff),
        "include_heteroatoms": bool(include_heteroatoms),
        "num_confs": int(num_confs),
        "rmsd_threshold": float(rmsd_threshold),
        "num_candidates": int(num_candidates),
        "preselect_k": int(preselect_k),
        "max_matches": int(max_matches),
        "max_joint_matches": int(max_joint_matches),
        "optimize": bool(optimize),
        "optimizer": str(optimizer),
        "opt_steps": int(opt_steps),
        "release_steps": int(release_steps),
        "opt_lr": float(opt_lr),
        "opt_batch_size": int(opt_batch_size),
        "scorer": str(problem.scorer.name),
        "torsion_penalty": bool(torsion_penalty),
        "restraint_aggregation": "mean",
        "top_k": int(top_k) if top_k is not None else None,
        "random_seed": int(random_seed),
    }
    compact_json = {"sort_keys": True, "separators": (",", ":")}
    molecule_metadata = {
        "Mode": "interaction",
        "Anchor_Strategy": "explicit_multi_atom_pair_distances",
        "Search_Method": "guided_se3_release" if optimization_applied else "guided_random_placement",
        "Interaction_Logic": "all",
        "Num_Interactions": len(interactions),
        "Interactions": json.dumps(interaction_records, **compact_json),
        "Primary_Interaction_Index": primary_interaction_index,
        "Joint_Hypotheses": json.dumps(joint_hypothesis_records, **compact_json),
        "Num_Joint_Hypotheses": len(hypotheses),
        "Num_Viable_Joint_Hypotheses": active_hypothesis_count,
        "Num_Viable_Conformer_Hypothesis_Pairs": len(state_cycle),
        "Canonical_SMILES": canonical_smiles,
        "Ligand_Index_Space": "canonical_heavy_atom_topology",
        "Output_Coordinate_Decimals": OUTPUT_COORDINATE_DECIMALS,
        "Restraint_Formula": RESTRAINT_FORMULA,
        "Restraint_Aggregation": "mean",
        "Conformer_IDs": ",".join(str(conf_id) for conf_id in representative_ids),
        "Num_Conformer_Attempts": num_confs,
        "Num_Candidates": num_candidates,
        "Num_Preselected": preselect_k,
        "Num_Valid": int(valid_mask.sum().item()),
        "Max_Matches_Per_Selector": max_matches,
        "Max_Joint_Matches_Requested": max_joint_matches,
        "Random_Seed": random_seed,
        "Torsion_Penalty_Requested": torsion_penalty,
        "Torsion_Penalty_Applied": problem.scorer.torsion_penalty_applied,
        "Input_Ligand_Rotatable_Bonds": input_rotatable_bonds,
        "Score_Rotatable_Bonds": problem.scorer.effective_rotatable_bonds,
        "Intramolecular_Reference": float(final_components.intramolecular_reference.detach().cpu().item()),
        "Receptor_Structure_Fingerprint": pocket_context.structure_fingerprint,
        "Receptor_Source_Fingerprint": pocket_context.source_fingerprint,
        "Receptor_Structure_Scope": "residue_union_pocket",
        "Atom_Typing": pocket_context.atom_typing_version,
        "Optimization_Requested": optimize,
        "Optimization_Applied": optimization_applied,
        "Optimization_Improved": optimization_improved,
        "Optimizer": optimizer,
        "Optimization_Steps_Requested": opt_steps,
        "Optimization_Release_Steps": release_steps,
        "Optimization_Learning_Rate": opt_lr,
        "Guide_Optimization_Stats": json.dumps(guide_stats.as_dict(), **compact_json),
        "Release_Optimization_Stats": json.dumps(release_stats.as_dict(), **compact_json),
        "Guide_Objective_Semantics": ("scorer_search_energy_plus_mean_flat_bottom_restraints"),
        "Release_Objective_Semantics": "scorer_search_energy_only",
        "Search_Parameters": json.dumps(search_parameters, **compact_json),
        "Protonation_Limitation": PROTONATION_LIMITATION,
    }

    pose_ids: list[str] = []
    surviving_per_pose_metadata: list[dict[str, object]] = []
    surviving_pose_interactions: list[dict[str, object]] = []
    for row_index in surviving_indices.tolist():
        candidate_id = preselected_indices[row_index]
        hypothesis_index = preselected_hypothesis_indices[row_index]
        hypothesis = hypotheses[hypothesis_index]
        conformer_ordinal = int(conformer_ordinals[candidate_id].item())
        interaction_values = []
        for interaction_index, anchor in enumerate(hypothesis.anchors):
            interaction_values.append(
                {
                    "interaction_index": interaction_index,
                    "ligand_atom_index": anchor.ligand_atom_index,
                    "ligand_anchor_element": anchor.element,
                    "ligand_anchor_formal_charge": anchor.formal_charge,
                    "ligand_match_index": anchor.match_index,
                    "ligand_match": list(anchor.representative_match),
                    "initial_distance": float(initial_distances[row_index, interaction_index].item()),
                    "guided_distance": float(guided_distances[row_index, interaction_index].item()),
                    "final_distance": float(final_distances[row_index, interaction_index].item()),
                    "initial_restraint_energy": float(initial_restraints[row_index, interaction_index].item()),
                    "guided_restraint_energy": float(guided_restraints[row_index, interaction_index].item()),
                    "final_restraint_energy": float(final_restraints[row_index, interaction_index].item()),
                    "satisfied": bool(satisfaction[row_index, interaction_index].item()),
                }
            )
        pose_ids.append(f"candidate_{candidate_id:05d}")
        surviving_pose_interactions.append(
            {
                "pose_id": pose_ids[-1],
                "joint_hypothesis_index": hypothesis_index,
                "source_conformer": int(representative_ids[conformer_ordinal]),
                "source_representative_index": conformer_ordinal,
                "interactions": interaction_values,
            }
        )
        surviving_per_pose_metadata.append(
            {
                "Joint_Hypothesis_Index": hypothesis_index,
                "Joint_Ligand_Atom_Indices": json.dumps(list(hypothesis.ligand_atom_indices), separators=(",", ":")),
                "Joint_Ligand_Match_Indices": json.dumps(list(hypothesis.ligand_match_indices), separators=(",", ":")),
                "Source_Conformer": int(representative_ids[conformer_ordinal]),
                "Source_Representative_Index": conformer_ordinal,
                "Candidate_Coarse_Search_Energy": (f"{float(initial_search_energies[candidate_id].item()):.8f}"),
                "Candidate_Coarse_Selection_Energy": (f"{float(initial_selection_energies[candidate_id].item()):.8f}"),
                "Interaction_Distances": json.dumps(interaction_values, **compact_json),
                "Initial_Restraint_Energy": (f"{float(initial_restraints[row_index].mean().item()):.8f}"),
                "Guided_Restraint_Energy": (f"{float(guided_restraints[row_index].mean().item()):.8f}"),
                "Final_Restraint_Energy": (f"{float(final_restraints[row_index].mean().item()):.8f}"),
                "Restraint_Satisfaction": "True",
            }
        )

    output_path = Path(output_dir) / "interaction_poses.sdf"
    selected = write_ranked_poses(
        ligand,
        final_surviving_coords,
        final_components.score,
        str(output_path),
        scorer_name=problem.scorer.name,
        score_units=problem.scorer.units,
        score_semantics=SCORE_SEMANTICS,
        scorer_fingerprint=problem.scorer.fingerprint,
        search_energies=final_components.search_energy,
        initial_scores=initial_components.score,
        pose_ids=pose_ids,
        top_k=top_k,
        molecule_metadata=molecule_metadata,
        per_pose_metadata=surviving_per_pose_metadata,
    )
    runtime = time.perf_counter() - started
    best_index = int(torch.argmin(final_components.score).item())
    result: dict[str, object] = {
        "mode": "interaction",
        "anchor_dock_version": __version__,
        "output_file": str(output_path),
        "num_poses": int(selected.numel()),
        "num_candidates": num_candidates,
        "num_preselected": preselect_k,
        "valid_poses": int(valid_mask.sum().item()),
        "num_representatives": len(representative_ids),
        "best_score": float(final_components.score[best_index].detach().cpu()),
        "best_search_energy": float(final_components.search_energy[best_index].detach().cpu()),
        "score_units": problem.scorer.units,
        "scorer": problem.scorer.name,
        "scorer_fingerprint": problem.scorer.fingerprint,
        "score_semantics": SCORE_SEMANTICS,
        "interaction_logic": "all",
        "interactions": interaction_records,
        "num_interactions": len(interactions),
        "primary_interaction_index": primary_interaction_index,
        "joint_hypotheses": joint_hypothesis_records,
        "pose_interactions": [surviving_pose_interactions[index] for index in selected.tolist()],
        "num_joint_hypotheses": len(hypotheses),
        "num_viable_joint_hypotheses": active_hypothesis_count,
        "num_viable_conformer_hypothesis_pairs": len(state_cycle),
        "max_joint_matches": max_joint_matches,
        "restraint_formula": RESTRAINT_FORMULA,
        "restraint_aggregation": "mean",
        "output_coordinate_decimals": OUTPUT_COORDINATE_DECIMALS,
        "canonical_smiles": canonical_smiles,
        "receptor_structure_fingerprint": pocket_context.structure_fingerprint,
        "receptor_structure_scope": "residue_union_pocket",
        "receptor_source_fingerprint": pocket_context.source_fingerprint,
        "atom_typing": pocket_context.atom_typing_version,
        "torsion_penalty_requested": torsion_penalty,
        "torsion_penalty_applied": problem.scorer.torsion_penalty_applied,
        "input_ligand_rotatable_bonds": input_rotatable_bonds,
        "score_rotatable_bonds": problem.scorer.effective_rotatable_bonds,
        "intramolecular_reference": float(final_components.intramolecular_reference.detach().cpu().item()),
        "optimization_requested": optimize,
        "optimization_applied": optimization_applied,
        "optimization_improved": optimization_improved,
        "optimization_config": {
            "optimizer": optimizer,
            "steps": opt_steps,
            "release_steps": release_steps,
            "learning_rate": opt_lr,
            "batch_size": opt_batch_size,
        },
        "guide_optimization": guide_stats.as_dict(),
        "release_optimization": release_stats.as_dict(),
        "guide_objective_semantics": ("scorer_search_energy_plus_mean_flat_bottom_restraints"),
        "release_objective_semantics": "scorer_search_energy_only",
        "search_parameters": search_parameters,
        "representative_conformer_ids": [int(value) for value in representative_ids],
        "protonation_limitation": PROTONATION_LIMITATION,
        "optimized": optimization_applied,
        "runtime": runtime,
        "device": str(pocket_context.device),
    }
    if verbose:
        print(
            f"multi-interaction docking complete: {result['num_poses']} poses, "
            f"best={result['best_score']:.4f}, {runtime:.2f}s"
        )
    return result


def _dock_single_interaction(
    protein_pdb: str | os.PathLike[str],
    query_ligand: str | os.PathLike[str] | Chem.Mol,
    output_dir: str | os.PathLike[str] = "anchor_dock_interaction",
    *,
    receptor_residue: str,
    receptor_atom: str,
    ligand_smarts: str,
    target_distance: float,
    distance_tolerance: float,
    pocket_cutoff: float = 12.0,
    include_heteroatoms: bool = True,
    num_confs: int = 32,
    rmsd_threshold: float = 1.0,
    num_candidates: int = 128,
    preselect_k: int = 16,
    max_matches: int = 16,
    max_joint_matches: int = 64,
    optimize: bool = True,
    optimizer: Literal["adam", "adamw", "lbfgs"] = "adam",
    opt_steps: int = 50,
    release_steps: int = 25,
    opt_lr: float = 0.05,
    opt_batch_size: int = 32,
    scorer: ScorerLike = "softdock",
    torsion_penalty: bool = True,
    restraint_weight: float = 10.0,
    top_k: int | None = 10,
    random_seed: int = 42,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    """Interaction-guided docking pipeline using flat-bottom distance restraints."""
    started = time.perf_counter()

    # Validate all cheap controls before parsing or preparing either molecule.
    target_distance = _positive_finite("target_distance", target_distance)
    distance_tolerance = _positive_finite("distance_tolerance", distance_tolerance)
    if distance_tolerance >= target_distance:
        raise ValueError("distance_tolerance must satisfy 0 < tolerance < target_distance")
    pocket_cutoff = _positive_finite("pocket_cutoff", pocket_cutoff)
    rmsd_threshold = _positive_finite("rmsd_threshold", rmsd_threshold)
    opt_lr = _positive_finite("opt_lr", opt_lr)
    restraint_weight = _positive_finite("restraint_weight", restraint_weight, allow_zero=True)
    num_confs = _positive_int("num_confs", num_confs)
    num_candidates = _positive_int("num_candidates", num_candidates)
    preselect_k = _positive_int("preselect_k", preselect_k)
    max_matches = _positive_int("max_matches", max_matches)
    max_joint_matches = _positive_int("max_joint_matches", max_joint_matches)
    opt_steps = _positive_int("opt_steps", opt_steps, allow_zero=True)
    release_steps = _positive_int("release_steps", release_steps, allow_zero=True)
    opt_batch_size = _positive_int("opt_batch_size", opt_batch_size)
    random_seed = _positive_int("random_seed", random_seed, allow_zero=True)
    if random_seed > 2**31 - 1:
        raise ValueError("random_seed must not exceed 2147483647")
    if not isinstance(include_heteroatoms, bool):
        raise TypeError("include_heteroatoms must be a bool")
    if not isinstance(optimize, bool):
        raise TypeError("optimize must be a bool")
    if not isinstance(torsion_penalty, bool):
        raise TypeError("torsion_penalty must be a bool")
    if optimizer not in {"adam", "adamw", "lbfgs"}:
        raise ValueError("optimizer must be adam, adamw, or lbfgs")
    if optimize and opt_steps > 0 and release_steps == 0:
        raise ValueError("release_steps must be positive when guided optimization steps are requested")
    if preselect_k > num_candidates:
        raise ValueError("preselect_k must be a positive integer less than or equal to num_candidates")
    if top_k is not None:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer or None")

    # 1) Prepare residue-centered receptor pocket
    protein_path = os.path.abspath(os.fspath(protein_pdb))
    source_fingerprint = file_content_fingerprint(protein_path)

    # Resolve receptor atom fail-closed
    raw_for_sel = Chem.MolFromPDBFile(protein_path, sanitize=False, removeHs=False)
    if raw_for_sel is None:
        raise ValueError(f"failed to load receptor PDB from {protein_pdb}")
    sel = select_receptor_atom(protein_path, raw_for_sel, receptor_residue, receptor_atom)

    target_device = choose_device(device)

    cache_key = (
        source_fingerprint,
        sel.residue_id,
        sel.atom_name,
        float(pocket_cutoff),
        bool(include_heteroatoms),
        str(target_device),
        ATOM_TYPING_VERSION,
    )

    pocket_context = _INTERACTION_CONTEXT_CACHE.get(cache_key)
    if pocket_context is not None:
        _INTERACTION_CONTEXT_CACHE.move_to_end(cache_key)
    else:
        # Load receptor with hydrogens removed
        receptor_no_hs = Chem.MolFromPDBFile(protein_path, sanitize=False, removeHs=True)
        if receptor_no_hs is None:
            raise ValueError(f"failed to load receptor PDB from {protein_pdb}")

        pocket_mol = extract_pocket_around_residue(
            receptor_no_hs,
            residue_spec=sel.residue_id,
            cutoff=pocket_cutoff,
            include_heteroatoms=include_heteroatoms,
        )
        _reject_pocket_altlocs(pocket_mol)

        pocket_context = receptor_context_from_mol(
            pocket_mol,
            target_device,
            source_path=protein_path,
            source_fingerprint=source_fingerprint,
        )
        _INTERACTION_CONTEXT_CACHE[cache_key] = pocket_context
        _INTERACTION_CONTEXT_CACHE.move_to_end(cache_key)
        while len(_INTERACTION_CONTEXT_CACHE) > INTERACTION_CONTEXT_CACHE_MAXSIZE:
            _INTERACTION_CONTEXT_CACHE.popitem(last=False)

    # 2) Load/canonicalize connected heavy ligand, generate/cluster conformers, resolve matches
    ligand, canonical_smiles = load_ligand(query_ligand, add_hydrogens=False)
    ligand, representative_ids = generate_conformers_and_cluster(
        ligand,
        target_device,
        num_confs=num_confs,
        rmsd_threshold=rmsd_threshold,
        add_hydrogens=True,
        random_seed=random_seed,
    )
    if not representative_ids:
        raise RuntimeError("interaction docking conformer generation produced no representatives")
    # Select against the exact canonical heavy-atom topology used for scoring/export.
    anchors = select_ligand_anchors(ligand, ligand_smarts, max_matches=max_matches)

    match_count = len(anchors)
    conformer_count = len(representative_ids)
    if match_count > max_joint_matches:
        raise MatchLimitExceededError(
            f"interaction selector produced {match_count} joint hypotheses, exceeding "
            f"max_joint_matches={max_joint_matches}; refine the SMARTS pattern or raise the explicit bound"
        )

    # Validate candidate and preselect count against match count
    if num_candidates < match_count:
        raise ValueError(
            f"num_candidates ({num_candidates}) is less than the number of ligand anchor matches ({match_count}). "
            "Please refine/increase num_candidates to be at least match_count."
        )
    if preselect_k < match_count:
        raise ValueError(
            f"preselect_k ({preselect_k}) is less than the number of ligand anchor matches ({match_count}). "
            "Please refine/increase preselect_k to be at least match_count."
        )

    # 4) Deterministic candidate allocation
    representative_coords = torch.stack(
        [
            torch.tensor(ligand.GetConformer(conf_id).GetPositions(), dtype=torch.float32)
            for conf_id in representative_ids
        ]
    ).to(target_device)

    candidate_indices = torch.arange(num_candidates, device=target_device)
    match_indices = candidate_indices % match_count
    conformer_ordinals = (candidate_indices // match_count) % conformer_count

    base_coords = representative_coords[conformer_ordinals]
    pivot_atom_indices = torch.tensor(
        [anchors[match_index].ligand_atom_index for match_index in match_indices.cpu().tolist()],
        dtype=torch.long,
        device=target_device,
    )

    # Sample seeded uniform sphere directions
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(random_seed))
    normal = torch.randn((num_candidates, 3), generator=generator)
    directions = normal / torch.linalg.vector_norm(normal, dim=1, keepdim=True).clamp_min(1e-12)
    directions = directions.to(target_device)

    # Sample seeded uniform rotation vectors
    rotation_vectors = sample_uniform_rotation_vectors(num_candidates, generator).to(target_device)

    receptor_coord_tensor = torch.tensor(sel.coordinate, dtype=torch.float32, device=target_device)
    centers = receptor_coord_tensor[None, :] + target_distance * directions

    # Initialize Engine and Prepare Scorer
    input_rotatable_bonds = int(rdMolDescriptors.CalcNumRotatableBonds(ligand))
    num_rotatable_bonds = input_rotatable_bonds if torsion_penalty else 0

    engine = DockingEngine(
        scorer=scorer,
        device=target_device,
        optimizer=optimizer,
        num_steps=opt_steps,
        learning_rate=opt_lr,
        batch_size=opt_batch_size,
    )

    problem = engine.prepare(
        ligand,
        pocket_context,
        num_rotatable_bonds=num_rotatable_bonds,
    )

    # 5) Build initial coordinates and coarse-score problem.scorer.search_energy in chunks
    initial_search_energies = []
    initial_coords_list = []

    with torch.no_grad():
        for start in range(0, num_candidates, opt_batch_size):
            end = min(start + opt_batch_size, num_candidates)
            chunk_base = base_coords[start:end]
            chunk_pivots = pivot_atom_indices[start:end]
            chunk_centers = centers[start:end]
            chunk_rot_vecs = rotation_vectors[start:end]

            chunk_angles = torch.linalg.vector_norm(chunk_rot_vecs, dim=1)
            chunk_rot = get_batched_rotation_matrix(chunk_rot_vecs, chunk_angles)

            chunk_pivot_coords = chunk_base[torch.arange(end - start, device=target_device), chunk_pivots, :]
            chunk_centered = chunk_base - chunk_pivot_coords[:, None, :]
            chunk_rotated = torch.matmul(chunk_centered, chunk_rot.transpose(1, 2))
            chunk_init_coords = chunk_rotated + chunk_centers[:, None, :]

            chunk_energy = problem.scorer.search_energy(chunk_init_coords)

            initial_search_energies.append(chunk_energy)
            initial_coords_list.append(chunk_init_coords)

        initial_search_energies = torch.cat(initial_search_energies)
        initial_coords = torch.cat(initial_coords_list)

    # 6) Stratified preselector
    preselected_indices = preselect_candidates(
        energies=initial_search_energies.cpu(),
        match_indices=match_indices.cpu(),
        conformer_ordinals=conformer_ordinals.cpu(),
        preselect_k=preselect_k,
        match_count=match_count,
        conformer_count=conformer_count,
    )

    preselected_initial_coords = initial_coords[preselected_indices]
    preselected_base_coords = base_coords[preselected_indices]
    preselected_centers = centers[preselected_indices]
    preselected_rotation_vectors = rotation_vectors[preselected_indices]
    preselected_pivot_atom_indices = pivot_atom_indices[preselected_indices]

    # 7) Optimize only preselected rows, grouped by pivot atom index
    num_atoms = len(ligand.GetAtoms())
    guided_coords = torch.empty((preselect_k, num_atoms, 3), dtype=torch.float32, device=target_device)
    final_coords = torch.empty((preselect_k, num_atoms, 3), dtype=torch.float32, device=target_device)
    optimization_improved = False

    if optimize:
        pivot_to_preselected_positions: dict[int, list[int]] = {}
        for idx, pivot in enumerate(preselected_pivot_atom_indices.tolist()):
            pivot_to_preselected_positions.setdefault(pivot, []).append(idx)

        guide_stats_list = []
        release_stats_list = []

        for pivot, pos_list in pivot_to_preselected_positions.items():
            group_base_coords = preselected_base_coords[pos_list]
            group_centers = preselected_centers[pos_list]
            group_rot_vecs = preselected_rotation_vectors[pos_list]

            def group_additional_energy_fn(values: torch.Tensor) -> torch.Tensor:
                dists = interaction_distances(values, pivot, receptor_coord_tensor)
                return flat_bottom_distance_restraint(
                    dists,
                    target_distance=target_distance,
                    distance_tolerance=distance_tolerance,
                    restraint_weight=restraint_weight,
                )

            guided_chunk, final_chunk, guide_stats_chunk, release_stats_chunk = engine.optimize_se3(
                problem,
                group_base_coords,
                pivot,
                centers=group_centers,
                rotation_vectors=group_rot_vecs,
                additional_energy_fn=group_additional_energy_fn,
                release_steps=release_steps,
            )

            guided_coords[pos_list] = guided_chunk
            final_coords[pos_list] = final_chunk

            guide_stats_list.append(guide_stats_chunk)
            release_stats_list.append(release_stats_chunk)

        guide_stats = _aggregate_stats(guide_stats_list, preselect_k)
        release_stats = _aggregate_stats(release_stats_list, preselect_k)

        optimization_applied = opt_steps > 0 or release_steps > 0
    else:
        guided_coords = preselected_initial_coords.clone()
        final_coords = preselected_initial_coords.clone()

        initial_best = float(initial_search_energies[preselected_indices].min().item())
        zero_stats = OptimizationStats(
            average_steps=0.0,
            minimum_steps=0,
            maximum_steps=0,
            num_poses=preselect_k,
            initial_best_energy=initial_best,
            final_best_energy=initial_best,
        )
        guide_stats = zero_stats
        release_stats = zero_stats

        optimization_applied = False
    # Match the SDF coordinate precision before filtering/scoring so an exported
    # pose cannot cross the requested boundary due to serialization rounding.
    coordinate_scale = float(10**OUTPUT_COORDINATE_DECIMALS)
    export_final_coords = torch.round(final_coords.to(torch.float64) * coordinate_scale) / coordinate_scale

    # 8) Calculate final distances and discard every row outside [target-tolerance, target+tolerance]
    initial_dists = torch.empty(preselect_k, dtype=torch.float32, device=target_device)
    guided_dists = torch.empty(preselect_k, dtype=torch.float32, device=target_device)
    final_dists = torch.empty(preselect_k, dtype=torch.float64, device=target_device)
    receptor_coord_export = torch.tensor(sel.coordinate, dtype=torch.float64, device=target_device)

    for idx, pivot in enumerate(preselected_pivot_atom_indices.tolist()):
        initial_dists[idx] = torch.linalg.vector_norm(preselected_initial_coords[idx, pivot, :] - receptor_coord_tensor)
        guided_dists[idx] = torch.linalg.vector_norm(guided_coords[idx, pivot, :] - receptor_coord_tensor)
        final_dists[idx] = torch.linalg.vector_norm(export_final_coords[idx, pivot, :] - receptor_coord_export)

    initial_restraint_energies = flat_bottom_distance_restraint(
        initial_dists, target_distance, distance_tolerance, restraint_weight
    )
    guided_restraint_energies = flat_bottom_distance_restraint(
        guided_dists, target_distance, distance_tolerance, restraint_weight
    )
    final_restraint_energies = flat_bottom_distance_restraint(
        final_dists, target_distance, distance_tolerance, restraint_weight
    )

    lower_bound = target_distance - distance_tolerance
    upper_bound = target_distance + distance_tolerance
    valid_mask = (final_dists >= lower_bound) & (final_dists <= upper_bound)

    if not valid_mask.any():
        raise RuntimeError("no poses satisfied the interaction distance restraint after release")

    surviving_indices = torch.nonzero(valid_mask).flatten()

    # Score only the survivors using unmodified scorer
    final_surviving_coords = export_final_coords[surviving_indices]
    initial_surviving_coords = preselected_initial_coords[surviving_indices]

    initial_components, final_components = engine.report_scores(
        problem,
        initial_surviving_coords,
        final_surviving_coords.to(dtype=initial_surviving_coords.dtype),
    )
    if optimization_applied:
        optimization_improved = bool(
            final_components.score.min().item() < initial_components.score.min().item() - 1e-12
        )

    # 9) Write output_dir/interaction_poses.sdf via write_ranked_poses
    search_parameters = {
        "pocket_cutoff": float(pocket_cutoff),
        "include_heteroatoms": bool(include_heteroatoms),
        "num_confs": int(num_confs),
        "rmsd_threshold": float(rmsd_threshold),
        "num_candidates": int(num_candidates),
        "preselect_k": int(preselect_k),
        "max_matches": int(max_matches),
        "max_joint_matches": int(max_joint_matches),
        "optimize": bool(optimize),
        "optimizer": str(optimizer),
        "opt_steps": int(opt_steps),
        "release_steps": int(release_steps),
        "opt_lr": float(opt_lr),
        "opt_batch_size": int(opt_batch_size),
        "scorer": str(problem.scorer.name),
        "torsion_penalty": bool(torsion_penalty),
        "restraint_weight": float(restraint_weight),
        "top_k": int(top_k) if top_k is not None else None,
        "random_seed": int(random_seed),
        "receptor_residue": str(receptor_residue),
        "receptor_atom": str(receptor_atom),
        "ligand_smarts": str(ligand_smarts),
        "target_distance": float(target_distance),
        "distance_tolerance": float(distance_tolerance),
    }

    molecule_metadata = {
        "Mode": "interaction",
        "Anchor_Strategy": "explicit_atom_pair_distance",
        "Search_Method": "guided_se3_release" if optimization_applied else "guided_random_placement",
        "Receptor_Residue": sel.residue_id,
        "Receptor_Atom": sel.atom_name,
        "Receptor_RDKit_Index": sel.rdkit_index,
        "Receptor_PDB_Serial": sel.pdb_serial,
        "Receptor_Element": sel.element,
        "Receptor_Coordinate": ",".join(f"{v:.4f}" for v in sel.coordinate),
        "Receptor_Occupancy": sel.occupancy,
        "Ligand_SMARTS": ligand_smarts,
        "Canonical_SMILES": canonical_smiles,
        "Ligand_Index_Space": "canonical_heavy_atom_topology",
        "Ligand_Matches": json.dumps([asdict(match) for match in anchors], sort_keys=True, separators=(",", ":")),
        "Num_Ligand_Matches": match_count,
        "Target_Distance": target_distance,
        "Distance_Tolerance": distance_tolerance,
        "Distance_Lower": target_distance - distance_tolerance,
        "Distance_Upper": target_distance + distance_tolerance,
        "Output_Coordinate_Decimals": OUTPUT_COORDINATE_DECIMALS,
        "Restraint_Formula": RESTRAINT_FORMULA,
        "Restraint_Weight": restraint_weight,
        "Conformer_IDs": ",".join(str(cid) for cid in representative_ids),
        "Num_Conformer_Attempts": num_confs,
        "Num_Candidates": num_candidates,
        "Num_Preselected": preselect_k,
        "Num_Valid": int(valid_mask.sum().item()),
        "Max_Matches_Requested": max_matches,
        "Max_Joint_Matches_Requested": max_joint_matches,
        "Random_Seed": random_seed,
        "Torsion_Penalty_Requested": torsion_penalty,
        "Torsion_Penalty_Applied": problem.scorer.torsion_penalty_applied,
        "Input_Ligand_Rotatable_Bonds": input_rotatable_bonds,
        "Score_Rotatable_Bonds": problem.scorer.effective_rotatable_bonds,
        "Intramolecular_Reference": float(final_components.intramolecular_reference.detach().cpu().item()),
        "Receptor_Structure_Fingerprint": pocket_context.structure_fingerprint,
        "Receptor_Source_Fingerprint": pocket_context.source_fingerprint,
        "Receptor_Structure_Scope": "residue_pocket",
        "Atom_Typing": pocket_context.atom_typing_version,
        "Optimization_Requested": optimize,
        "Optimization_Applied": optimization_applied,
        "Optimization_Improved": optimization_improved,
        "Optimizer": optimizer,
        "Optimization_Steps_Requested": opt_steps,
        "Optimization_Release_Steps": release_steps,
        "Optimization_Learning_Rate": opt_lr,
        "Guide_Optimization_Stats": json.dumps(guide_stats.as_dict(), sort_keys=True, separators=(",", ":")),
        "Release_Optimization_Stats": json.dumps(release_stats.as_dict(), sort_keys=True, separators=(",", ":")),
        "Guide_Objective_Semantics": "scorer_search_energy_plus_flat_bottom_restraint",
        "Release_Objective_Semantics": "scorer_search_energy_only",
        "Search_Parameters": json.dumps(search_parameters, sort_keys=True, separators=(",", ":")),
        "Protonation_Limitation": PROTONATION_LIMITATION,
    }

    pose_ids = []
    surviving_per_pose_metadata = []
    for idx in surviving_indices.tolist():
        cid = preselected_indices[idx]
        match_idx = cid % match_count
        anchor = anchors[match_idx]
        conf_ord = (cid // match_count) % conformer_count
        conf_id = representative_ids[conf_ord]

        pose_ids.append(f"candidate_{cid:05d}")
        surviving_per_pose_metadata.append(
            {
                "Ligand_Anchor_Index": anchor.ligand_atom_index,
                "Ligand_Anchor_Element": anchor.element,
                "Ligand_Anchor_Match_Index": anchor.match_index,
                "Source_Conformer": int(conf_id),
                "Source_Representative_Index": int(conf_ord),
                "Candidate_Coarse_Search_Energy": (f"{float(initial_search_energies[cid].item()):.8f}"),
                "Ligand_Anchor_Match": json.dumps(list(anchor.representative_match), separators=(",", ":")),
                "Ligand_Anchor_Formal_Charge": anchor.formal_charge,
                "Initial_Distance": f"{float(initial_dists[idx].item()):.8f}",
                "Guided_Distance": f"{float(guided_dists[idx].item()):.8f}",
                "Final_Distance": f"{float(final_dists[idx].item()):.8f}",
                "Initial_Restraint_Energy": f"{float(initial_restraint_energies[idx].item()):.8f}",
                "Guided_Restraint_Energy": f"{float(guided_restraint_energies[idx].item()):.8f}",
                "Final_Restraint_Energy": f"{float(final_restraint_energies[idx].item()):.8f}",
                "Restraint_Satisfaction": "True",
            }
        )

    output_path = Path(output_dir) / "interaction_poses.sdf"
    selected = write_ranked_poses(
        ligand,
        final_surviving_coords,
        final_components.score,
        str(output_path),
        scorer_name=problem.scorer.name,
        score_units=problem.scorer.units,
        score_semantics=SCORE_SEMANTICS,
        scorer_fingerprint=problem.scorer.fingerprint,
        search_energies=final_components.search_energy,
        initial_scores=initial_components.score,
        pose_ids=pose_ids,
        top_k=top_k,
        molecule_metadata=molecule_metadata,
        per_pose_metadata=surviving_per_pose_metadata,
    )

    runtime = time.perf_counter() - started
    best_idx = int(torch.argmin(final_components.score).item())
    intramolecular_reference_value = float(final_components.intramolecular_reference.detach().cpu())

    result: dict[str, object] = {
        "mode": "interaction",
        "anchor_dock_version": __version__,
        "output_file": str(output_path),
        "num_poses": int(selected.numel()),
        "num_candidates": num_candidates,
        "num_preselected": preselect_k,
        "valid_poses": int(valid_mask.sum().item()),
        "num_representatives": len(representative_ids),
        "best_score": float(final_components.score[best_idx].detach().cpu()),
        "best_search_energy": float(final_components.search_energy[best_idx].detach().cpu()),
        "score_units": problem.scorer.units,
        "scorer": problem.scorer.name,
        "scorer_fingerprint": problem.scorer.fingerprint,
        "score_semantics": SCORE_SEMANTICS,
        "receptor_residue": sel.residue_id,
        "receptor_atom": sel.atom_name,
        "receptor_atom_element": sel.element,
        "receptor_atom_coordinate": sel.coordinate,
        "receptor_atom_rdkit_index": sel.rdkit_index,
        "receptor_atom_pdb_serial": sel.pdb_serial,
        "receptor_atom_occupancy": sel.occupancy,
        "ligand_smarts": ligand_smarts,
        "ligand_matches": [m.ligand_atom_index for m in anchors],
        "ligand_match_records": [asdict(match) for match in anchors],
        "num_ligand_matches": match_count,
        "target_distance": target_distance,
        "distance_tolerance": distance_tolerance,
        "distance_lower": target_distance - distance_tolerance,
        "distance_upper": target_distance + distance_tolerance,
        "output_coordinate_decimals": OUTPUT_COORDINATE_DECIMALS,
        "restraint_formula": RESTRAINT_FORMULA,
        "restraint_weight": restraint_weight,
        "canonical_smiles": canonical_smiles,
        "receptor_structure_fingerprint": pocket_context.structure_fingerprint,
        "receptor_structure_scope": "residue_pocket",
        "receptor_source_fingerprint": pocket_context.source_fingerprint,
        "atom_typing": pocket_context.atom_typing_version,
        "torsion_penalty_requested": torsion_penalty,
        "torsion_penalty_applied": problem.scorer.torsion_penalty_applied,
        "input_ligand_rotatable_bonds": input_rotatable_bonds,
        "score_rotatable_bonds": problem.scorer.effective_rotatable_bonds,
        "intramolecular_reference": intramolecular_reference_value,
        "optimization_requested": optimize,
        "optimization_applied": optimization_applied,
        "optimization_improved": optimization_improved,
        "optimization_config": {
            "optimizer": optimizer,
            "steps": opt_steps,
            "release_steps": release_steps,
            "learning_rate": opt_lr,
            "batch_size": opt_batch_size,
        },
        "guide_optimization": guide_stats.as_dict(),
        "release_optimization": release_stats.as_dict(),
        "guide_objective_semantics": "scorer_search_energy_plus_flat_bottom_restraint",
        "release_objective_semantics": "scorer_search_energy_only",
        "search_parameters": search_parameters,
        "representative_conformer_ids": [int(value) for value in representative_ids],
        "protonation_limitation": PROTONATION_LIMITATION,
        "optimized": optimization_applied,
        "runtime": runtime,
        "device": str(pocket_context.device),
    }

    if verbose:
        print(
            f"interaction docking complete: {result['num_poses']} poses, "
            f"best={result['best_score']:.4f}, {runtime:.2f}s"
        )

    return result
