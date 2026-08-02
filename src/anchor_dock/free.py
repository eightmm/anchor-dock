"""Unanchored multistart local docking with the Torch engine."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from ._version import __version__
from .core.conformers import generate_conformers_and_cluster
from .core.engine import DockingEngine
from .core.io import load_ligand, load_receptor_context
from .core.optimization import FreePoseModel, OptimizationStats
from .core.output import write_ranked_poses
from .core.scoring import ScorerLike

SCORE_SEMANTICS = "unanchored_multistart_local_pose_ranking"


def _vector3(value: Sequence[float] | torch.Tensor, name: str, device: torch.device) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float32, device=device)
    if result.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values")
    return result


def _sample_uniform_rotation_vectors(
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample Haar-uniform SO(3) rotations as principal axis-angle vectors."""
    quaternions = torch.randn((count, 4), generator=generator)
    quaternions /= torch.linalg.vector_norm(quaternions, dim=1, keepdim=True).clamp_min(1e-12)
    quaternions *= torch.where(quaternions[:, :1] < 0.0, -1.0, 1.0)
    vector = quaternions[:, 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=1, keepdim=True)
    axes = vector / vector_norm.clamp_min(1e-12)
    fallback = torch.zeros_like(axes)
    fallback[:, 0] = 1.0
    axes = torch.where(vector_norm > 1e-12, axes, fallback)
    angles = 2.0 * torch.atan2(vector_norm[:, 0], quaternions[:, 0].clamp_min(0.0))
    return axes * angles[:, None]


def dock_free(
    protein_pdb: str | os.PathLike[str],
    query_ligand: str | os.PathLike[str] | Chem.Mol,
    output_dir: str | os.PathLike[str] = "anchor_dock_free",
    *,
    center: Sequence[float] | torch.Tensor | None = None,
    box_size: Sequence[float] | torch.Tensor = (20.0, 20.0, 20.0),
    num_confs: int = 64,
    rmsd_threshold: float = 1.0,
    num_starts: int = 128,
    optimize: bool = True,
    optimizer: Literal["adam", "adamw", "lbfgs"] = "adam",
    opt_steps: int = 150,
    opt_lr: float = 0.05,
    opt_batch_size: int = 128,
    scorer: ScorerLike = "softdock",
    torsion_penalty: bool = True,
    boundary_weight: float = 10.0,
    top_k: int | None = 20,
    random_seed: int = 42,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    """Run randomized multistart local docking without a geometric anchor.

    This is not AutoDock Vina's global Monte-Carlo search. It is a deterministic
    multistart local Torch baseline over translation, rotation and torsions.
    """
    if num_starts <= 0:
        raise ValueError("num_starts must be positive")
    if boundary_weight < 0:
        raise ValueError("boundary_weight must be non-negative")
    started = time.perf_counter()
    receptor = load_receptor_context(protein_pdb, device)
    ligand, canonical_smiles = load_ligand(query_ligand, add_hydrogens=False)
    ligand, representative_ids = generate_conformers_and_cluster(
        ligand,
        receptor.device,
        num_confs=num_confs,
        rmsd_threshold=rmsd_threshold,
        add_hydrogens=True,
        random_seed=random_seed,
    )
    if not representative_ids:
        raise RuntimeError("free docking conformer generation produced no representatives")
    representative_coords = torch.stack(
        [
            torch.tensor(ligand.GetConformer(conf_id).GetPositions(), dtype=torch.float32)
            for conf_id in representative_ids
        ]
    )
    source_indices = torch.arange(num_starts) % representative_coords.shape[0]
    base_coords = representative_coords[source_indices].to(receptor.device)

    target_center = receptor.coords.mean(dim=0) if center is None else _vector3(center, "center", receptor.device)
    target_box = _vector3(box_size, "box_size", receptor.device)
    if torch.any(target_box <= 0):
        raise ValueError("box_size values must be positive")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(random_seed))
    random_unit = torch.rand((num_starts, 3), generator=generator) - 0.5
    centers = target_center.cpu() + random_unit * target_box.cpu()
    rotation_vectors = _sample_uniform_rotation_vectors(num_starts, generator)

    input_rotatable_bonds = int(rdMolDescriptors.CalcNumRotatableBonds(ligand))
    num_rotatable_bonds = input_rotatable_bonds if torsion_penalty else 0
    engine = DockingEngine(
        scorer,
        device=receptor.device,
        optimizer=optimizer,
        num_steps=opt_steps,
        learning_rate=opt_lr,
        batch_size=opt_batch_size,
    )
    problem = engine.prepare(
        ligand,
        receptor,
        num_rotatable_bonds=num_rotatable_bonds,
    )
    score_rotatable_bonds = problem.scorer.effective_rotatable_bonds
    torsion_penalty_applied = problem.scorer.torsion_penalty_applied
    initial_model = FreePoseModel(ligand, base_coords, centers, rotation_vectors, receptor.device)
    with torch.no_grad():
        initial_coords = initial_model()
    if optimize:
        final_coords, optimization_stats = engine.optimize_free(
            problem,
            base_coords,
            centers=centers,
            rotation_vectors=rotation_vectors,
            box_center=target_center,
            box_size=target_box,
            boundary_weight=boundary_weight,
        )
    else:
        final_coords = initial_coords
        with torch.no_grad():
            initial_energy = problem.scorer.search_energy(initial_coords)
            excess = torch.relu(torch.abs(initial_coords - target_center) - target_box * 0.5)
            initial_energy = initial_energy + excess.square().sum(dim=(1, 2)) * boundary_weight
        best_energy = float(initial_energy.min().detach().cpu())
        optimization_stats = OptimizationStats(
            average_steps=0.0,
            minimum_steps=0,
            maximum_steps=0,
            num_poses=num_starts,
            initial_best_energy=best_energy,
            final_best_energy=best_energy,
        )
    lower = target_center - target_box * 0.5
    upper = target_center + target_box * 0.5
    valid = ((final_coords >= lower) & (final_coords <= upper)).all(dim=(1, 2))
    if not valid.any():
        raise RuntimeError("all free-docking starts ended outside the search box")
    final_coords = final_coords[valid]
    initial_coords = initial_coords[valid]
    source_indices = source_indices[valid.cpu()]
    initial_components, final_components = engine.report_scores(problem, initial_coords, final_coords)
    intramolecular_reference = float(final_components.intramolecular_reference.detach().cpu())
    optimization_applied = optimization_stats.maximum_steps > 0
    optimization_improved = optimization_stats.final_best_energy < optimization_stats.initial_best_energy - 1e-12
    search_parameters = {
        "box_center": [float(value) for value in target_center.detach().cpu()],
        "box_size": [float(value) for value in target_box.detach().cpu()],
        "num_confs": num_confs,
        "rmsd_threshold": rmsd_threshold,
        "num_starts": num_starts,
        "optimize": optimize,
        "optimizer": optimizer,
        "opt_steps": opt_steps,
        "opt_lr": opt_lr,
        "opt_batch_size": opt_batch_size,
        "torsion_penalty_requested": torsion_penalty,
        "boundary_weight": boundary_weight,
        "top_k": top_k,
        "random_seed": random_seed,
    }

    output_path = Path(output_dir) / "free_poses.sdf"
    selected = write_ranked_poses(
        ligand,
        final_coords,
        final_components.score,
        str(output_path),
        scorer_name=problem.scorer.name,
        score_units=problem.scorer.units,
        score_semantics=SCORE_SEMANTICS,
        scorer_fingerprint=problem.scorer.fingerprint,
        search_energies=final_components.search_energy,
        initial_scores=initial_components.score,
        pose_ids=[f"start_{index:05d}" for index in torch.nonzero(valid, as_tuple=False).flatten().tolist()],
        top_k=top_k,
        molecule_metadata={
            "Mode": "free",
            "Anchor_Strategy": "none",
            "Search_Method": ("multistart_local_gradient" if optimization_applied else "multistart_random_placement"),
            "Canonical_SMILES": canonical_smiles,
            "Receptor_Structure_Fingerprint": receptor.structure_fingerprint,
            "Receptor_Structure_Scope": "input_receptor",
            "Receptor_Source_Fingerprint": receptor.source_fingerprint,
            "Atom_Typing": receptor.atom_typing_version,
            "Box_Center": ",".join(f"{float(value):.4f}" for value in target_center.cpu()),
            "Box_Size": ",".join(f"{float(value):.4f}" for value in target_box.cpu()),
            "Random_Seed": random_seed,
            "Optimization_Requested": optimize,
            "Optimization_Applied": optimization_applied,
            "Optimization_Improved": optimization_improved,
            "Optimizer": optimizer,
            "Optimization_Steps_Requested": opt_steps,
            "Optimization_Learning_Rate": opt_lr,
            "Search_Parameters": json.dumps(search_parameters, sort_keys=True, separators=(",", ":")),
            "Torsion_Penalty_Requested": torsion_penalty,
            "Torsion_Penalty_Applied": torsion_penalty_applied,
            "Input_Ligand_Rotatable_Bonds": input_rotatable_bonds,
            "Score_Rotatable_Bonds": score_rotatable_bonds,
            "Intramolecular_Reference": intramolecular_reference,
        },
        per_pose_metadata=[{"Source_Conformer": int(index)} for index in source_indices],
    )
    runtime = time.perf_counter() - started
    best_idx = int(torch.argmin(final_components.score).item())
    result: dict[str, object] = {
        "mode": "free",
        "anchor_dock_version": __version__,
        "output_file": str(output_path),
        "num_poses": int(selected.numel()),
        "num_starts": num_starts,
        "valid_starts": int(valid.sum().item()),
        "num_representatives": len(representative_ids),
        "best_score": float(final_components.score[best_idx].detach().cpu()),
        "best_search_energy": float(final_components.search_energy[best_idx].detach().cpu()),
        "score_units": problem.scorer.units,
        "scorer": problem.scorer.name,
        "scorer_fingerprint": problem.scorer.fingerprint,
        "score_semantics": SCORE_SEMANTICS,
        "canonical_smiles": canonical_smiles,
        "receptor_structure_fingerprint": receptor.structure_fingerprint,
        "receptor_structure_scope": "input_receptor",
        "receptor_source_fingerprint": receptor.source_fingerprint,
        "torsion_penalty_requested": torsion_penalty,
        "torsion_penalty_applied": torsion_penalty_applied,
        "input_ligand_rotatable_bonds": input_rotatable_bonds,
        "score_rotatable_bonds": score_rotatable_bonds,
        "intramolecular_reference": intramolecular_reference,
        "optimization": optimization_stats.as_dict(),
        "search_parameters": search_parameters,
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
        "runtime": runtime,
        "device": str(receptor.device),
    }
    if verbose:
        print(f"free docking complete: {result['num_poses']} poses, best={result['best_score']:.4f}, {runtime:.2f}s")
    return result
