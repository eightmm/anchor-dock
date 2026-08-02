"""Unanchored multistart local docking with the Torch engine."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Literal, Sequence

import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from .core.conformers import generate_conformers_and_cluster
from .core.engine import DockingEngine
from .core.io import load_ligand, load_receptor_context
from .core.output import write_ranked_poses
from .core.scoring import ScorerLike


def _vector3(value: Sequence[float] | torch.Tensor, name: str, device: torch.device) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float32, device=device)
    if result.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values")
    return result


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
    optimizer: Literal["adam", "lbfgs"] = "adam",
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
        [torch.tensor(ligand.GetConformer(conf_id).GetPositions(), dtype=torch.float32) for conf_id in representative_ids]
    )
    source_indices = torch.arange(num_starts) % representative_coords.shape[0]
    base_coords = representative_coords[source_indices].to(receptor.device)

    target_center = (
        receptor.coords.mean(dim=0)
        if center is None
        else _vector3(center, "center", receptor.device)
    )
    target_box = _vector3(box_size, "box_size", receptor.device)
    if torch.any(target_box <= 0):
        raise ValueError("box_size values must be positive")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(random_seed))
    random_unit = torch.rand((num_starts, 3), generator=generator) - 0.5
    centers = target_center.cpu() + random_unit * target_box.cpu()
    axes = torch.randn((num_starts, 3), generator=generator)
    axes = axes / torch.linalg.vector_norm(axes, dim=1, keepdim=True).clamp_min(1e-12)
    angles = torch.rand(num_starts, generator=generator) * torch.pi
    rotation_vectors = axes * angles[:, None]

    num_rotatable_bonds = (
        int(rdMolDescriptors.CalcNumRotatableBonds(ligand)) if torsion_penalty else 0
    )
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
    final_coords, optimization_stats = engine.optimize_free(
        problem,
        base_coords,
        centers=centers,
        rotation_vectors=rotation_vectors,
        box_center=target_center,
        box_size=target_box,
        boundary_weight=boundary_weight,
    )
    # The model's randomized placement is the initial state. Reconstruct it with
    # zero optimization steps through the same parameterization for reporting.
    from .core.optimization import FreePoseModel

    initial_model = FreePoseModel(ligand, base_coords, centers, rotation_vectors, receptor.device)
    with torch.no_grad():
        initial_coords = initial_model()
    lower = target_center - target_box * 0.5
    upper = target_center + target_box * 0.5
    valid = ((final_coords >= lower) & (final_coords <= upper)).all(dim=(1, 2))
    if not valid.any():
        raise RuntimeError("all free-docking starts ended outside the search box")
    final_coords = final_coords[valid]
    initial_coords = initial_coords[valid]
    source_indices = source_indices[valid.cpu()]
    initial_components, final_components = engine.report_scores(problem, initial_coords, final_coords)

    output_path = Path(output_dir) / "free_poses.sdf"
    selected = write_ranked_poses(
        ligand,
        final_coords,
        final_components.score,
        str(output_path),
        scorer_name=problem.scorer.name,
        score_units=problem.scorer.units,
        search_energies=final_components.search_energy,
        initial_scores=initial_components.score,
        pose_ids=[f"start_{index:05d}" for index in torch.nonzero(valid, as_tuple=False).flatten().tolist()],
        top_k=top_k,
        molecule_metadata={
            "Mode": "free",
            "Anchor_Strategy": "none",
            "Search_Method": "multistart_local_gradient",
            "Canonical_SMILES": canonical_smiles,
            "Atom_Typing": receptor.atom_typing_version,
            "Box_Center": ",".join(f"{float(value):.4f}" for value in target_center.cpu()),
            "Box_Size": ",".join(f"{float(value):.4f}" for value in target_box.cpu()),
            "Random_Seed": random_seed,
        },
        per_pose_metadata=[{"Source_Conformer": int(index)} for index in source_indices],
    )
    runtime = time.perf_counter() - started
    best_idx = int(torch.argmin(final_components.score).item())
    result: dict[str, object] = {
        "mode": "free",
        "output_file": str(output_path),
        "num_poses": int(selected.numel()),
        "num_starts": num_starts,
        "valid_starts": int(valid.sum().item()),
        "num_representatives": len(representative_ids),
        "best_score": float(final_components.score[best_idx].detach().cpu()),
        "best_search_energy": float(final_components.search_energy[best_idx].detach().cpu()),
        "score_units": problem.scorer.units,
        "scorer": problem.scorer.name,
        "score_semantics": "unanchored_multistart_local_pose_ranking",
        "canonical_smiles": canonical_smiles,
        "optimization": optimization_stats.as_dict(),
        "runtime": runtime,
        "device": str(receptor.device),
    }
    if verbose:
        print(
            f"free docking complete: {result['num_poses']} poses, "
            f"best={result['best_score']:.4f}, {runtime:.2f}s"
        )
    return result
