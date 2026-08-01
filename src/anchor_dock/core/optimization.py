"""Gradient-based torsion optimization against the shared scoring engine."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import torch
from rdkit import Chem

from .kinematics import LigandKinematics
from .masks import compute_intramolecular_mask
from .scoring import precompute_interaction_matrices, vina_scoring


def optimize_torsions_vina(
    mol: Chem.Mol,
    ref_indices: Iterable[int],
    init_coords: torch.Tensor,
    pocket_coords: torch.Tensor,
    query_features: dict[str, torch.Tensor],
    pocket_features: dict[str, torch.Tensor],
    device: torch.device | str,
    num_steps: int = 100,
    lr: float = 0.1,
    freeze_anchor: bool = True,
    num_rotatable_bonds: int | None = None,
    weight_preset: str = "vina",
    batch_size: int = 8,
    optimizer: Literal["adam", "adamw", "lbfgs"] = "adam",
    early_stopping: bool = True,
    patience: int = 30,
    min_delta: float = 1e-5,
    intermolecular_exclusion_mask: torch.Tensor | None = None,
    precomputed_matrices: dict[str, torch.Tensor] | None = None,
    return_stats: bool = False,
    *,
    freeze_mcs: bool | None = None,
    intramolecular_exclude_indices: Iterable[int] | None = None,
):
    """Optimize torsions while preserving a selected anchor frame."""
    if freeze_mcs is not None:
        freeze_anchor = freeze_mcs
    device = torch.device(device)
    coords = init_coords.to(device=device, dtype=torch.float32)
    single_pose = coords.ndim == 2
    if single_pose:
        coords = coords.unsqueeze(0)
    if coords.ndim != 3:
        raise ValueError("init_coords must be [N,3] or [B,N,3]")
    if num_steps < 0 or batch_size <= 0:
        raise ValueError("num_steps must be non-negative and batch_size positive")

    intra_mask = compute_intramolecular_mask(mol, device, intramolecular_exclude_indices)
    if precomputed_matrices is None:
        precomputed_matrices = precompute_interaction_matrices(query_features, pocket_features, device)

    probe = LigandKinematics(mol, ref_indices, coords[0], device, freeze_anchor=freeze_anchor)
    if probe.num_torsions == 0 or num_steps == 0:
        result = coords[0] if single_pose else coords.clone()
        stats = {"avg_steps": 0.0, "min_steps": 0, "max_steps": 0, "n_poses": coords.shape[0]}
        return (result, stats) if return_stats else result

    output = torch.empty_like(coords)
    step_counts = torch.zeros(coords.shape[0], dtype=torch.long, device=device)

    def score(batch_coords: torch.Tensor) -> torch.Tensor:
        return vina_scoring(
            batch_coords,
            pocket_coords,
            query_features,
            pocket_features,
            num_rotatable_bonds,
            weight_preset,
            intramolecular_mask=intra_mask,
            precomputed_matrices=precomputed_matrices,
            intermolecular_exclusion_mask=intermolecular_exclusion_mask,
        )

    for start in range(0, coords.shape[0], batch_size):
        stop = min(start + batch_size, coords.shape[0])
        batch = coords[start:stop]
        if optimizer == "lbfgs":
            optimized: list[torch.Tensor] = []
            for local_idx in range(batch.shape[0]):
                model = LigandKinematics(mol, ref_indices, batch[local_idx], device, freeze_anchor=freeze_anchor)
                opt = torch.optim.LBFGS(
                    model.parameters(), lr=lr, max_iter=20, history_size=10, line_search_fn="strong_wolfe"
                )
                best = float("inf")
                stale = 0
                for _ in range(num_steps):
                    def closure() -> torch.Tensor:
                        opt.zero_grad(set_to_none=True)
                        loss = score(model().unsqueeze(0)).sum()
                        loss.backward()
                        return loss

                    loss_value = float(opt.step(closure).detach())
                    step_counts[start + local_idx] += 1
                    if loss_value < best - min_delta:
                        best, stale = loss_value, 0
                    else:
                        stale += 1
                    if early_stopping and stale >= patience:
                        break
                with torch.no_grad():
                    optimized.append(model())
            output[start:stop] = torch.stack(optimized)
            continue

        model = LigandKinematics(mol, ref_indices, batch, device, freeze_anchor=freeze_anchor)
        if optimizer == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer == "adamw":
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        converged = torch.zeros(batch.shape[0], dtype=torch.bool, device=device)
        stale = torch.zeros(batch.shape[0], dtype=torch.long, device=device)
        best = torch.full((batch.shape[0],), float("inf"), device=device)
        for _ in range(num_steps):
            active = ~converged if early_stopping else torch.ones_like(converged)
            if not active.any():
                break
            opt.zero_grad(set_to_none=True)
            current = model()
            losses = score(current)
            losses[active].sum().backward()
            if model.thetas.grad is not None and early_stopping:
                model.thetas.grad[converged] = 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            active_idx = torch.nonzero(active, as_tuple=False).flatten()
            step_counts[start + active_idx] += 1
            detached = losses.detach()
            improved = detached < best - min_delta
            best = torch.where(improved, detached, best)
            stale = torch.where(improved, torch.zeros_like(stale), stale + 1)
            converged = stale >= patience
        with torch.no_grad():
            output[start:stop] = model()

    result = output[0] if single_pose else output
    stats = {
        "avg_steps": float(step_counts.float().mean().item()),
        "min_steps": int(step_counts.min().item()),
        "max_steps": int(step_counts.max().item()),
        "n_poses": int(coords.shape[0]),
    }
    return (result, stats) if return_stats else result
