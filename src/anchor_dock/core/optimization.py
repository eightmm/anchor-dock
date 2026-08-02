"""Scorer-independent differentiable pose optimization."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from rdkit import Chem

from .kinematics import LigandKinematics, get_batched_rotation_matrix
from .scoring import PreparedScorer

OptimizerName = Literal["adam", "lbfgs"]


@dataclass(frozen=True)
class OptimizationStats:
    average_steps: float
    minimum_steps: int
    maximum_steps: int
    num_poses: int
    initial_best_energy: float
    final_best_energy: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "average_steps": self.average_steps,
            "minimum_steps": self.minimum_steps,
            "maximum_steps": self.maximum_steps,
            "num_poses": self.num_poses,
            "initial_best_energy": self.initial_best_energy,
            "final_best_energy": self.final_best_energy,
        }


def _batch_parameters(model: nn.Module, batch_size: int) -> list[nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.numel() > 0]
    for parameter in parameters:
        if parameter.ndim == 0 or parameter.shape[0] != batch_size:
            raise ValueError("all optimizable pose parameters must use the pose batch as their first dimension")
    return parameters


def _restore_rows(parameters: list[nn.Parameter], snapshots: list[torch.Tensor], rows: torch.Tensor) -> None:
    if not rows.any():
        return
    with torch.no_grad():
        for parameter, snapshot in zip(parameters, snapshots, strict=True):
            parameter[rows] = snapshot[rows]


def _update_best_rows(
    parameters: list[nn.Parameter],
    snapshots: list[torch.Tensor],
    improved: torch.Tensor,
) -> None:
    if not improved.any():
        return
    with torch.no_grad():
        for parameter, snapshot in zip(parameters, snapshots, strict=True):
            snapshot[improved] = parameter.detach()[improved]


def _clear_adam_rows(optimizer: torch.optim.Optimizer, parameters: list[nn.Parameter], rows: torch.Tensor) -> None:
    if not rows.any():
        return
    for parameter in parameters:
        state = optimizer.state.get(parameter, {})
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(key)
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == rows.shape[0]:
                value[rows] = 0


def _checked_losses(values: torch.Tensor, batch_size: int) -> torch.Tensor:
    if values.ndim == 0 and batch_size == 1:
        values = values.reshape(1)
    if values.shape != (batch_size,):
        raise ValueError(f"energy function must return shape [{batch_size}], got {tuple(values.shape)}")
    if not torch.isfinite(values).all():
        raise FloatingPointError("energy function returned NaN or infinity")
    return values


def optimize_pose_module(
    model: nn.Module,
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    num_steps: int = 100,
    learning_rate: float = 0.05,
    optimizer: OptimizerName = "adam",
    early_stopping: bool = True,
    patience: int = 30,
    min_delta: float = 1e-5,
    gradient_clip: float | None = 1.0,
) -> tuple[torch.Tensor, OptimizationStats]:
    """Optimize a pose module and restore the best state for every pose."""
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if patience <= 0:
        raise ValueError("patience must be positive")
    if min_delta < 0:
        raise ValueError("min_delta must be non-negative")
    if gradient_clip is not None and gradient_clip <= 0:
        raise ValueError("gradient_clip must be positive or None")
    initial_coords = model()
    if initial_coords.ndim == 2:
        initial_coords = initial_coords.unsqueeze(0)
    batch_size = initial_coords.shape[0]
    parameters = _batch_parameters(model, batch_size)

    with torch.no_grad():
        initial_losses = _checked_losses(energy_fn(initial_coords), batch_size).detach()
    if not parameters or num_steps == 0:
        stats = OptimizationStats(
            0.0,
            0,
            0,
            batch_size,
            float(initial_losses.min().cpu()),
            float(initial_losses.min().cpu()),
        )
        return model(), stats

    if optimizer == "lbfgs" and batch_size != 1:
        raise ValueError("LBFGS optimization is performed one pose at a time")

    best_losses = initial_losses.clone()
    best_parameters = [parameter.detach().clone() for parameter in parameters]
    stale = torch.zeros(batch_size, dtype=torch.long, device=initial_losses.device)
    converged = torch.zeros(batch_size, dtype=torch.bool, device=initial_losses.device)
    step_counts = torch.zeros(batch_size, dtype=torch.long, device=initial_losses.device)

    if optimizer == "adam":
        torch_optimizer: torch.optim.Optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    elif optimizer == "lbfgs":
        torch_optimizer = torch.optim.LBFGS(
            parameters,
            lr=learning_rate,
            max_iter=1,
            max_eval=5,
            history_size=10,
            line_search_fn="strong_wolfe",
        )
    else:
        raise ValueError(f"unknown optimizer: {optimizer}")

    for _ in range(num_steps):
        active = ~converged if early_stopping else torch.ones_like(converged)
        if not active.any():
            break

        if optimizer == "lbfgs":
            def closure() -> torch.Tensor:
                torch_optimizer.zero_grad(set_to_none=True)
                losses = _checked_losses(energy_fn(model()), batch_size)
                loss = losses[active].sum()
                loss.backward()
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
                return loss

            torch_optimizer.step(closure)
        else:
            torch_optimizer.zero_grad(set_to_none=True)
            losses = _checked_losses(energy_fn(model()), batch_size)
            loss = losses[active].sum()
            if not loss.requires_grad:
                raise RuntimeError("energy function is not differentiable with respect to pose parameters")
            loss.backward()
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad[~active] = 0
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
            torch_optimizer.step()
            _restore_rows(parameters, best_parameters, ~active)
            _clear_adam_rows(torch_optimizer, parameters, ~active)

        step_counts[active] += 1
        with torch.no_grad():
            new_losses = _checked_losses(energy_fn(model()), batch_size).detach()
        improved = active & (new_losses < best_losses - min_delta)
        _update_best_rows(parameters, best_parameters, improved)
        best_losses = torch.where(improved, new_losses, best_losses)
        stale = torch.where(improved, torch.zeros_like(stale), torch.where(active, stale + 1, stale))
        if early_stopping:
            converged |= stale >= patience

    _restore_rows(parameters, best_parameters, torch.ones_like(converged))
    final_coords = model()
    stats = OptimizationStats(
        average_steps=float(step_counts.float().mean().cpu()),
        minimum_steps=int(step_counts.min().cpu()),
        maximum_steps=int(step_counts.max().cpu()),
        num_poses=batch_size,
        initial_best_energy=float(initial_losses.min().cpu()),
        final_best_energy=float(best_losses.min().cpu()),
    )
    return final_coords, stats


def optimize_torsions(
    mol: Chem.Mol,
    anchor_indices: Iterable[int],
    init_coords: torch.Tensor,
    scorer: PreparedScorer,
    device: torch.device | str,
    *,
    num_steps: int = 100,
    learning_rate: float = 0.05,
    batch_size: int = 128,
    optimizer: OptimizerName = "adam",
    early_stopping: bool = True,
    patience: int = 30,
    min_delta: float = 1e-5,
    freeze_anchor: bool = True,
) -> tuple[torch.Tensor, OptimizationStats]:
    """Optimize torsions for one molecule topology in memory-bounded chunks."""
    device = torch.device(device)
    coords = init_coords.to(device=device, dtype=torch.float32)
    single_pose = coords.ndim == 2
    if single_pose:
        coords = coords.unsqueeze(0)
    if coords.ndim != 3:
        raise ValueError("init_coords must have shape [N,3] or [B,N,3]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    output = torch.empty_like(coords)
    all_steps: list[float] = []
    minimum_steps: list[int] = []
    maximum_steps: list[int] = []
    initial_best: list[float] = []
    final_best: list[float] = []
    for start in range(0, coords.shape[0], batch_size):
        stop = min(start + batch_size, coords.shape[0])
        chunk = coords[start:stop]
        if optimizer == "lbfgs":
            chunk_outputs: list[torch.Tensor] = []
            chunk_stats: list[OptimizationStats] = []
            for pose in chunk:
                model = LigandKinematics(mol, anchor_indices, pose, device, freeze_anchor=freeze_anchor)
                optimized, stats = optimize_pose_module(
                    model,
                    lambda values: scorer.search_energy(values),
                    num_steps=num_steps,
                    learning_rate=learning_rate,
                    optimizer="lbfgs",
                    early_stopping=early_stopping,
                    patience=patience,
                    min_delta=min_delta,
                )
                chunk_outputs.append(optimized)
                chunk_stats.append(stats)
            output[start:stop] = torch.stack(chunk_outputs)
            stats_for_chunk = chunk_stats
        else:
            model = LigandKinematics(mol, anchor_indices, chunk, device, freeze_anchor=freeze_anchor)
            optimized, stats = optimize_pose_module(
                model,
                scorer.search_energy,
                num_steps=num_steps,
                learning_rate=learning_rate,
                optimizer="adam",
                early_stopping=early_stopping,
                patience=patience,
                min_delta=min_delta,
            )
            output[start:stop] = optimized
            stats_for_chunk = [stats]

        for stats in stats_for_chunk:
            all_steps.extend([stats.average_steps] * stats.num_poses)
            minimum_steps.append(stats.minimum_steps)
            maximum_steps.append(stats.maximum_steps)
            initial_best.append(stats.initial_best_energy)
            final_best.append(stats.final_best_energy)

    aggregate = OptimizationStats(
        average_steps=float(sum(all_steps) / len(all_steps)) if all_steps else 0.0,
        minimum_steps=min(minimum_steps, default=0),
        maximum_steps=max(maximum_steps, default=0),
        num_poses=coords.shape[0],
        initial_best_energy=min(initial_best, default=0.0),
        final_best_energy=min(final_best, default=0.0),
    )
    result = output[0] if single_pose else output
    return result, aggregate


class FreePoseModel(nn.Module):
    """Torsion + rigid SE(3) parameters for unanchored local pose search."""

    def __init__(
        self,
        mol: Chem.Mol,
        base_coords: torch.Tensor,
        initial_centers: torch.Tensor,
        initial_rotation_vectors: torch.Tensor,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        device = torch.device(device)
        coords = base_coords.to(device=device, dtype=torch.float32)
        if coords.ndim != 3:
            raise ValueError("base_coords must have shape [B,N,3]")
        centered = coords - coords.mean(dim=1, keepdim=True)
        self.kinematics = LigandKinematics(mol, (), centered, device, freeze_anchor=False)
        self.translations = nn.Parameter(initial_centers.to(device=device, dtype=torch.float32).clone())
        self.rotation_vectors = nn.Parameter(
            initial_rotation_vectors.to(device=device, dtype=torch.float32).clone()
        )

    def forward(self) -> torch.Tensor:
        flexible = self.kinematics()
        center = flexible.mean(dim=1, keepdim=True)
        centered = flexible - center
        angle = torch.linalg.vector_norm(self.rotation_vectors, dim=1)
        rotation = get_batched_rotation_matrix(self.rotation_vectors, angle)
        rotated = torch.matmul(centered, rotation.transpose(1, 2))
        return rotated + self.translations[:, None, :]
