"""Differentiable generic atom-pair distance restraint primitives."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import torch


def _finite_scalar(value: Real, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def interaction_distances(
    coords: torch.Tensor,
    ligand_atom_index: int,
    receptor_coord: torch.Tensor,
) -> torch.Tensor:
    """Return one receptor-to-ligand heavy-atom distance per pose."""
    if not isinstance(coords, torch.Tensor) or not coords.is_floating_point():
        raise TypeError("coords must be a floating-point torch.Tensor")
    if coords.ndim != 3 or coords.shape[-1] != 3 or coords.shape[0] <= 0 or coords.shape[1] <= 0:
        raise ValueError("coords must have non-empty shape [B,N,3]")
    if not isinstance(ligand_atom_index, int) or isinstance(ligand_atom_index, bool):
        raise TypeError("ligand_atom_index must be an integer")
    if not 0 <= ligand_atom_index < coords.shape[1]:
        raise ValueError("ligand_atom_index is out of bounds")
    if not isinstance(receptor_coord, torch.Tensor) or receptor_coord.shape != (3,):
        raise ValueError("receptor_coord must be a torch.Tensor with shape [3]")
    if not torch.isfinite(coords).all() or not torch.isfinite(receptor_coord).all():
        raise ValueError("interaction coordinates must be finite")
    receptor = receptor_coord.to(device=coords.device, dtype=coords.dtype)
    return torch.linalg.vector_norm(coords[:, ligand_atom_index, :] - receptor, dim=1)


def interaction_distance_matrix(
    coords: torch.Tensor,
    ligand_atom_indices: Sequence[int],
    receptor_coords: torch.Tensor,
) -> torch.Tensor:
    """Return receptor-to-ligand distances with shape ``[poses, interactions]``."""
    if not isinstance(coords, torch.Tensor) or not coords.is_floating_point():
        raise TypeError("coords must be a floating-point torch.Tensor")
    if coords.ndim != 3 or coords.shape[-1] != 3 or coords.shape[0] <= 0 or coords.shape[1] <= 0:
        raise ValueError("coords must have non-empty shape [B,N,3]")
    indices = tuple(ligand_atom_indices)
    if not indices:
        raise ValueError("ligand_atom_indices must be non-empty")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
        raise TypeError("ligand_atom_indices must contain integers")
    if any(index < 0 or index >= coords.shape[1] for index in indices):
        raise ValueError("ligand_atom_indices contains an out-of-range value")
    if not isinstance(receptor_coords, torch.Tensor) or receptor_coords.shape != (len(indices), 3):
        raise ValueError("receptor_coords must have shape [K,3]")
    if not torch.isfinite(coords).all() or not torch.isfinite(receptor_coords).all():
        raise ValueError("interaction coordinates must be finite")
    index_tensor = torch.tensor(indices, dtype=torch.long, device=coords.device)
    receptor = receptor_coords.to(device=coords.device, dtype=coords.dtype)
    return torch.linalg.vector_norm(coords[:, index_tensor, :] - receptor[None, :, :], dim=2)


def flat_bottom_distance_restraint(
    distances: torch.Tensor,
    target_distance: float,
    distance_tolerance: float,
    restraint_weight: float,
) -> torch.Tensor:
    """Return ``weight * relu(abs(distance-target)-tolerance)^2`` per pose."""
    if not isinstance(distances, torch.Tensor) or not distances.is_floating_point():
        raise TypeError("distances must be a floating-point torch.Tensor")
    if distances.ndim != 1 or distances.numel() <= 0 or not torch.isfinite(distances).all():
        raise ValueError("distances must be a non-empty finite [B] tensor")
    target = _finite_scalar(target_distance, "target_distance")
    tolerance = _finite_scalar(distance_tolerance, "distance_tolerance")
    weight = _finite_scalar(restraint_weight, "restraint_weight")
    if target <= 0:
        raise ValueError("target_distance must be positive")
    if not 0 < tolerance < target:
        raise ValueError("distance_tolerance must satisfy 0 < tolerance < target_distance")
    if weight < 0:
        raise ValueError("restraint_weight must be non-negative")
    violation = torch.relu(torch.abs(distances - target) - tolerance)
    return weight * violation.square()


def flat_bottom_distance_restraint_matrix(
    distances: torch.Tensor,
    target_distances: Sequence[Real] | torch.Tensor,
    distance_tolerances: Sequence[Real] | torch.Tensor,
    restraint_weights: Sequence[Real] | torch.Tensor,
) -> torch.Tensor:
    """Return one weighted flat-bottom energy per pose and interaction."""
    if not isinstance(distances, torch.Tensor) or not distances.is_floating_point():
        raise TypeError("distances must be a floating-point torch.Tensor")
    if distances.ndim != 2 or distances.shape[0] <= 0 or distances.shape[1] <= 0:
        raise ValueError("distances must have non-empty shape [B,K]")
    if not torch.isfinite(distances).all():
        raise ValueError("distances must be finite")
    count = distances.shape[1]

    def vector(name: str, values: Sequence[Real] | torch.Tensor) -> torch.Tensor:
        try:
            result = torch.as_tensor(values, dtype=distances.dtype, device=distances.device)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite numeric values") from exc
        if result.shape != (count,) or not torch.isfinite(result).all():
            raise ValueError(f"{name} must have finite shape [K]")
        return result

    targets = vector("target_distances", target_distances)
    tolerances = vector("distance_tolerances", distance_tolerances)
    weights = vector("restraint_weights", restraint_weights)
    if (targets <= 0).any():
        raise ValueError("target_distances must be positive")
    if (tolerances <= 0).any() or (tolerances >= targets).any():
        raise ValueError("each distance tolerance must satisfy 0 < tolerance < target")
    if (weights < 0).any():
        raise ValueError("restraint_weights must be non-negative")
    violation = torch.relu(torch.abs(distances - targets[None, :]) - tolerances[None, :])
    return weights[None, :] * violation.square()


def mean_flat_bottom_distance_restraint(
    distances: torch.Tensor,
    target_distances: Sequence[Real] | torch.Tensor,
    distance_tolerances: Sequence[Real] | torch.Tensor,
    restraint_weights: Sequence[Real] | torch.Tensor,
) -> torch.Tensor:
    """Average per-interaction penalties while preserving single-item behavior."""
    return flat_bottom_distance_restraint_matrix(
        distances,
        target_distances,
        distance_tolerances,
        restraint_weights,
    ).mean(dim=1)
