"""Differentiable generic atom-pair distance restraint primitives."""

from __future__ import annotations

import math
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
