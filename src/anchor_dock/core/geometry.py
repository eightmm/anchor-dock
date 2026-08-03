"""Scorer-independent geometric sampling utilities."""

from __future__ import annotations

import torch


def sample_uniform_rotation_vectors(count: int, generator: torch.Generator) -> torch.Tensor:
    """Sample Haar-uniform SO(3) rotations as principal axis-angle vectors."""
    if count <= 0:
        raise ValueError("count must be positive")
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
