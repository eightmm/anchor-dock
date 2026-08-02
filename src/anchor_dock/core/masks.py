"""Interaction masks shared by all AnchorDock strategies."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from rdkit import Chem

from .topology import build_rigid_topology


def compute_intramolecular_mask(
    mol: Chem.Mol,
    device: torch.device | str,
    exclude_atom_indices: Iterable[int] | None = None,
) -> torch.Tensor:
    """Return variable non-local atom pairs for intramolecular scoring.

    The mask excludes 1-2, 1-3 and 1-4 pairs and all pairs inside one rigid
    frame. Those distances do not change under torsional optimization and should
    not contribute to the pose-search objective.
    """
    device = torch.device(device)
    graph_distance = Chem.GetDistanceMatrix(mol)
    mask = torch.from_numpy(graph_distance > 3).to(device=device, dtype=torch.bool)

    topology = build_rigid_topology(mol)
    frame_ids = torch.tensor(topology.atom_to_frame, dtype=torch.long, device=device)
    mask &= frame_ids[:, None] != frame_ids[None, :]

    if exclude_atom_indices:
        valid = sorted({int(idx) for idx in exclude_atom_indices if 0 <= int(idx) < mol.GetNumAtoms()})
        if valid:
            indices = torch.tensor(valid, dtype=torch.long, device=device)
            mask[indices, :] = False
            mask[:, indices] = False

    mask.fill_diagonal_(False)
    return mask


def normalize_pair_mask(
    mask: torch.Tensor | None,
    batch_size: int,
    rows: int,
    cols: int,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    """Validate and broadcast a pair mask to ``[B, rows, cols]``."""
    if mask is None:
        return None
    mask = mask.to(device=device, dtype=torch.bool)
    if mask.ndim == 2:
        if tuple(mask.shape) != (rows, cols):
            raise ValueError(f"pair mask shape {tuple(mask.shape)} != {(rows, cols)}")
        return mask.unsqueeze(0).expand(batch_size, -1, -1)
    if mask.ndim == 3:
        if tuple(mask.shape[1:]) != (rows, cols):
            raise ValueError(f"pair mask shape {tuple(mask.shape)} incompatible with {(rows, cols)}")
        if mask.shape[0] == 1:
            return mask.expand(batch_size, -1, -1)
        if mask.shape[0] != batch_size:
            raise ValueError(f"pair mask batch {mask.shape[0]} != {batch_size}")
        return mask
    raise ValueError(f"pair mask must be rank 2 or 3, got rank {mask.ndim}")
