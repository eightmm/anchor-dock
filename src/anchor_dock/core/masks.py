"""Interaction masks shared by reference and covalent docking."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from rdkit import Chem


def compute_intramolecular_mask(
    mol: Chem.Mol,
    device: torch.device | str,
    exclude_atom_indices: Iterable[int] | None = None,
) -> torch.Tensor:
    """Return an ``[N, N]`` mask of non-local intramolecular interactions.

    1-2, 1-3 and same-ring pairs are excluded. Optional atom indices can be
    removed entirely, which is useful for protein-derived atoms attached to a
    covalent ligand adduct.
    """
    device = torch.device(device)
    graph_distance = Chem.GetDistanceMatrix(mol)
    mask = torch.from_numpy(graph_distance > 2).to(device=device, dtype=torch.bool)

    for ring in mol.GetRingInfo().AtomRings():
        ring_idx = torch.tensor(ring, dtype=torch.long, device=device)
        mask[ring_idx[:, None], ring_idx[None, :]] = False

    if exclude_atom_indices:
        valid = [idx for idx in exclude_atom_indices if 0 <= idx < mol.GetNumAtoms()]
        if valid:
            idx = torch.tensor(valid, dtype=torch.long, device=device)
            mask[idx, :] = False
            mask[:, idx] = False

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
