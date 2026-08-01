"""Cached loading utilities for repeatedly reused protein pockets."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import torch
from rdkit import Chem


@dataclass(frozen=True)
class PocketBundle:
    mol: Chem.Mol
    coords: torch.Tensor
    features: dict


_POCKET_CACHE: dict[tuple[str, int, int, str], PocketBundle] = {}


def _cache_key(protein_pdb: str, device: torch.device) -> tuple[str, int, int, str]:
    abs_path = os.path.abspath(protein_pdb)
    stat = os.stat(abs_path)
    return abs_path, stat.st_mtime_ns, stat.st_size, str(device)


def load_pocket_bundle(
    protein_pdb: str,
    device: torch.device,
    feature_builder: Callable[[Chem.Mol], dict],
) -> PocketBundle:
    """
    Load and cache pocket molecule data for repeated runs on the same receptor.

    The cache is invalidated when the pocket file path, modified time, file size,
    or target device changes.
    """
    key = _cache_key(protein_pdb, device)
    cached = _POCKET_CACHE.get(key)
    if cached is not None:
        return cached

    pocket_mol = Chem.MolFromPDBFile(protein_pdb, sanitize=False, removeHs=True)
    if pocket_mol is None:
        raise ValueError(f"Failed to load protein pocket from {protein_pdb}")

    pocket_coords = torch.tensor(
        pocket_mol.GetConformer().GetPositions(),
        dtype=torch.float32,
        device=device,
    )
    pocket_features = feature_builder(pocket_mol)

    bundle = PocketBundle(
        mol=pocket_mol,
        coords=pocket_coords,
        features=pocket_features,
    )
    _POCKET_CACHE[key] = bundle
    return bundle


def clear_pocket_cache() -> None:
    """Clear all cached pocket bundles."""
    _POCKET_CACHE.clear()
