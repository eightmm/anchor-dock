"""Conformer generation and memory-bounded RMSD clustering."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit.ML.Cluster import Butina


def _condensed_kabsch_rmsd(coords: torch.Tensor, chunk_size: int = 64) -> list[float]:
    """Return Butina's lower-triangle RMSD vector without an O(C²) SVD tensor."""
    num_conformers, num_atoms, _ = coords.shape
    if num_conformers <= 1:
        return []
    centered = coords - coords.mean(dim=1, keepdim=True)
    norms = centered.square().sum(dim=(1, 2))
    condensed: list[float] = []

    for start in range(0, num_conformers, chunk_size):
        stop = min(start + chunk_size, num_conformers)
        rows = centered[start:stop]
        covariance = torch.einsum("iac,jad->ijcd", rows, centered)
        # One SVD provides both the singular values and the reflection check.
        u, singular, vh = torch.linalg.svd(covariance.reshape(-1, 3, 3), full_matrices=False)
        singular = singular.reshape(stop - start, num_conformers, 3)
        # Kabsch reflection correction changes the smallest singular value's sign.
        reflected = (torch.linalg.det(vh.transpose(1, 2) @ u.transpose(1, 2)) < 0).reshape(
            stop - start, num_conformers
        )
        singular_sum = singular.sum(dim=2) - 2.0 * singular[:, :, 2] * reflected
        squared = (
            norms[start:stop, None]
            + norms[None, :]
            - 2.0 * singular_sum
        ).clamp_min(0.0)
        rmsd = torch.sqrt(squared / num_atoms)
        for local_idx, row_idx in enumerate(range(start, stop)):
            if row_idx:
                condensed.extend(rmsd[local_idx, :row_idx].detach().cpu().tolist())
    return condensed


def generate_conformers_and_cluster(
    mol: Chem.Mol,
    device: torch.device | str,
    num_confs: int = 1000,
    rmsd_threshold: float = 1.0,
    coord_map: Mapping[int, Point3D] | None = None,
    *,
    exact_constraints_before_clustering: bool = True,
    add_hydrogens: bool = True,
    random_seed: int = 42,
    rmsd_chunk_size: int = 64,
) -> tuple[Chem.Mol, list[int]]:
    """Generate ETKDG conformers and select Butina representatives.

    Force-field relaxation is intentionally not performed here. A docking
    strategy may relax only its selected representatives while enforcing the
    appropriate anchors.
    """
    if num_confs <= 0:
        raise ValueError("num_confs must be positive")
    if rmsd_threshold <= 0:
        raise ValueError("rmsd_threshold must be positive")
    if rmsd_chunk_size <= 0:
        raise ValueError("rmsd_chunk_size must be positive")
    target_device = torch.device(device)
    working = Chem.AddHs(Chem.Mol(mol)) if add_hydrogens else Chem.Mol(mol)

    params = AllChem.ETKDGv3()
    params.pruneRmsThresh = -1.0
    params.randomSeed = int(random_seed)
    params.numThreads = 0
    if coord_map:
        params.SetCoordMap(dict(coord_map))
    conformer_ids = list(AllChem.EmbedMultipleConfs(working, numConfs=int(num_confs), params=params))
    if not conformer_ids and coord_map:
        params = AllChem.ETKDGv3()
        params.pruneRmsThresh = -1.0
        params.randomSeed = int(random_seed)
        params.numThreads = 0
        conformer_ids = list(AllChem.EmbedMultipleConfs(working, numConfs=int(num_confs), params=params))
    if not conformer_ids:
        raise RuntimeError("conformer generation failed")

    if coord_map and exact_constraints_before_clustering:
        for conformer_id in conformer_ids:
            conformer = working.GetConformer(conformer_id)
            for atom_idx, point in coord_map.items():
                conformer.SetAtomPosition(int(atom_idx), Point3D(point.x, point.y, point.z))

    heavy = Chem.RemoveHs(working)
    heavy_ids = list(range(heavy.GetNumConformers()))
    if len(heavy_ids) == 1:
        return heavy, heavy_ids
    coords = torch.stack(
        [torch.tensor(heavy.GetConformer(cid).GetPositions(), dtype=torch.float32) for cid in heavy_ids]
    ).to(target_device)
    condensed = _condensed_kabsch_rmsd(coords, chunk_size=rmsd_chunk_size)
    clusters = Butina.ClusterData(condensed, len(heavy_ids), float(rmsd_threshold), isDistData=True)
    representatives = [int(cluster[0]) for cluster in clusters]
    return heavy, representatives
