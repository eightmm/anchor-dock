"""Conformer generation and clustering primitives."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit.ML.Cluster import Butina


def _batched_kabsch_rmsd(coords: torch.Tensor) -> torch.Tensor:
    """Return a full pairwise Kabsch RMSD matrix for ``[C,N,3]`` coordinates."""
    centered = coords - coords.mean(dim=1, keepdim=True)
    covariance = torch.einsum("iac,jad->ijcd", centered, centered)
    flat = covariance.reshape(-1, 3, 3)
    u, singular, vh = torch.linalg.svd(flat)
    rotation = vh.transpose(1, 2) @ u.transpose(1, 2)
    reflected = torch.linalg.det(rotation) < 0
    singular_sum = singular.sum(dim=1) - 2.0 * singular[:, 2] * reflected
    norms = centered.square().sum(dim=(1, 2))
    squared = norms[:, None] + norms[None, :] - 2.0 * singular_sum.reshape(coords.shape[0], coords.shape[0])
    return torch.sqrt(squared.clamp_min(0.0) / coords.shape[1])


def generate_conformers_and_cluster(
    mol: Chem.Mol,
    device: torch.device | str,
    num_confs: int = 1000,
    rmsd_threshold: float = 1.0,
    coordMap: Mapping[int, Point3D] | None = None,
    *,
    exact_coordmap_before_clustering: bool = True,
    add_hydrogens: bool = True,
    random_seed: int = 42,
) -> tuple[Chem.Mol, list[int]]:
    """Generate ETKDG conformers and select Butina cluster representatives.

    No force-field minimization is applied. Strategy-specific relaxation should
    happen after representative selection, avoiding topology-dependent MMFF
    failures for covalent adducts.
    """
    if num_confs <= 0:
        raise ValueError("num_confs must be positive")
    if rmsd_threshold <= 0:
        raise ValueError("rmsd_threshold must be positive")
    device = torch.device(device)
    working = Chem.AddHs(Chem.Mol(mol)) if add_hydrogens else Chem.Mol(mol)

    kwargs = {
        "numConfs": int(num_confs),
        "pruneRmsThresh": -1.0,
        "randomSeed": int(random_seed),
        "numThreads": 0,
        "ETversion": 2,
    }
    if coordMap:
        kwargs["coordMap"] = dict(coordMap)
    conformer_ids = list(AllChem.EmbedMultipleConfs(working, **kwargs))
    if not conformer_ids and coordMap:
        kwargs.pop("coordMap", None)
        conformer_ids = list(AllChem.EmbedMultipleConfs(working, **kwargs))
    if not conformer_ids:
        raise RuntimeError("Conformer generation failed")

    if coordMap and exact_coordmap_before_clustering:
        for conformer_id in conformer_ids:
            conformer = working.GetConformer(conformer_id)
            for atom_idx, point in coordMap.items():
                conformer.SetAtomPosition(int(atom_idx), Point3D(point.x, point.y, point.z))

    heavy = Chem.RemoveHs(working)
    heavy_ids = list(range(heavy.GetNumConformers()))
    if len(heavy_ids) == 1:
        return heavy, [heavy_ids[0]]

    coords = torch.stack(
        [torch.tensor(heavy.GetConformer(cid).GetPositions(), dtype=torch.float32) for cid in heavy_ids]
    ).to(device)
    rmsd = _batched_kabsch_rmsd(coords)
    upper = torch.triu_indices(len(heavy_ids), len(heavy_ids), offset=1, device=device)
    condensed = rmsd[upper[1], upper[0]].detach().cpu().tolist()
    clusters = Butina.ClusterData(condensed, len(heavy_ids), float(rmsd_threshold), isDistData=True)
    representatives = [int(cluster[0]) for cluster in clusters]
    return heavy, representatives
