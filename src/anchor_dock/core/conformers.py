"""Conformer generation and memory-bounded RMSD clustering."""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom
from rdkit.Geometry import Point3D
from rdkit.ML.Cluster import Butina

_LOCAL_DISTANCE_BOUNDS_TOLERANCE = 0.10
_ABSOLUTE_ANCHOR_TOLERANCE = 1e-6


def _has_invalid_bond_geometry(
    mol: Chem.Mol,
    positions: np.ndarray,
    exempt_pairs: frozenset[tuple[int, int]],
) -> bool:
    bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    graph_distances = Chem.GetDistanceMatrix(mol)
    for first in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(first).GetAtomicNum() == 1:
            continue
        for second in range(first + 1, mol.GetNumAtoms()):
            if (first, second) in exempt_pairs:
                continue
            if mol.GetAtomWithIdx(second).GetAtomicNum() == 1:
                continue
            # Bonds and 1-3 distances protect lengths and valence angles.
            # 1-4 distances are torsion-dependent and may legitimately lie
            # outside distance-geometry defaults in experimental references.
            if not 0.0 < graph_distances[first, second] <= 2.0:
                continue
            distance = float(np.linalg.norm(positions[first] - positions[second]))
            lower = float(bounds[second, first])
            upper = float(bounds[first, second])
            if not np.isfinite(distance) or not (
                lower - _LOCAL_DISTANCE_BOUNDS_TOLERANCE <= distance <= upper + _LOCAL_DISTANCE_BOUNDS_TOLERANCE
            ):
                return True
    return False


def _validate_coordinate_map(
    coord_map: Mapping[int, Point3D],
    num_atoms: int,
) -> dict[int, Point3D]:
    validated: dict[int, Point3D] = {}
    for raw_index, point in coord_map.items():
        try:
            atom_index = operator.index(raw_index)
        except TypeError as exc:
            raise ValueError(f"coordinate-map atom index {raw_index!r} is not an integer") from exc
        if not 0 <= atom_index < num_atoms:
            raise ValueError(f"coordinate-map atom index {atom_index} outside 0..{num_atoms - 1}")
        try:
            xyz = np.asarray([float(point.x), float(point.y), float(point.z)], dtype=float)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"coordinate-map value for atom {atom_index} is not a Point3D") from exc
        if not np.isfinite(xyz).all():
            raise ValueError(f"coordinate-map value for atom {atom_index} must be finite")
        validated[atom_index] = Point3D(*map(float, xyz))
    return validated


def _validate_conformer_against_map(
    mol: Chem.Mol,
    conformer_id: int,
    coord_map: Mapping[int, Point3D],
    exempt_pairs: frozenset[tuple[int, int]],
) -> bool:
    conformer = mol.GetConformer(conformer_id)
    positions = conformer.GetPositions()
    anchor_indices = sorted(int(index) for index in coord_map)
    target = np.asarray(
        [[coord_map[index].x, coord_map[index].y, coord_map[index].z] for index in anchor_indices],
        dtype=float,
    )
    anchor_residuals = np.linalg.norm(positions[anchor_indices] - target, axis=1)
    if not np.isfinite(positions).all() or not np.isfinite(anchor_residuals).all():
        return False
    if float(anchor_residuals.max(initial=0.0)) > _ABSOLUTE_ANCHOR_TOLERANCE:
        return False
    return not _has_invalid_bond_geometry(mol, positions, exempt_pairs)


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
        reflected = (torch.linalg.det(vh.transpose(1, 2) @ u.transpose(1, 2)) < 0).reshape(stop - start, num_conformers)
        singular_sum = singular.sum(dim=2) - 2.0 * singular[:, :, 2] * reflected
        squared = (norms[start:stop, None] + norms[None, :] - 2.0 * singular_sum).clamp_min(0.0)
        rmsd = torch.sqrt(squared / num_atoms)
        for local_idx, row_idx in enumerate(range(start, stop)):
            if row_idx:
                condensed.extend(rmsd[local_idx, :row_idx].detach().cpu().tolist())
    return condensed


def _condensed_direct_rmsd(coords: torch.Tensor, chunk_size: int = 64) -> list[float]:
    """Return absolute-frame RMSD for receptor-anchored conformers."""
    num_conformers, num_atoms, _ = coords.shape
    if num_conformers <= 1:
        return []
    norms = coords.square().sum(dim=(1, 2))
    condensed: list[float] = []
    for start in range(0, num_conformers, chunk_size):
        stop = min(start + chunk_size, num_conformers)
        cross = torch.einsum("iac,jac->ij", coords[start:stop], coords)
        squared = (norms[start:stop, None] + norms[None, :] - 2.0 * cross).clamp_min(0.0)
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
    distance_bound_exempt_pairs: set[tuple[int, int]] | None = None,
    conformer_preprocessor: Callable[[Chem.Mol, int], bool] | None = None,
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
    if coord_map:
        coord_map = _validate_coordinate_map(coord_map, mol.GetNumAtoms())
    target_device = torch.device(device)
    working = Chem.AddHs(Chem.Mol(mol)) if add_hydrogens else Chem.Mol(mol)

    params = AllChem.ETKDGv3()
    params.pruneRmsThresh = -1.0
    params.randomSeed = int(random_seed)
    params.numThreads = 0
    if coord_map:
        params.SetCoordMap(dict(coord_map))
        # RDKit otherwise treats coordMap primarily as distance constraints.
        # Random-coordinate embedding honors the supplied absolute frame, which
        # avoids distorting exact reference or protein-side anchors afterwards.
        params.useRandomCoords = True
    conformer_ids = list(AllChem.EmbedMultipleConfs(working, numConfs=int(num_confs), params=params))
    if not conformer_ids:
        constraint_note = " with coordinate constraints" if coord_map else ""
        raise RuntimeError(f"conformer generation failed{constraint_note}")

    if (coord_map and exact_constraints_before_clustering) or conformer_preprocessor is not None:
        exempt_pairs = frozenset(
            tuple(sorted((int(first), int(second)))) for first, second in (distance_bound_exempt_pairs or set())
        )
        invalid_conformers: list[int] = []
        for conformer_id in conformer_ids:
            valid = True
            if conformer_preprocessor is not None:
                valid = conformer_preprocessor(working, conformer_id)
            if valid and coord_map and exact_constraints_before_clustering:
                valid = _validate_conformer_against_map(
                    working,
                    conformer_id,
                    coord_map,
                    exempt_pairs,
                )
            if not valid:
                invalid_conformers.append(conformer_id)
        for conformer_id in invalid_conformers:
            working.RemoveConformer(conformer_id)
        if working.GetNumConformers() == 0:
            raise RuntimeError("coordinate constraints produced invalid local geometry")

    heavy = Chem.RemoveHs(working)
    heavy_ids = [conformer.GetId() for conformer in heavy.GetConformers()]
    if len(heavy_ids) == 1:
        return heavy, heavy_ids
    coords = torch.stack(
        [torch.tensor(heavy.GetConformer(cid).GetPositions(), dtype=torch.float32) for cid in heavy_ids]
    ).to(target_device)
    if coord_map and exact_constraints_before_clustering:
        condensed = _condensed_direct_rmsd(coords, chunk_size=rmsd_chunk_size)
    else:
        condensed = _condensed_kabsch_rmsd(coords, chunk_size=rmsd_chunk_size)
    clusters = Butina.ClusterData(condensed, len(heavy_ids), float(rmsd_threshold), isDistData=True)
    representatives = [heavy_ids[int(cluster[0])] for cluster in clusters]
    return heavy, representatives
