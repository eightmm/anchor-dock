"""Batched forward kinematics for differentiable ligand torsion refinement."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from rdkit import Chem


def get_rotation_matrix(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation matrix for one axis and angle."""
    return get_batched_rotation_matrix(axis.reshape(1, 3), theta.reshape(1))[0]


def get_batched_rotation_matrix(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation matrices for ``axis=[B,3]`` and ``theta=[B]``."""
    axis = axis / torch.linalg.vector_norm(axis, dim=1, keepdim=True).clamp_min(1e-12)
    x, y, z = axis.unbind(dim=1)
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    one_minus = 1.0 - cos_t
    return torch.stack(
        [
            cos_t + x * x * one_minus,
            x * y * one_minus - z * sin_t,
            x * z * one_minus + y * sin_t,
            x * y * one_minus + z * sin_t,
            cos_t + y * y * one_minus,
            y * z * one_minus - x * sin_t,
            x * z * one_minus - y * sin_t,
            y * z * one_minus + x * sin_t,
            cos_t + z * z * one_minus,
        ],
        dim=1,
    ).reshape(-1, 3, 3)


def _descendants(tree: dict[int, list[tuple[int, int, int]]], frames: list[list[int]], frame: int) -> list[int]:
    result: list[int] = []
    for child, _, _ in tree[frame]:
        result.extend(frames[child])
        result.extend(_descendants(tree, frames, child))
    return result


def build_kinematic_topology(
    mol: Chem.Mol,
    anchor_indices: Iterable[int],
    freeze_anchor: bool = True,
) -> dict[str, Any]:
    """Split a molecule into rigid frames connected by rotatable bonds."""
    num_atoms = mol.GetNumAtoms()
    anchor_set = {int(idx) for idx in anchor_indices}
    if not anchor_set:
        anchor_set = {0}

    pattern = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    rotatable = [set(match) for match in mol.GetSubstructMatches(pattern)] if pattern else []
    if freeze_anchor:
        rotatable = [pair for pair in rotatable if not pair.issubset(anchor_set)]

    adjacency: dict[int, list[int]] = {idx: [] for idx in range(num_atoms)}
    rotatable_keys = {frozenset(pair) for pair in rotatable}
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if frozenset((u, v)) not in rotatable_keys:
            adjacency[u].append(v)
            adjacency[v].append(u)

    frames: list[list[int]] = []
    atom_to_frame: dict[int, int] = {}
    visited: set[int] = set()
    for start in range(num_atoms):
        if start in visited:
            continue
        frame: list[int] = []
        queue = deque([start])
        visited.add(start)
        while queue:
            atom_idx = queue.popleft()
            frame.append(atom_idx)
            atom_to_frame[atom_idx] = len(frames)
            for neighbor in adjacency[atom_idx]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        frames.append(frame)

    root = max(range(len(frames)), key=lambda idx: len(set(frames[idx]) & anchor_set))
    tree: dict[int, list[tuple[int, int, int]]] = {idx: [] for idx in range(len(frames))}
    queue = deque([root])
    visited_frames = {root}
    edges: list[tuple[int, int, int]] = []
    while queue:
        current = queue.popleft()
        for atom_idx in frames[current]:
            for neighbor_atom in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                neighbor_idx = neighbor_atom.GetIdx()
                neighbor_frame = atom_to_frame[neighbor_idx]
                if neighbor_frame == current or neighbor_frame in visited_frames:
                    continue
                tree[current].append((neighbor_frame, atom_idx, neighbor_idx))
                edges.append((atom_idx, neighbor_idx, neighbor_frame))
                visited_frames.add(neighbor_frame)
                queue.append(neighbor_frame)

    atoms_to_rotate = [frames[child] + _descendants(tree, frames, child) for _, _, child in edges]
    return {
        "num_atoms": num_atoms,
        "frames": frames,
        "tree": tree,
        "parent_atoms": [edge[0] for edge in edges],
        "child_atoms": [edge[1] for edge in edges],
        "child_frames": [edge[2] for edge in edges],
        "atoms_to_rotate": atoms_to_rotate,
        "num_torsions": len(edges),
    }


class LigandKinematics(nn.Module):
    """Single implementation for both ``[N,3]`` and ``[B,N,3]`` inputs."""

    def __init__(
        self,
        mol: Chem.Mol,
        ref_indices: Iterable[int],
        init_coords: torch.Tensor,
        device: torch.device | str,
        freeze_anchor: bool = True,
        freeze_mcs: bool | None = None,
    ) -> None:
        super().__init__()
        if freeze_mcs is not None:
            freeze_anchor = freeze_mcs
        self.device = torch.device(device)
        topology = build_kinematic_topology(mol, ref_indices, freeze_anchor)
        self.num_atoms = topology["num_atoms"]
        self.num_torsions = topology["num_torsions"]
        self.parent_atoms = topology["parent_atoms"]
        self.child_atoms = topology["child_atoms"]
        self.child_frames = topology["child_frames"]
        self.atoms_to_rotate = [
            torch.tensor(indices, dtype=torch.long, device=self.device)
            for indices in topology["atoms_to_rotate"]
        ]

        coords = init_coords.to(device=self.device, dtype=torch.float32)
        if coords.ndim == 2:
            self.is_batched = False
            coords = coords.unsqueeze(0)
        elif coords.ndim == 3:
            self.is_batched = True
        else:
            raise ValueError(f"init_coords must be [N,3] or [B,N,3], got {tuple(coords.shape)}")
        self.register_buffer("base_coords", coords.clone())
        self.thetas = nn.Parameter(torch.zeros(coords.shape[0], self.num_torsions, device=self.device))

    def forward(self) -> torch.Tensor:
        coords = self.base_coords.clone()
        for torsion_idx in range(self.num_torsions):
            parent = self.parent_atoms[torsion_idx]
            child = self.child_atoms[torsion_idx]
            rotated_atoms = self.atoms_to_rotate[torsion_idx]
            if rotated_atoms.numel() == 0:
                continue
            origin = coords[:, parent, :]
            axis = coords[:, child, :] - origin
            rotation = get_batched_rotation_matrix(axis, self.thetas[:, torsion_idx])
            vectors = coords[:, rotated_atoms, :] - origin[:, None, :]
            rotated = torch.matmul(vectors, rotation.transpose(1, 2)) + origin[:, None, :]
            next_coords = coords.clone()
            next_coords[:, rotated_atoms, :] = rotated
            coords = next_coords
        return coords if self.is_batched else coords[0]


class BatchedLigandKinematics(LigandKinematics):
    """Alias retained for reference-mode callers that name the batched form."""

    def __init__(self, mol, ref_indices, init_coords, device, freeze_mcs: bool = True):
        super().__init__(mol, ref_indices, init_coords, device, freeze_anchor=freeze_mcs)
