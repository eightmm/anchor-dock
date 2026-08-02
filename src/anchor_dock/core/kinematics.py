"""Differentiable rigid-frame and torsion kinematics."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from rdkit import Chem

from .topology import build_rigid_topology, component_side


def get_batched_rotation_matrix(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rodrigues matrices for ``axis=[B,3]`` and ``theta=[B]``.

    Degenerate axes produce an identity rotation rather than a scaled cosine
    matrix. This matters for coincident atom coordinates during failed embeds.
    """
    if axis.ndim != 2 or axis.shape[-1] != 3 or theta.ndim != 1 or axis.shape[0] != theta.shape[0]:
        raise ValueError("axis and theta must have shapes [B,3] and [B]")
    norm = torch.linalg.vector_norm(axis, dim=1, keepdim=True)
    valid = norm.squeeze(1) > 1e-12
    fallback = torch.zeros_like(axis)
    fallback[:, 0] = 1.0
    unit = torch.where(valid[:, None], axis / norm.clamp_min(1e-12), fallback)
    angle = torch.where(valid, theta, torch.zeros_like(theta))

    x, y, z = unit.unbind(dim=1)
    sin_t = torch.sin(angle)
    cos_t = torch.cos(angle)
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


def get_rotation_matrix(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation matrix for one axis and angle."""
    return get_batched_rotation_matrix(axis.reshape(1, 3), theta.reshape(1))[0]


def _depth_order(
    num_frames: int,
    directed_edges: list[tuple[int, int, int, int, list[int]]],
) -> list[tuple[int, int, int, int, list[int]]]:
    """Order directed frame edges from fixed roots toward terminal branches."""
    outgoing: list[list[int]] = [[] for _ in range(num_frames)]
    indegree = [0] * num_frames
    for edge_idx, (parent, child, _, _, _) in enumerate(directed_edges):
        outgoing[parent].append(edge_idx)
        indegree[child] += 1

    roots = deque(idx for idx, degree in enumerate(indegree) if degree == 0)
    depth = [0] * num_frames
    while roots:
        frame = roots.popleft()
        for edge_idx in outgoing[frame]:
            child = directed_edges[edge_idx][1]
            depth[child] = max(depth[child], depth[frame] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                roots.append(child)
    return sorted(directed_edges, key=lambda item: (depth[item[1]], item[1], item[2], item[3]))


def build_kinematic_topology(
    mol: Chem.Mol,
    anchor_indices: Iterable[int] = (),
    freeze_anchor: bool = True,
) -> dict[str, Any]:
    """Build torsion actions that never move requested anchor atoms.

    For every rotatable bond the two graph sides are inspected. If anchors lie
    on both sides, that torsion is disabled. If anchors lie on only one side,
    the opposite side is rotated. This is stricter than choosing one anchor-rich
    root frame and correctly handles distributed MCS anchors.
    """
    topology = build_rigid_topology(mol)
    num_frames = len(topology.frames)
    valid_anchors = {int(idx) for idx in anchor_indices if 0 <= int(idx) < mol.GetNumAtoms()}
    if not valid_anchors and mol.GetNumAtoms():
        valid_anchors = {0}
    anchor_frames = {topology.atom_to_frame[idx] for idx in valid_anchors}
    default_root = min(anchor_frames) if anchor_frames else 0
    all_frames = set(range(num_frames))

    directed: list[tuple[int, int, int, int, list[int]]] = []
    disabled: list[tuple[int, int]] = []
    for edge_idx, (left_frame, right_frame, left_atom, right_atom) in enumerate(topology.frame_edges):
        left_side = component_side(num_frames, topology.frame_edges, edge_idx, left_frame)
        right_side = all_frames - left_side
        left_has_anchor = bool(anchor_frames & left_side)
        right_has_anchor = bool(anchor_frames & right_side)

        if freeze_anchor and left_has_anchor and right_has_anchor:
            disabled.append(tuple(sorted((left_atom, right_atom))))
            continue
        if left_has_anchor and not right_has_anchor:
            parent, child = left_frame, right_frame
            parent_atom, child_atom = left_atom, right_atom
            rotated_frames = right_side
        elif right_has_anchor and not left_has_anchor:
            parent, child = right_frame, left_frame
            parent_atom, child_atom = right_atom, left_atom
            rotated_frames = left_side
        elif default_root in left_side:
            parent, child = left_frame, right_frame
            parent_atom, child_atom = left_atom, right_atom
            rotated_frames = right_side
        else:
            parent, child = right_frame, left_frame
            parent_atom, child_atom = right_atom, left_atom
            rotated_frames = left_side

        atoms_to_rotate = sorted(atom for frame in rotated_frames for atom in topology.frames[frame])
        if freeze_anchor and valid_anchors.intersection(atoms_to_rotate):
            raise RuntimeError("internal topology error: a frozen anchor was assigned to a rotating side")
        directed.append((parent, child, parent_atom, child_atom, atoms_to_rotate))

    directed = _depth_order(num_frames, directed)
    return {
        "num_atoms": mol.GetNumAtoms(),
        "frames": [list(frame) for frame in topology.frames],
        "atom_to_frame": list(topology.atom_to_frame),
        "parent_frames": [item[0] for item in directed],
        "child_frames": [item[1] for item in directed],
        "parent_atoms": [item[2] for item in directed],
        "child_atoms": [item[3] for item in directed],
        "atoms_to_rotate": [item[4] for item in directed],
        "disabled_torsions": disabled,
        "num_torsions": len(directed),
    }


class LigandKinematics(nn.Module):
    """Forward kinematics for one pose or a batch of poses of one topology."""

    def __init__(
        self,
        mol: Chem.Mol,
        anchor_indices: Iterable[int],
        init_coords: torch.Tensor,
        device: torch.device | str,
        *,
        freeze_anchor: bool = True,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        topology = build_kinematic_topology(mol, anchor_indices, freeze_anchor=freeze_anchor)
        self.num_atoms = topology["num_atoms"]
        self.num_torsions = topology["num_torsions"]
        self.parent_atoms = topology["parent_atoms"]
        self.child_atoms = topology["child_atoms"]
        self.disabled_torsions = topology["disabled_torsions"]
        self.atoms_to_rotate = [
            torch.tensor(indices, dtype=torch.long, device=self.device) for indices in topology["atoms_to_rotate"]
        ]

        coords = init_coords.to(device=self.device, dtype=torch.float32)
        if coords.ndim == 2:
            self.is_batched = False
            coords = coords.unsqueeze(0)
        elif coords.ndim == 3:
            self.is_batched = True
        else:
            raise ValueError(f"init_coords must be [N,3] or [B,N,3], got {tuple(coords.shape)}")
        if coords.shape[1:] != (self.num_atoms, 3):
            raise ValueError(f"coordinate shape {tuple(coords.shape)} does not match {self.num_atoms} atoms")
        self.register_buffer("base_coords", coords.clone())
        self.thetas = nn.Parameter(torch.zeros(coords.shape[0], self.num_torsions, device=self.device))

    def forward(self) -> torch.Tensor:
        coords = self.base_coords.clone()
        for torsion_idx in range(self.num_torsions):
            rotated_atoms = self.atoms_to_rotate[torsion_idx]
            if rotated_atoms.numel() == 0:
                continue
            parent = self.parent_atoms[torsion_idx]
            child = self.child_atoms[torsion_idx]
            origin = coords[:, parent, :]
            axis = coords[:, child, :] - origin
            rotation = get_batched_rotation_matrix(axis, self.thetas[:, torsion_idx])
            vectors = coords[:, rotated_atoms, :] - origin[:, None, :]
            rotated = torch.matmul(vectors, rotation.transpose(1, 2)) + origin[:, None, :]
            next_coords = coords.clone()
            next_coords[:, rotated_atoms, :] = rotated
            coords = next_coords
        return coords if self.is_batched else coords[0]
