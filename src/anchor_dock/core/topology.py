"""Molecular graph utilities shared by masks and differentiable kinematics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from rdkit import Chem

_ROTATABLE_BOND_PATTERN = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")


@dataclass(frozen=True)
class RigidTopology:
    """Rigid components connected by non-ring rotatable bonds."""

    frames: tuple[tuple[int, ...], ...]
    atom_to_frame: tuple[int, ...]
    rotatable_bonds: tuple[tuple[int, int], ...]
    frame_edges: tuple[tuple[int, int, int, int], ...]


def find_rotatable_bonds(mol: Chem.Mol) -> list[tuple[int, int]]:
    """Return deterministic, unique heavy-atom rotatable bonds."""
    if _ROTATABLE_BOND_PATTERN is None:
        return []
    bonds = {tuple(sorted(match)) for match in mol.GetSubstructMatches(_ROTATABLE_BOND_PATTERN)}
    return sorted(bonds)


def build_rigid_topology(mol: Chem.Mol) -> RigidTopology:
    """Split ``mol`` into rigid frames by removing rotatable bonds."""
    num_atoms = mol.GetNumAtoms()
    rotatable = find_rotatable_bonds(mol)
    cut_edges = {frozenset(pair) for pair in rotatable}

    adjacency: list[list[int]] = [[] for _ in range(num_atoms)]
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if frozenset((begin, end)) in cut_edges:
            continue
        adjacency[begin].append(end)
        adjacency[end].append(begin)

    frames: list[tuple[int, ...]] = []
    atom_to_frame = [-1] * num_atoms
    visited: set[int] = set()
    for start in range(num_atoms):
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        atoms: list[int] = []
        while queue:
            atom_idx = queue.popleft()
            atoms.append(atom_idx)
            for neighbor in adjacency[atom_idx]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        frame_idx = len(frames)
        frame = tuple(sorted(atoms))
        frames.append(frame)
        for atom_idx in frame:
            atom_to_frame[atom_idx] = frame_idx

    frame_edges: list[tuple[int, int, int, int]] = []
    for begin, end in rotatable:
        frame_edges.append((atom_to_frame[begin], atom_to_frame[end], begin, end))

    return RigidTopology(
        frames=tuple(frames),
        atom_to_frame=tuple(atom_to_frame),
        rotatable_bonds=tuple(rotatable),
        frame_edges=tuple(frame_edges),
    )


def component_side(
    num_frames: int,
    frame_edges: tuple[tuple[int, int, int, int], ...],
    blocked_edge: int,
    start_frame: int,
) -> set[int]:
    """Return the frame side reachable without crossing one selected edge."""
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(num_frames)]
    for edge_idx, (left, right, _, _) in enumerate(frame_edges):
        adjacency[left].append((right, edge_idx))
        adjacency[right].append((left, edge_idx))

    side: set[int] = set()
    queue = deque([start_frame])
    while queue:
        frame = queue.popleft()
        if frame in side:
            continue
        side.add(frame)
        for neighbor, edge_idx in adjacency[frame]:
            if edge_idx != blocked_edge and neighbor not in side:
                queue.append(neighbor)
    return side
