"""Bounded deterministic joint hypotheses for multi-interaction docking."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .selectors import LigandAnchorMatch, MatchLimitExceededError


@dataclass(frozen=True)
class JointLigandHypothesis:
    """One ordered ligand-anchor assignment across all interaction items."""

    hypothesis_index: int
    anchors: tuple[LigandAnchorMatch, ...]

    @property
    def ligand_atom_indices(self) -> tuple[int, ...]:
        return tuple(anchor.ligand_atom_index for anchor in self.anchors)

    @property
    def ligand_match_indices(self) -> tuple[int, ...]:
        return tuple(anchor.match_index for anchor in self.anchors)


def enumerate_joint_hypotheses(
    anchor_groups: Sequence[Sequence[LigandAnchorMatch]],
    *,
    max_joint_matches: int,
) -> list[JointLigandHypothesis]:
    """Return the ordered Cartesian product without silent truncation."""
    if not isinstance(max_joint_matches, int) or isinstance(max_joint_matches, bool):
        raise ValueError("max_joint_matches must be a positive integer")
    if max_joint_matches <= 0:
        raise ValueError("max_joint_matches must be a positive integer")
    if not anchor_groups or any(not group for group in anchor_groups):
        raise ValueError("anchor_groups must contain one non-empty group per interaction")
    joint_count = math.prod(len(group) for group in anchor_groups)
    if joint_count > max_joint_matches:
        counts = " x ".join(str(len(group)) for group in anchor_groups)
        raise MatchLimitExceededError(
            f"interaction selectors produce {joint_count} joint hypotheses ({counts}), "
            f"exceeding max_joint_matches={max_joint_matches}; refine the SMARTS patterns "
            "or raise the explicit bound"
        )
    return [
        JointLigandHypothesis(index, tuple(anchors)) for index, anchors in enumerate(itertools.product(*anchor_groups))
    ]


def pairwise_shell_feasible(
    ligand_coords: torch.Tensor,
    ligand_atom_indices: Sequence[int],
    receptor_coords: torch.Tensor,
    target_distances: torch.Tensor,
    distance_tolerances: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> bool:
    """Conservatively reject rigid conformers impossible for any restraint pair.

    Each ligand atom must lie in a spherical shell around its receptor atom.
    For every pair, the ligand internal distance must fall within the minimum
    and maximum possible separation of points drawn from the two shells.
    Passing this necessary check is not a global feasibility claim.
    """
    if ligand_coords.ndim != 2 or ligand_coords.shape[-1] != 3:
        raise ValueError("ligand_coords must have shape [N,3]")
    if receptor_coords.ndim != 2 or receptor_coords.shape[-1] != 3:
        raise ValueError("receptor_coords must have shape [K,3]")
    count = len(ligand_atom_indices)
    if count == 0 or receptor_coords.shape[0] != count:
        raise ValueError("one ligand and receptor atom is required per interaction")
    if target_distances.shape != (count,) or distance_tolerances.shape != (count,):
        raise ValueError("target distances and tolerances must have shape [K]")
    if not torch.isfinite(ligand_coords).all() or not torch.isfinite(receptor_coords).all():
        raise ValueError("interaction coordinates must be finite")
    if not torch.isfinite(target_distances).all() or not torch.isfinite(distance_tolerances).all():
        raise ValueError("interaction distance windows must be finite")
    tensors = (ligand_coords, receptor_coords, target_distances, distance_tolerances)
    if any(not torch.is_floating_point(value) for value in tensors):
        raise ValueError("interaction geometry must use floating-point tensors")
    if epsilon < 0 or not math.isfinite(float(epsilon)):
        raise ValueError("epsilon must be a non-negative finite number")

    # A fixed absolute epsilon can reject a mathematically feasible shell boundary
    # after float32 rounding. Scale the guard by the least precise input dtype;
    # this only makes the necessary preflight check more conservative.
    machine_epsilon = max(torch.finfo(value.dtype).eps for value in tensors)

    lowers = target_distances - distance_tolerances
    uppers = target_distances + distance_tolerances
    if (lowers <= 0).any() or (distance_tolerances <= 0).any():
        raise ValueError("interaction distance windows must be positive")

    for left in range(count):
        left_atom = ligand_atom_indices[left]
        if not isinstance(left_atom, int) or isinstance(left_atom, bool):
            raise TypeError("ligand_atom_indices must contain integers")
        if not 0 <= left_atom < ligand_coords.shape[0]:
            raise ValueError("ligand_atom_indices contains an out-of-range value")
        for right in range(left + 1, count):
            right_atom = ligand_atom_indices[right]
            if not isinstance(right_atom, int) or isinstance(right_atom, bool):
                raise TypeError("ligand_atom_indices must contain integers")
            if not 0 <= right_atom < ligand_coords.shape[0]:
                raise ValueError("ligand_atom_indices contains an out-of-range value")

            ligand_separation = torch.linalg.vector_norm(ligand_coords[left_atom] - ligand_coords[right_atom])
            receptor_separation = torch.linalg.vector_norm(receptor_coords[left] - receptor_coords[right])
            minimum = torch.stack(
                (
                    receptor_separation - uppers[left] - uppers[right],
                    lowers[left] - receptor_separation - uppers[right],
                    lowers[right] - receptor_separation - uppers[left],
                    receptor_separation.new_zeros(()),
                )
            ).max()
            maximum = receptor_separation + uppers[left] + uppers[right]
            magnitude = max(
                1.0,
                abs(float(ligand_separation.item())),
                abs(float(minimum.item())),
                abs(float(maximum.item())),
            )
            comparison_epsilon = max(float(epsilon), 16.0 * machine_epsilon * magnitude)
            if ligand_separation < minimum - comparison_epsilon or ligand_separation > maximum + comparison_epsilon:
                return False
    return True
