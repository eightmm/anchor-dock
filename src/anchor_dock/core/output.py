"""Pose export helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import torch
from rdkit import Chem
from rdkit.Geometry import Point3D


def write_ranked_poses(
    mol: Chem.Mol,
    coords: torch.Tensor,
    scores: torch.Tensor,
    output_path: str,
    *,
    initial_scores: torch.Tensor | None = None,
    pose_ids: Sequence[int] | None = None,
    top_k: int | None = None,
    per_pose_metadata: Sequence[Mapping[str, object]] | None = None,
) -> torch.Tensor:
    """Sort poses by score and write them to one SDF."""
    if coords.ndim != 3 or scores.ndim != 1 or coords.shape[0] != scores.shape[0]:
        raise ValueError("coords and scores must have shapes [P,N,3] and [P]")
    if initial_scores is not None and initial_scores.shape != scores.shape:
        raise ValueError("initial_scores must match scores")
    order = torch.argsort(scores)
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        order = order[:top_k]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = Chem.SDWriter(output_path)
    try:
        for rank, tensor_idx in enumerate(order.tolist(), start=1):
            out = Chem.Mol(mol)
            out.RemoveAllConformers()
            conformer = Chem.Conformer(out.GetNumAtoms())
            for atom_idx, xyz in enumerate(coords[tensor_idx].detach().cpu().tolist()):
                conformer.SetAtomPosition(atom_idx, Point3D(*map(float, xyz)))
            out.AddConformer(conformer, assignId=True)
            pose_id = tensor_idx if pose_ids is None else pose_ids[tensor_idx]
            final = float(scores[tensor_idx].detach().cpu())
            out.SetProp("_Name", f"Pose_{pose_id}_Rank_{rank}")
            out.SetProp("Rank", str(rank))
            out.SetProp("Vina_Score", f"{final:.6f}")
            out.SetProp("Vina_Score_Final", f"{final:.6f}")
            if initial_scores is not None:
                initial = float(initial_scores[tensor_idx].detach().cpu())
                out.SetProp("Vina_Score_Initial", f"{initial:.6f}")
                out.SetProp("Vina_Score_Delta", f"{final - initial:.6f}")
            if per_pose_metadata is not None:
                for key, value in per_pose_metadata[tensor_idx].items():
                    out.SetProp(str(key), str(value))
            writer.write(out)
    finally:
        writer.close()
    return order


def final_selection(
    mol,
    representative_cids,
    aligned_coords,
    scores,
    initial_scores=None,
    top_k=None,
    output_path="output.sdf",
):
    """Compatibility wrapper used by the former CovVina API."""
    return write_ranked_poses(
        mol,
        aligned_coords,
        scores,
        output_path,
        initial_scores=initial_scores,
        pose_ids=representative_cids,
        top_k=top_k,
    )
