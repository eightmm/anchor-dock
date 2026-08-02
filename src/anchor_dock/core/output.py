"""Deterministic pose export with one AnchorDock metadata schema."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import torch
from rdkit import Chem
from rdkit.Geometry import Point3D

from .._version import __version__

OUTPUT_SCHEMA_VERSION = "1"
_RESERVED_METADATA_KEYS = {
    "Rank",
    "Pose_ID",
    "Version",
    "Output_Schema",
    "Scorer",
    "Scorer_Fingerprint",
    "Score_Units",
    "Score_Semantics",
    "Score",
    "Search_Energy",
    "Initial_Score",
    "Score_Delta",
}


def write_ranked_poses(
    mol: Chem.Mol,
    coords: torch.Tensor,
    scores: torch.Tensor,
    output_path: str,
    *,
    scorer_name: str,
    score_units: str,
    score_semantics: str,
    scorer_fingerprint: str,
    search_energies: torch.Tensor | None = None,
    initial_scores: torch.Tensor | None = None,
    pose_ids: Sequence[int | str] | None = None,
    top_k: int | None = None,
    molecule_metadata: Mapping[str, object] | None = None,
    per_pose_metadata: Sequence[Mapping[str, object]] | None = None,
) -> torch.Tensor:
    """Sort poses by score and write one SDF using ``AnchorDock_*`` tags."""
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError("coords must have shape [P,N,3]")
    if coords.shape[1] != mol.GetNumAtoms():
        raise ValueError(f"coords contain {coords.shape[1]} atoms but molecule contains {mol.GetNumAtoms()}")
    if scores.ndim != 1 or scores.shape[0] != coords.shape[0]:
        raise ValueError("scores must have shape [P]")
    if not torch.isfinite(coords).all() or not torch.isfinite(scores).all():
        raise ValueError("coords and scores must contain only finite values")
    for name, values in (("search_energies", search_energies), ("initial_scores", initial_scores)):
        if values is not None and values.shape != scores.shape:
            raise ValueError(f"{name} must match scores")
        if values is not None and not torch.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values")
    if pose_ids is not None and len(pose_ids) != coords.shape[0]:
        raise ValueError("pose_ids must match pose count")
    if per_pose_metadata is not None and len(per_pose_metadata) != coords.shape[0]:
        raise ValueError("per_pose_metadata must match pose count")
    metadata_groups = [molecule_metadata or {}, *(per_pose_metadata or ())]
    for metadata in metadata_groups:
        collisions = _RESERVED_METADATA_KEYS.intersection(metadata)
        if collisions:
            raise ValueError(f"metadata cannot overwrite reserved keys: {sorted(collisions)}")
    if not scorer_name or not score_units or not score_semantics or not scorer_fingerprint:
        raise ValueError("scorer provenance fields must be non-empty")

    order = torch.argsort(scores, stable=True)
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        order = order[:top_k]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    writer = Chem.SDWriter(output_path)
    try:
        for rank, tensor_idx in enumerate(order.tolist(), start=1):
            output_mol = Chem.Mol(mol)
            output_mol.RemoveAllConformers()
            conformer = Chem.Conformer(output_mol.GetNumAtoms())
            for atom_idx, xyz in enumerate(coords[tensor_idx].detach().cpu().tolist()):
                conformer.SetAtomPosition(atom_idx, Point3D(*map(float, xyz)))
            output_mol.AddConformer(conformer, assignId=True)

            pose_id = tensor_idx if pose_ids is None else pose_ids[tensor_idx]
            final = float(scores[tensor_idx].detach().cpu())
            output_mol.SetProp("_Name", f"AnchorDock_{pose_id}_Rank_{rank}")
            output_mol.SetProp("AnchorDock_Rank", str(rank))
            output_mol.SetProp("AnchorDock_Pose_ID", str(pose_id))
            output_mol.SetProp("AnchorDock_Version", __version__)
            output_mol.SetProp("AnchorDock_Output_Schema", OUTPUT_SCHEMA_VERSION)
            output_mol.SetProp("AnchorDock_Scorer", scorer_name)
            output_mol.SetProp("AnchorDock_Scorer_Fingerprint", scorer_fingerprint)
            output_mol.SetProp("AnchorDock_Score_Units", score_units)
            output_mol.SetProp("AnchorDock_Score_Semantics", score_semantics)
            output_mol.SetProp("AnchorDock_Score", f"{final:.8f}")
            if search_energies is not None:
                search = float(search_energies[tensor_idx].detach().cpu())
                output_mol.SetProp("AnchorDock_Search_Energy", f"{search:.8f}")
            if initial_scores is not None:
                initial = float(initial_scores[tensor_idx].detach().cpu())
                output_mol.SetProp("AnchorDock_Initial_Score", f"{initial:.8f}")
                output_mol.SetProp("AnchorDock_Score_Delta", f"{final - initial:.8f}")
            if molecule_metadata is not None:
                for key, value in molecule_metadata.items():
                    output_mol.SetProp(f"AnchorDock_{key}", str(value))
            if per_pose_metadata is not None:
                for key, value in per_pose_metadata[tensor_idx].items():
                    output_mol.SetProp(f"AnchorDock_{key}", str(value))
            writer.write(output_mol)
    finally:
        writer.close()
    return order
