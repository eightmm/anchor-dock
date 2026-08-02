from __future__ import annotations

from pathlib import Path

import pytest
import torch
from rdkit import Chem

from anchor_dock.core.output import write_ranked_poses


def _write(mol: Chem.Mol, coords: torch.Tensor, scores: torch.Tensor, path: Path, **kwargs):
    return write_ranked_poses(
        mol,
        coords,
        scores,
        str(path),
        scorer_name="test",
        score_units="arbitrary",
        score_semantics="test_pose_ranking",
        scorer_fingerprint="sha256:test",
        **kwargs,
    )


def test_output_rejects_atom_count_mismatch(tmp_path: Path) -> None:
    mol = Chem.MolFromSmiles("CC")
    with pytest.raises(ValueError, match="atoms"):
        _write(mol, torch.zeros(1, 1, 3), torch.zeros(1), tmp_path / "bad.sdf")


@pytest.mark.parametrize("field", ["coords", "scores", "search", "initial"])
def test_output_rejects_nonfinite_values(field: str, tmp_path: Path) -> None:
    mol = Chem.MolFromSmiles("CC")
    coords = torch.zeros(1, 2, 3)
    scores = torch.zeros(1)
    kwargs = {}
    if field == "coords":
        coords[0, 0, 0] = torch.nan
    elif field == "scores":
        scores[0] = torch.inf
    elif field == "search":
        kwargs["search_energies"] = torch.tensor([torch.nan])
    else:
        kwargs["initial_scores"] = torch.tensor([torch.inf])
    with pytest.raises(ValueError, match="finite"):
        _write(mol, coords, scores, tmp_path / f"{field}.sdf", **kwargs)


def test_output_rejects_reserved_metadata_overwrite(tmp_path: Path) -> None:
    mol = Chem.MolFromSmiles("CC")
    with pytest.raises(ValueError, match="reserved"):
        _write(
            mol,
            torch.zeros(1, 2, 3),
            torch.zeros(1),
            tmp_path / "reserved.sdf",
            molecule_metadata={"Score": "forged"},
        )


def test_equal_scores_preserve_input_pose_order(tmp_path: Path) -> None:
    mol = Chem.MolFromSmiles("C")
    _write(
        mol,
        torch.zeros(3, 1, 3),
        torch.zeros(3),
        tmp_path / "ties.sdf",
        pose_ids=["first", "second", "third"],
    )
    poses = [pose for pose in Chem.SDMolSupplier(str(tmp_path / "ties.sdf")) if pose]
    assert [pose.GetProp("AnchorDock_Pose_ID") for pose in poses] == ["first", "second", "third"]
