"""Scientific-invariant regression tests for the bundled 10gs example."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from rdkit import Chem

from anchor_dock import dock_reference
from anchor_dock.core.conformers import _has_invalid_bond_geometry
from anchor_dock.core.io import load_ligand
from anchor_dock.reference.mcs import select_mcs_mappings

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "10gs"
PROTEIN = EXAMPLES / "10gs_pocket.pdb"
REFERENCE = EXAMPLES / "10gs_ligand.sdf"

CASES = {
    "single_vina": {"query_ligand": "CC(=O)Nc1ccc(O)cc1", "mcs_mode": "single", "scorer": "vina"},
    "single_vinardo": {
        "query_ligand": "CC(=O)Nc1ccc(O)cc1",
        "mcs_mode": "single",
        "scorer": "vinardo",
    },
    "multi": {
        "query_ligand": "CC(C(=O)O)c1ccc(cc1)CC(C)C",
        "mcs_mode": "multi",
        "scorer": "vina",
    },
}


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_reference_output_is_deterministic_and_physically_valid(case_id: str, tmp_path: Path) -> None:
    options = CASES[case_id]
    common = {
        **options,
        "num_confs": 12,
        "rmsd_threshold": 1.0,
        "max_mappings": 8,
        "relax": False,
        "optimize": False,
        "top_k": 3,
        "random_seed": 2026,
        "device": "cpu",
        "verbose": False,
    }
    first = dock_reference(PROTEIN, REFERENCE, output_dir=tmp_path / "first", **common)
    second = dock_reference(PROTEIN, REFERENCE, output_dir=tmp_path / "second", **common)

    assert first["mcs_size"] >= 3
    assert first["mcs_positions_selected"]
    assert first["failed_mappings"] == []
    assert math.isfinite(first["best_score"])
    assert math.isfinite(first["best_search_energy"])
    assert first["best_score"] == pytest.approx(second["best_score"], abs=1e-7)
    assert first["best_search_energy"] == pytest.approx(second["best_search_energy"], abs=1e-7)
    assert first["mcs_positions_selected"] == second["mcs_positions_selected"]

    first_poses = [mol for mol in Chem.SDMolSupplier(first["output_file"], removeHs=False) if mol]
    second_poses = [mol for mol in Chem.SDMolSupplier(second["output_file"], removeHs=False) if mol]
    assert len(first_poses) == len(second_poses) == first["num_poses"]
    first_scores = [float(mol.GetProp("AnchorDock_Score")) for mol in first_poses]
    second_scores = [float(mol.GetProp("AnchorDock_Score")) for mol in second_poses]
    assert first_scores == pytest.approx(second_scores, abs=1e-7)
    assert first_scores == sorted(first_scores)
    assert [int(mol.GetProp("AnchorDock_Rank")) for mol in first_poses] == list(range(1, len(first_poses) + 1))

    reference = Chem.RemoveHs(Chem.SDMolSupplier(str(REFERENCE), removeHs=False)[0])
    query, _ = load_ligand(options["query_ligand"])
    selection = select_mcs_mappings(
        reference,
        query,
        mode=options["mcs_mode"],
        max_mappings=8,
    )
    reference_conformer = reference.GetConformer()
    for pose in first_poses:
        assert pose.GetProp("AnchorDock_Version") == "0.3.0"
        assert pose.GetProp("AnchorDock_Score_Semantics") == "anchor-conditioned_pose_ranking"
        assert pose.HasProp("AnchorDock_Scorer_Fingerprint")
        assert not pose.HasProp("Vina_Score")
        mapping_position = int(pose.GetProp("AnchorDock_MCS_Position"))
        mapping = selection.mappings[mapping_position - 1]
        conformer = pose.GetConformer()
        for reference_idx, query_idx in mapping:
            assert (
                conformer.GetAtomPosition(query_idx).Distance(reference_conformer.GetAtomPosition(reference_idx))
                <= 1e-4
            )
        assert not _has_invalid_bond_geometry(
            pose,
            conformer.GetPositions(),
            frozenset(),
        )
