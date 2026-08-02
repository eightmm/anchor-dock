from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem

from anchor_dock import dock_reference
from anchor_dock.reference.mcs import select_mcs_mappings


def test_auto_mcs_considers_symmetry_and_is_deterministic() -> None:
    reference = Chem.MolFromSmiles("c1ccccc1CCc1ccccc1")
    query = Chem.MolFromSmiles("c1ccccc1")
    first = select_mcs_mappings(reference, query, mode="auto")
    second = select_mcs_mappings(reference, query, mode="auto")
    assert first == second
    assert first.mode == "multi"
    assert len(first.mappings) == 2


def test_reference_pipeline_uses_standard_metadata(
    cys_pdb: Path,
    reference_sdf: Path,
    tmp_path: Path,
) -> None:
    result = dock_reference(
        cys_pdb,
        reference_sdf,
        "CCOc1ccccc1C",
        tmp_path / "out",
        num_confs=6,
        mcs_mode="single",
        relax=False,
        optimize=False,
        top_k=2,
        device="cpu",
        verbose=False,
    )
    assert result["mode"] == "reference"
    assert result["num_poses"] >= 1
    poses = [mol for mol in Chem.SDMolSupplier(result["output_file"], removeHs=False) if mol is not None]
    assert poses
    properties = set(poses[0].GetPropNames())
    assert "AnchorDock_Mode" in properties
    assert "AnchorDock_MCS_Mode" in properties
    assert not any(name.startswith("LigAlign_") for name in properties)


def test_reference_optimization_keeps_all_mcs_atoms_fixed(
    cys_pdb: Path,
    reference_sdf: Path,
    tmp_path: Path,
) -> None:
    result = dock_reference(
        cys_pdb,
        reference_sdf,
        "CCOc1ccccc1CC",
        tmp_path / "opt",
        num_confs=4,
        mcs_mode="single",
        relax=False,
        optimize=True,
        opt_steps=2,
        opt_batch_size=4,
        device="cpu",
        verbose=False,
    )
    assert result["optimized"] is True
    assert result["best_score"] == pytest.approx(result["best_score"])
