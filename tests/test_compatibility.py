from __future__ import annotations

import importlib


def test_public_and_legacy_imports() -> None:
    anchor_dock = importlib.import_module("anchor_dock")
    assert callable(anchor_dock.dock_covalent)
    assert callable(anchor_dock.dock_covalent_batch)
    assert callable(anchor_dock.dock_reference)
    assert callable(anchor_dock.dock_reference_batch)

    scoring = importlib.import_module("lig_align.scoring.vina_scoring")
    kinematics = importlib.import_module("lig_align.alignment.kinematics")
    assert callable(scoring.vina_scoring)
    assert hasattr(kinematics, "BatchedLigandKinematics")
    assert hasattr(kinematics, "_build_kinematic_topology")
