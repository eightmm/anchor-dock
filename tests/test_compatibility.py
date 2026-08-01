from __future__ import annotations

import importlib

import pytest


def test_public_api_surface() -> None:
    anchor_dock = importlib.import_module("anchor_dock")
    for name in ("dock_covalent", "dock_covalent_batch", "dock_reference", "dock_reference_batch"):
        assert callable(getattr(anchor_dock, name)), name


def test_reference_package_exports_its_implementation() -> None:
    reference = importlib.import_module("anchor_dock.reference")
    for name in ("run_pipeline", "run_batch", "find_mcs_with_positions", "auto_select_mcs_mapping",
                 "generate_conformers_and_cluster", "process_query_ligand", "load_pocket_bundle",
                 "final_selection", "relax_pose_with_fixed_core"):
        assert hasattr(reference, name), name
    assert reference.LigandAligner is not None


@pytest.mark.parametrize("removed", ["lig_align", "cov_vina"])
def test_retired_namespaces_are_gone(removed: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(removed)
