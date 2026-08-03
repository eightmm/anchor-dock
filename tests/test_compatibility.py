from __future__ import annotations

import importlib

import pytest


def test_public_api_contains_canonical_and_one_release_aliases() -> None:
    package = importlib.import_module("anchor_dock")
    canonical = {
        "DockingEngine",
        "DockingJob",
        "LigandRecord",
        "dock_batch",
        "dock_covalent",
        "dock_interaction",
        "dock_reference",
    }
    compatibility = {
        "dock_covalent_batch",
        "dock_reference_batch",
        "run_batch_docking",
        "run_covalent_pipeline",
        "run_reference_pipeline",
    }
    assert canonical | compatibility <= set(package.__all__)


def test_removed_free_api_is_absent() -> None:
    package = importlib.import_module("anchor_dock")
    core = importlib.import_module("anchor_dock.core")
    assert not hasattr(package, "dock_free")
    assert not hasattr(package.DockingJob, "free")
    assert not hasattr(package.DockingEngine, "optimize_free")
    assert not hasattr(core, "FreePoseModel")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("anchor_dock.free")


def test_clear_all_caches_includes_interaction_cache() -> None:
    package = importlib.import_module("anchor_dock")
    pipeline = importlib.import_module("anchor_dock.interaction.pipeline")
    pipeline._INTERACTION_CONTEXT_CACHE[("f", "r", "a", 1.0, True, "cpu", "v")] = object()
    package.clear_all_caches()
    assert not pipeline._INTERACTION_CONTEXT_CACHE


def test_reference_legacy_keywords_warn_once_and_translate(monkeypatch) -> None:
    captured = {}

    def fake(*args, **kwargs):
        captured.update({"args": args, "kwargs": kwargs})
        return {"mode": "reference"}

    monkeypatch.setattr("anchor_dock.reference.pipeline.dock_reference", fake)
    from anchor_dock import dock_reference

    with pytest.warns(FutureWarning) as records:
        result = dock_reference(
            "protein.pdb",
            query_ligand="CCO",
            ref_ligand="reference.sdf",
            mmff_optimize=False,
            freeze_mcs=True,
            weight_preset="vinardo",
        )
    assert len(records) == 1
    assert result == {"mode": "reference"}
    assert captured["kwargs"] == {
        "query_ligand": "CCO",
        "reference_ligand": "reference.sdf",
        "relax": False,
        "freeze_anchor": True,
        "scorer": "vinardo",
    }


def test_reference_subpackage_legacy_keywords_use_same_adapter(monkeypatch) -> None:
    captured = {}

    def fake(*args, **kwargs):
        captured.update({"args": args, "kwargs": kwargs})
        return {"mode": "reference"}

    monkeypatch.setattr("anchor_dock.reference.pipeline.dock_reference", fake)
    from anchor_dock.reference import dock_reference

    with pytest.warns(FutureWarning) as records:
        result = dock_reference(
            "protein.pdb",
            query_ligand="CCO",
            ref_ligand="reference.sdf",
            weight_preset="vina",
        )
    assert len(records) == 1
    assert result == {"mode": "reference"}
    assert captured["kwargs"] == {
        "query_ligand": "CCO",
        "reference_ligand": "reference.sdf",
        "scorer": "vina",
    }


def test_reference_legacy_positional_tail_binds_to_old_signature(monkeypatch) -> None:
    captured = {}

    def fake(*args, **kwargs):
        captured.update({"args": args, "kwargs": kwargs})
        return {"mode": "reference"}

    monkeypatch.setattr("anchor_dock.reference.pipeline.dock_reference", fake)
    from anchor_dock import dock_reference

    with pytest.warns(FutureWarning) as records:
        dock_reference(
            "protein.pdb",
            "reference.sdf",
            "CCO",
            "output",
            25,
            0.75,
            "multi",
            6,
            2,
            False,
            True,
            "adamw",
            12,
            0.02,
            4,
            False,
            "vinardo",
            False,
            "cpu",
            False,
        )
    assert len(records) == 1
    assert captured["args"] == ("protein.pdb", "reference.sdf", "CCO", "output")
    assert captured["kwargs"] == {
        "num_confs": 25,
        "rmsd_threshold": 0.75,
        "mcs_mode": "multi",
        "min_fragment_size": 6,
        "max_fragments": 2,
        "relax": False,
        "optimize": True,
        "optimizer": "adamw",
        "opt_steps": 12,
        "opt_lr": 0.02,
        "opt_batch_size": 4,
        "freeze_anchor": False,
        "scorer": "vinardo",
        "torsion_penalty": False,
        "device": "cpu",
        "verbose": False,
    }


def test_legacy_keyword_conflicts_and_vina_lp_fail_before_dispatch(monkeypatch) -> None:
    def unexpected(*args, **kwargs):
        raise AssertionError((args, kwargs))

    monkeypatch.setattr("anchor_dock.reference.pipeline.dock_reference", unexpected)
    from anchor_dock import dock_reference

    with pytest.raises(ValueError, match="both"):
        dock_reference(
            protein_pdb="p.pdb",
            reference_ligand="new.sdf",
            ref_ligand="old.sdf",
            query_ligand="CCO",
        )
    with pytest.raises(ValueError, match="vina_lp"):
        dock_reference("p.pdb", "r.sdf", "CCO", scorer="vina_lp")


def test_covalent_omitted_optimize_preserves_false_with_warning(monkeypatch) -> None:
    captured = {}

    def fake(*args, **kwargs):
        captured.update(kwargs)
        return {"mode": "covalent"}

    monkeypatch.setattr("anchor_dock.covalent.pipeline.dock_covalent", fake)
    from anchor_dock import dock_covalent

    with pytest.warns(FutureWarning) as records:
        dock_covalent("p.pdb", "C=CC(=O)N")
    assert len(records) == 1
    assert captured["optimize"] is False


def test_covalent_subpackage_legacy_keywords_use_same_adapter(monkeypatch) -> None:
    captured = {}

    def fake(*args, **kwargs):
        captured.update({"args": args, "kwargs": kwargs})
        return {"mode": "covalent"}

    monkeypatch.setattr("anchor_dock.covalent.pipeline.dock_covalent", fake)
    from anchor_dock.covalent import dock_covalent

    with pytest.warns(FutureWarning) as records:
        result = dock_covalent(
            "protein.pdb",
            "C=CC(=O)N",
            weight_preset="vinardo",
        )
    assert len(records) == 1
    assert result == {"mode": "covalent"}
    assert captured["kwargs"] == {"scorer": "vinardo", "optimize": False}


def test_covalent_legacy_positional_tail_binds_to_old_signature(monkeypatch) -> None:
    captured = {}

    def fake(*args, **kwargs):
        captured.update({"args": args, "kwargs": kwargs})
        return {"mode": "covalent"}

    monkeypatch.setattr("anchor_dock.covalent.pipeline.dock_covalent", fake)
    from anchor_dock import dock_covalent

    with pytest.warns(FutureWarning) as records:
        dock_covalent(
            "protein.pdb",
            "C=CC(=O)N",
            "CYS145:A",
            "output",
            9.0,
            None,
            25,
            0.75,
            60,
            5,
            True,
            "lbfgs",
            10,
            0.1,
            2,
            "vinardo",
            False,
            False,
            1,
            "cpu",
            False,
            0,
            False,
        )
    assert len(records) == 1
    assert captured["args"] == ("protein.pdb", "C=CC(=O)N", "CYS145:A", "output")
    assert captured["kwargs"] == {
        "pocket_cutoff": 9.0,
        "num_confs": 25,
        "rmsd_threshold": 0.75,
        "rotation_scan_step": 60,
        "rotation_top_k": 5,
        "optimize": True,
        "optimizer": "lbfgs",
        "opt_steps": 10,
        "opt_lr": 0.1,
        "opt_batch_size": 2,
        "scorer": "vinardo",
        "torsion_penalty": False,
        "top_k": 1,
        "device": "cpu",
        "verbose": False,
        "warhead_index": 0,
        "strict_compatibility": False,
    }


def test_reference_batch_alias_routes_to_unified_batch(monkeypatch) -> None:
    captured = {}

    def fake(source, **kwargs):
        captured.update({"source": source, **kwargs})
        return [{"success": True}]

    monkeypatch.setattr("anchor_dock.batch.dock_batch", fake)
    from anchor_dock.reference import run_batch

    with pytest.warns(FutureWarning) as records:
        result = run_batch("p.pdb", "r.sdf", ["CCO"], verbose=False)
    assert len(records) == 1
    assert result == [{"success": True}]
    assert captured["mode"] == "reference"
    assert captured["reference_ligand"] == "r.sdf"


def test_removed_low_level_facade_is_not_recreated() -> None:
    reference = importlib.import_module("anchor_dock.reference")
    for name in ("LigandAligner", "find_mcs_with_positions", "load_pocket_bundle", "final_selection"):
        assert not hasattr(reference, name)


@pytest.mark.parametrize("removed", ["lig_align", "cov_vina"])
def test_retired_namespaces_are_gone(removed: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(removed)
