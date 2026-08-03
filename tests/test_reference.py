from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rdkit import Chem

from anchor_dock import dock_reference
from anchor_dock.reference.mcs import select_mcs_mappings


def test_mcs_timeout_discards_partial_result(monkeypatch: pytest.MonkeyPatch) -> None:
    from anchor_dock.reference import mcs

    monkeypatch.setattr(
        mcs.rdFMCS,
        "FindMCS",
        lambda *args, **kwargs: SimpleNamespace(canceled=True, smartsString="[#6]"),
    )
    with pytest.raises(TimeoutError, match="partial result discarded"):
        mcs.find_simple_mcs_mappings(Chem.MolFromSmiles("CCC"), Chem.MolFromSmiles("CCC"))


def test_explicit_single_does_not_run_cross_search(monkeypatch: pytest.MonkeyPatch) -> None:
    from anchor_dock.reference import mcs

    monkeypatch.setattr(
        mcs,
        "find_cross_mcs_mappings",
        lambda *args, **kwargs: pytest.fail("single mode must not run cross MCS"),
    )
    selection = mcs.select_mcs_mappings(
        Chem.MolFromSmiles("CCC"),
        Chem.MolFromSmiles("CCCC"),
        mode="single",
    )
    assert selection.mode == "single"
    assert selection.cross_size == 0


def test_auto_mcs_considers_symmetry_and_is_deterministic() -> None:
    reference = Chem.MolFromSmiles("c1ccccc1CCc1ccccc1")
    query = Chem.MolFromSmiles("c1ccccc1")
    first = select_mcs_mappings(reference, query, mode="auto")
    second = select_mcs_mappings(reference, query, mode="auto")
    assert first == second
    assert first.mode == "multi"
    assert len(first.mappings) == 24
    assert first.candidate_complete is False
    assert first.max_size_proven is False
    assert "bounded cross-fragment candidate search" in first.reason


def test_mcs_enumerates_symmetry_correspondences_with_a_deterministic_cap() -> None:
    reference = Chem.MolFromSmiles("Cc1ccccc1")
    query = Chem.MolFromSmiles("Oc1ccccc1")
    all_mappings = select_mcs_mappings(reference, query, mode="multi", max_mappings=64)
    assert len(all_mappings.mappings) == 12
    first = select_mcs_mappings(reference, query, mode="multi", max_mappings=5)
    second = select_mcs_mappings(reference, query, mode="multi", max_mappings=5)
    assert first == second
    assert len(first.mappings) == 5
    assert len(set(first.mappings)) == 5
    assert first.candidate_complete is False
    assert first.max_size_proven is True
    assert "capped at 5" in first.reason
    complete = select_mcs_mappings(reference, query, mode="multi", max_mappings=12)
    assert complete.candidate_complete is True

    two_positions = select_mcs_mappings(
        Chem.MolFromSmiles("c1ccccc1CCc2ccccc2"),
        Chem.MolFromSmiles("c1ccccc1"),
        mode="multi",
        max_mappings=2,
    )
    assert (
        len({tuple(sorted(reference_idx for reference_idx, _ in mapping)) for mapping in two_positions.mappings}) == 2
    )

    repeated_reference = Chem.MolFromSmiles("c1ccccc1[Si](c2ccccc2)(c3ccccc3)c4ccccc4")
    repeated_query = Chem.MolFromSmiles("c1ccccc1[P](c2ccccc2)(c3ccccc3)c4ccccc4")
    both_repeated = select_mcs_mappings(
        repeated_reference,
        repeated_query,
        mode="multi",
        max_mappings=2,
    )
    assert len({tuple(sorted(ref for ref, _ in mapping)) for mapping in both_repeated.mappings}) == 2
    assert len({tuple(sorted(query for _, query in mapping)) for mapping in both_repeated.mappings}) == 2


def test_cross_mcs_combines_disjoint_rings_and_auto_selects_it() -> None:
    reference = Chem.MolFromSmiles("c1ccccc1CCc2ccccc2")
    query = Chem.MolFromSmiles("c1ccccc1OCCOc2ccccc2")
    first = select_mcs_mappings(reference, query, mode="cross", max_fragments=2)
    second = select_mcs_mappings(reference, query, mode="cross", max_fragments=2)
    automatic = select_mcs_mappings(reference, query, mode="auto", max_fragments=2)
    assert first == second
    assert max(map(len, first.mappings)) == 12
    assert all(len({ref for ref, _ in mapping}) == len(mapping) for mapping in first.mappings)
    assert all(len({query for _, query in mapping}) == len(mapping) for mapping in first.mappings)
    assert automatic.mode == "cross"
    assert automatic.simple_size == 6
    assert automatic.cross_size == 12

    bounded = select_mcs_mappings(
        reference,
        query,
        mode="cross",
        max_fragments=2,
        max_mappings=1,
    )
    assert len(bounded.mappings) == 1
    assert len(bounded.mappings[0]) == 12


def test_cross_mcs_recovers_better_bounded_alternative_fragment_packing() -> None:
    reference = Chem.MolFromSmiles("[SiH2](NCCCCCCCCCC)(OCCCCCCCC)")
    query = Chem.MolFromSmiles("[PH](OCCCCCCCCCC)(NCCCCCCCC)")

    first = select_mcs_mappings(
        reference,
        query,
        mode="auto",
        min_fragment_size=9,
        max_fragments=2,
        max_mappings=64,
    )
    second = select_mcs_mappings(
        reference,
        query,
        mode="auto",
        min_fragment_size=9,
        max_fragments=2,
        max_mappings=64,
    )

    assert first == second
    assert first.mode == "cross"
    assert first.simple_size == 10
    assert first.cross_size == 18
    assert len(first.mappings[0]) == 18
    assert first.candidate_complete is False
    assert first.max_size_proven is False
    assert "global cross-fragment maximum not proven" in first.reason


def test_alternative_fragment_cut_materialization_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    from anchor_dock.reference import mcs

    reference = Chem.MolFromSmiles("c1ccccc1" + "[SiH2]" * 40 + "[SiH3]")
    query = Chem.MolFromSmiles("c1ccccc1" + "[PH]" * 40 + "[PH2]")
    assert reference is not None and query is not None
    original_rw_mol = mcs.Chem.RWMol
    materialized = 0

    def counted_rw_mol(*args, **kwargs):
        nonlocal materialized
        materialized += 1
        return original_rw_mol(*args, **kwargs)

    monkeypatch.setattr(mcs.Chem, "RWMol", counted_rw_mol)
    components = mcs._single_cut_components(reference, query, min_atoms=5)

    assert materialized <= mcs._MAX_ALTERNATIVE_CUTS
    assert len(components) <= mcs._MAX_ALTERNATIVE_COMPONENTS_PER_MOLECULE


def test_alternative_fragment_articulation_scan_does_not_use_python_recursion() -> None:
    from anchor_dock.reference import mcs

    long_chain = Chem.MolFromSmiles("C" * 1100)
    other = Chem.MolFromSmiles("C")
    assert long_chain is not None and other is not None
    assert mcs._single_cut_components(long_chain, other, min_atoms=5) == []


def test_alternative_fragment_packing_discards_node_budget_partial_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anchor_dock.reference import mcs

    monkeypatch.setattr(mcs, "_MAX_ALTERNATIVE_PACKING_NODES", 1)
    candidates = [[(index, index)] for index in range(4)]
    with pytest.raises(RuntimeError, match="partial packing discarded"):
        mcs._best_disjoint_fragment_mapping(candidates, max_fragments=3)


def test_mcs_rejects_candidate_limits_above_the_internal_bound() -> None:
    with pytest.raises(ValueError, match="max_mappings must be <= 4096"):
        select_mcs_mappings(
            Chem.MolFromSmiles("CCC"),
            Chem.MolFromSmiles("CCC"),
            max_mappings=4097,
        )


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
        torsion_penalty=False,
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
    assert poses[0].GetProp("AnchorDock_Version") == "0.5.0"
    assert poses[0].GetProp("AnchorDock_Score_Semantics") == "anchor-conditioned_pose_ranking"
    assert poses[0].HasProp("AnchorDock_Scorer_Fingerprint")
    assert poses[0].HasProp("AnchorDock_Receptor_Structure_Fingerprint")
    assert poses[0].HasProp("AnchorDock_Receptor_Source_Fingerprint")
    assert poses[0].HasProp("AnchorDock_Intramolecular_Reference")
    assert poses[0].GetProp("AnchorDock_MCS_Reference_Index_Space") == ("reference_heavy_atom_after_remove_hs")
    mapping_position = poses[0].GetProp("AnchorDock_MCS_Position")
    assert json.loads(poses[0].GetProp("AnchorDock_MCS_Mapping")) == result["mcs_mappings_selected"][mapping_position]
    assert result["reference_structure_fingerprint"].startswith("sha256:")
    assert result["receptor_structure_fingerprint"].startswith("sha256:")
    assert result["receptor_source_fingerprint"].startswith("sha256:")
    assert result["torsion_penalty_applied"] is False
    assert result["score_rotatable_bonds"] == 0
    assert result["input_ligand_rotatable_bonds"] > 0
    assert result["mcs_max_size_proven"] is True
    assert poses[0].HasProp("AnchorDock_MCS_Candidate_Complete")
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

    reference = Chem.RemoveHs(Chem.SDMolSupplier(str(reference_sdf), removeHs=False)[0])
    query = Chem.MolFromSmiles("CCOc1ccccc1CC")
    mapping = select_mcs_mappings(reference, query, mode="single").mappings[0]
    reference_conf = reference.GetConformer()
    poses = [mol for mol in Chem.SDMolSupplier(result["output_file"], removeHs=False) if mol is not None]
    assert poses
    for pose in poses:
        conformer = pose.GetConformer()
        for reference_idx, query_idx in mapping:
            reference_xyz = reference_conf.GetAtomPosition(reference_idx)
            query_xyz = conformer.GetAtomPosition(query_idx)
            assert query_xyz.Distance(reference_xyz) <= 1e-4


def test_reference_preserves_original_mapping_index_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    cys_pdb: Path,
    reference_sdf: Path,
    tmp_path: Path,
) -> None:
    from anchor_dock.reference import pipeline
    from anchor_dock.reference.mcs import MCSSelection

    reference = Chem.RemoveHs(Chem.SDMolSupplier(str(reference_sdf), removeHs=False)[0])
    query = Chem.MolFromSmiles("CCOc1ccccc1C")
    mapping = select_mcs_mappings(reference, query, mode="single").mappings[0]
    monkeypatch.setattr(
        pipeline,
        "select_mcs_mappings",
        lambda *args, **kwargs: MCSSelection("multi", (mapping, mapping), "test", len(mapping), 0),
    )
    original_generate = pipeline.generate_conformers_and_cluster
    calls = 0

    def fail_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic mapping failure")
        return original_generate(*args, **kwargs)

    monkeypatch.setattr(pipeline, "generate_conformers_and_cluster", fail_first)
    result = dock_reference(
        cys_pdb,
        reference_sdf,
        query,
        tmp_path / "mapping-provenance",
        num_confs=2,
        relax=False,
        device="cpu",
        verbose=False,
    )
    assert result["mcs_positions_attempted"] == 2
    assert result["mcs_positions_selected"] == [2]
    assert result["best_position"] == 2
    assert result["failed_mappings"][0]["selection_index"] == 1
    pose = next(mol for mol in Chem.SDMolSupplier(result["output_file"]) if mol is not None)
    assert pose.GetProp("AnchorDock_MCS_Position") == "2"


def test_reference_all_mapping_failures_preserve_structured_details(
    monkeypatch: pytest.MonkeyPatch,
    cys_pdb: Path,
    reference_sdf: Path,
    tmp_path: Path,
) -> None:
    from anchor_dock.reference import pipeline

    monkeypatch.setattr(
        pipeline,
        "generate_conformers_and_cluster",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )
    with pytest.raises(RuntimeError, match='"mapping".*"reason":"synthetic failure"'):
        dock_reference(
            cys_pdb,
            reference_sdf,
            "CCOc1ccccc1C",
            tmp_path / "all-failed",
            num_confs=2,
            mcs_mode="single",
            relax=False,
            device="cpu",
            verbose=False,
        )
