from __future__ import annotations

import math

import pytest
import torch
from rdkit import Chem

from anchor_dock.core.engine import DockingEngine
from anchor_dock.interaction import (
    clear_interaction_context_cache,
    dock_interaction,
)
from anchor_dock.interaction import pipeline as interaction_pipeline
from anchor_dock.interaction.pipeline import (
    _INTERACTION_CONTEXT_CACHE,
    INTERACTION_CONTEXT_CACHE_MAXSIZE,
    preselect_candidates,
)


def test_preselect_candidates_stable_ties() -> None:
    # 4 candidates, 2 matches, 2 conformers
    # candidate 0: match 0, conf 0
    # candidate 1: match 1, conf 0
    # candidate 2: match 0, conf 1
    # candidate 3: match 1, conf 1
    energies = torch.tensor([10.0, 15.0, 12.0, 11.0], dtype=torch.float32)
    match_indices = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    conformer_ordinals = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    # Let's preselect_k = 2.
    # Group 0 candidates: 0 (conf 0, E=10), 2 (conf 1, E=12)
    # Group 1 candidates: 1 (conf 0, E=15), 3 (conf 1, E=11)
    # The first selection from each match is its lowest-energy conformer winner.
    # Match 0 contributes candidate 0 and match 1 contributes candidate 3.
    selected = preselect_candidates(
        energies=energies,
        match_indices=match_indices,
        conformer_ordinals=conformer_ordinals,
        preselect_k=2,
        match_count=2,
        conformer_count=2,
    )
    assert selected == [0, 3]

    # Preselect k = 3
    selected_3 = preselect_candidates(
        energies=energies,
        match_indices=match_indices,
        conformer_ordinals=conformer_ordinals,
        preselect_k=3,
        match_count=2,
        conformer_count=2,
    )
    assert selected_3 == [0, 3, 2]


def test_preselect_candidates_uses_stable_candidate_id_ties() -> None:
    selected = preselect_candidates(
        energies=torch.ones(6),
        match_indices=torch.tensor([0, 1, 0, 1, 0, 1]),
        conformer_ordinals=torch.tensor([0, 0, 1, 1, 0, 0]),
        preselect_k=6,
        match_count=2,
        conformer_count=2,
    )
    assert selected == [0, 1, 2, 3, 4, 5]


def test_dock_interaction_no_optimize_smoke(cys_pdb, tmp_path) -> None:
    clear_interaction_context_cache()
    # Test dock_interaction with optimize=False
    output_dir = tmp_path / "smoke"
    result = dock_interaction(
        protein_pdb=cys_pdb,
        query_ligand="CCO",
        output_dir=output_dir,
        receptor_residue="CYS145:A",
        receptor_atom="SG",
        ligand_smarts="[#6:1]~[#8]",
        target_distance=3.0,
        distance_tolerance=0.5,
        num_confs=4,
        num_candidates=8,
        preselect_k=4,
        optimize=False,
        top_k=2,
        device="cpu",
        verbose=False,
    )

    assert result["mode"] == "interaction"
    assert result["num_poses"] > 0
    assert result["num_candidates"] == 8
    assert result["num_preselected"] == 4
    assert not result["optimization_applied"]

    # Verify SDF output file exists
    sdf_path = output_dir / "interaction_poses.sdf"
    assert sdf_path.is_file()

    # Load SDF and verify coordinate distances
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    poses = list(supplier)
    assert len(poses) > 0

    # The coordinates of the pivot atom should be exactly target_distance (3.0) away from the receptor coordinate
    receptor_coord = result["receptor_atom_coordinate"]
    for pose in poses:
        assert pose is not None
        # Verify metadata
        assert pose.GetProp("AnchorDock_Mode") == "interaction"
        assert pose.GetProp("AnchorDock_Output_Schema") == "3"
        assert pose.GetProp("AnchorDock_Search_Method") == "guided_random_placement"
        assert pose.GetProp("AnchorDock_Score_Semantics") == "interaction_conditioned_local_pose_ranking"

        # The scorer score/search energy properties must exist and not contain restraints
        score = float(pose.GetProp("AnchorDock_Score"))
        search_energy = float(pose.GetProp("AnchorDock_Search_Energy"))
        assert math.isfinite(score)
        assert math.isfinite(search_energy)

        # Retrieve the pivot/anchor atom coordinate
        pivot_idx = int(pose.GetProp("AnchorDock_Ligand_Anchor_Index"))
        pos = pose.GetConformer().GetAtomPosition(pivot_idx)
        dist = math.sqrt((pos.x - receptor_coord[0])**2 + (pos.y - receptor_coord[1])**2 + (pos.z - receptor_coord[2])**2)
        # Because optimize=False, final distance should be exactly target_distance (3.0) up to float precision
        assert dist == pytest.approx(3.0, abs=1e-3)


def test_dock_interaction_deterministic_seed(cys_pdb, tmp_path) -> None:
    out1 = tmp_path / "det1"
    out2 = tmp_path / "det2"

    result1 = dock_interaction(
        protein_pdb=cys_pdb,
        query_ligand="CCO",
        output_dir=out1,
        receptor_residue="CYS145:A",
        receptor_atom="SG",
        ligand_smarts="[#6:1]~[#8]",
        target_distance=3.0,
        distance_tolerance=0.5,
        num_confs=4,
        num_candidates=8,
        preselect_k=4,
        optimize=True,
        opt_steps=2,
        release_steps=1,
        random_seed=42,
        device="cpu",
        verbose=False,
    )

    result2 = dock_interaction(
        protein_pdb=cys_pdb,
        query_ligand="CCO",
        output_dir=out2,
        receptor_residue="CYS145:A",
        receptor_atom="SG",
        ligand_smarts="[#6:1]~[#8]",
        target_distance=3.0,
        distance_tolerance=0.5,
        num_confs=4,
        num_candidates=8,
        preselect_k=4,
        optimize=True,
        opt_steps=2,
        release_steps=1,
        random_seed=42,
        device="cpu",
        verbose=False,
    )

    assert result1["best_score"] == pytest.approx(result2["best_score"])
    assert result1["best_search_energy"] == pytest.approx(result2["best_search_energy"])
    poses1 = [pose for pose in Chem.SDMolSupplier(result1["output_file"], removeHs=False) if pose]
    poses2 = [pose for pose in Chem.SDMolSupplier(result2["output_file"], removeHs=False) if pose]
    assert [pose.GetProp("AnchorDock_Pose_ID") for pose in poses1] == [
        pose.GetProp("AnchorDock_Pose_ID") for pose in poses2
    ]
    for first, second in zip(poses1, poses2, strict=True):
        assert first.GetConformer().GetPositions() == pytest.approx(
            second.GetConformer().GetPositions()
        )


def test_dock_interaction_two_ligand_anchors(cys_pdb, tmp_path) -> None:
    # "CCO" (ethanol) has two carbon atoms. Under "[#6:1]", both are selected.
    output_dir = tmp_path / "two_anchors"
    result = dock_interaction(
        protein_pdb=cys_pdb,
        query_ligand="CCO",
        output_dir=output_dir,
        receptor_residue="CYS145:A",
        receptor_atom="SG",
        ligand_smarts="[#6:1]",
        target_distance=3.0,
        distance_tolerance=0.5,
        num_confs=2,
        num_candidates=4,
        preselect_k=2,
        optimize=False,
        device="cpu",
        verbose=False,
    )

    assert result["ligand_matches"] == [0, 1]  # Both carbons index 0 and 1 are matches!
    assert len(result["ligand_matches"]) == 2
    poses = [pose for pose in Chem.SDMolSupplier(result["output_file"], removeHs=False) if pose]
    assert {int(pose.GetProp("AnchorDock_Ligand_Anchor_Index")) for pose in poses} == {0, 1}


def test_dock_interaction_reports_only_physical_scores(cys_pdb, tmp_path, monkeypatch) -> None:
    penalty = 1_000_000.0

    def constant_restraint(distances, *args, **kwargs):
        return torch.full_like(distances, penalty)

    monkeypatch.setattr(interaction_pipeline, "flat_bottom_distance_restraint", constant_restraint)
    result = dock_interaction(
        protein_pdb=cys_pdb,
        query_ligand="CCO",
        output_dir=tmp_path / "score_separation",
        receptor_residue="CYS145:A",
        receptor_atom="SG",
        ligand_smarts="[#6:1]~[#8]",
        target_distance=3.0,
        distance_tolerance=0.5,
        num_confs=1,
        num_candidates=2,
        preselect_k=1,
        optimize=True,
        opt_steps=0,
        release_steps=0,
        top_k=1,
        device="cpu",
        verbose=False,
    )

    pose = next(iter(Chem.SDMolSupplier(result["output_file"], removeHs=False)))
    assert pose is not None
    assert float(pose.GetProp("AnchorDock_Final_Restraint_Energy")) == penalty
    assert float(result["guide_optimization"]["initial_best_energy"]) > penalty / 2
    assert float(pose.GetProp("AnchorDock_Score")) == pytest.approx(result["best_score"])
    assert float(pose.GetProp("AnchorDock_Search_Energy")) == pytest.approx(result["best_search_energy"])
    assert abs(result["best_search_energy"]) < penalty / 2
    assert result["optimization_requested"]
    assert not result["optimization_applied"]
    assert not result["optimization_improved"]
    assert pose.GetProp("AnchorDock_Search_Method") == "guided_random_placement"


def test_dock_interaction_preserves_sparse_conformer_id(cys_pdb, tmp_path, monkeypatch) -> None:
    original_generate = interaction_pipeline.generate_conformers_and_cluster

    def generate_with_sparse_id(*args, **kwargs):
        ligand, representative_ids = original_generate(*args, **kwargs)
        representative = Chem.Conformer(ligand.GetConformer(representative_ids[0]))
        ligand.RemoveAllConformers()
        representative.SetId(41)
        ligand.AddConformer(representative, assignId=False)
        return ligand, [41]

    monkeypatch.setattr(
        interaction_pipeline,
        "generate_conformers_and_cluster",
        generate_with_sparse_id,
    )
    result = dock_interaction(
        protein_pdb=cys_pdb,
        query_ligand="CCO",
        output_dir=tmp_path / "sparse_conformer",
        receptor_residue="CYS145:A",
        receptor_atom="SG",
        ligand_smarts="[#6:1]~[#8]",
        target_distance=3.0,
        distance_tolerance=0.5,
        num_confs=2,
        num_candidates=2,
        preselect_k=1,
        optimize=False,
        top_k=1,
        device="cpu",
        verbose=False,
    )
    pose = next(iter(Chem.SDMolSupplier(result["output_file"], removeHs=False)))
    assert pose is not None
    assert pose.GetProp("AnchorDock_Source_Conformer") == "41"
    assert pose.GetProp("AnchorDock_Source_Representative_Index") == "0"
    assert result["representative_conformer_ids"] == [41]


def test_dock_interaction_only_preselect_k_reaches_optimizer(cys_pdb, tmp_path, monkeypatch) -> None:
    original = DockingEngine.optimize_se3
    optimized_rows: list[int] = []

    def record_rows(self, problem, base_coords, pivot_atom_index, **kwargs):
        optimized_rows.append(int(base_coords.shape[0]))
        return original(self, problem, base_coords, pivot_atom_index, **kwargs)

    monkeypatch.setattr(DockingEngine, "optimize_se3", record_rows)
    output_dir = tmp_path / "preselect_k_limit"
    result = dock_interaction(
        protein_pdb=cys_pdb,
        query_ligand="CCO",
        output_dir=output_dir,
        receptor_residue="CYS145:A",
        receptor_atom="SG",
        ligand_smarts="[#6:1]~[#8]",
        target_distance=3.0,
        distance_tolerance=0.5,
        num_confs=4,
        num_candidates=8,
        preselect_k=2,
        optimize=True,
        opt_steps=2,
        release_steps=2,
        device="cpu",
        verbose=False,
    )

    assert result["guide_optimization"]["num_poses"] == 2
    assert result["release_optimization"]["num_poses"] == 2
    assert sum(optimized_rows) == 2


def test_dock_interaction_all_exported_rows_satisfy_interval(cys_pdb, tmp_path) -> None:
    output_dir = tmp_path / "interval_check"
    # We set target_distance = 3.0, tolerance = 0.5.
    # So final distance must be within [2.5, 3.5]
    result = dock_interaction(
        protein_pdb=cys_pdb,
        query_ligand="CCO",
        output_dir=output_dir,
        receptor_residue="CYS145:A",
        receptor_atom="SG",
        ligand_smarts="[#6:1]~[#8]",
        target_distance=3.0,
        distance_tolerance=0.5,
        num_confs=4,
        num_candidates=8,
        preselect_k=4,
        optimize=True,
        opt_steps=3,
        release_steps=3,
        device="cpu",
        verbose=False,
    )

    # Load all exported poses from SDF
    sdf_path = output_dir / "interaction_poses.sdf"
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    poses = list(supplier)

    receptor_coord = result["receptor_atom_coordinate"]
    for pose in poses:
        assert pose is not None
        pivot_idx = int(pose.GetProp("AnchorDock_Ligand_Anchor_Index"))
        pos = pose.GetConformer().GetAtomPosition(pivot_idx)
        dist = math.sqrt((pos.x - receptor_coord[0])**2 + (pos.y - receptor_coord[1])**2 + (pos.z - receptor_coord[2])**2)
        # Verify it satisfies target_distance (3.0) and distance_tolerance (0.5) with tiny epsilon 1e-5
        assert 2.5 - 1e-5 <= dist <= 3.5 + 1e-5


def test_dock_interaction_all_invalid_raises(cys_pdb, tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "all_invalid"
    original = DockingEngine.optimize_se3

    def force_invalid_release(self, *args, **kwargs):
        guided, final, guide_stats, release_stats = original(self, *args, **kwargs)
        return guided, final + 100.0, guide_stats, release_stats

    monkeypatch.setattr(DockingEngine, "optimize_se3", force_invalid_release)
    with pytest.raises(RuntimeError, match="no poses satisfied the interaction distance restraint"):
        dock_interaction(
            protein_pdb=cys_pdb,
            query_ligand="CCO",
            output_dir=output_dir,
            receptor_residue="CYS145:A",
            receptor_atom="SG",
            ligand_smarts="[#6:1]~[#8]",
            target_distance=3.0,
            distance_tolerance=0.5,
            num_confs=2,
            num_candidates=4,
            preselect_k=2,
            optimize=True,
            opt_steps=0,
            release_steps=0,
            device="cpu",
            verbose=False,
        )


def test_dock_interaction_invalid_arguments(cys_pdb, tmp_path) -> None:
    output_dir = tmp_path / "invalid_args"
    # Target distance <= 0
    with pytest.raises(ValueError, match="target_distance"):
        dock_interaction(
            protein_pdb=cys_pdb, query_ligand="CCO", output_dir=output_dir,
            receptor_residue="CYS145:A", receptor_atom="SG", ligand_smarts="[#6:1]~[#8]",
            target_distance=-1.0, distance_tolerance=0.5,
        )

    # Tolerance <= 0
    with pytest.raises(ValueError, match="distance_tolerance"):
        dock_interaction(
            protein_pdb=cys_pdb, query_ligand="CCO", output_dir=output_dir,
            receptor_residue="CYS145:A", receptor_atom="SG", ligand_smarts="[#6:1]~[#8]",
            target_distance=3.0, distance_tolerance=-0.5,
        )

    # Tolerance >= target
    with pytest.raises(ValueError, match="distance_tolerance"):
        dock_interaction(
            protein_pdb=cys_pdb, query_ligand="CCO", output_dir=output_dir,
            receptor_residue="CYS145:A", receptor_atom="SG", ligand_smarts="[#6:1]~[#8]",
            target_distance=3.0, distance_tolerance=4.0,
        )

    # preselect_k > num_candidates
    with pytest.raises(ValueError, match="preselect_k"):
        dock_interaction(
            protein_pdb=cys_pdb, query_ligand="CCO", output_dir=output_dir,
            receptor_residue="CYS145:A", receptor_atom="SG", ligand_smarts="[#6:1]~[#8]",
            target_distance=3.0, distance_tolerance=0.5,
            num_candidates=4, preselect_k=8,
        )

    with pytest.raises(ValueError, match="release_steps"):
        dock_interaction(
            protein_pdb=cys_pdb, query_ligand="CCO", output_dir=output_dir,
            receptor_residue="CYS145:A", receptor_atom="SG", ligand_smarts="[#6:1]~[#8]",
            target_distance=3.0, distance_tolerance=0.5,
            optimize=True, opt_steps=1, release_steps=0,
        )

    with pytest.raises(ValueError, match="optimizer"):
        dock_interaction(
            protein_pdb=cys_pdb, query_ligand="CCO", output_dir=output_dir,
            receptor_residue="CYS145:A", receptor_atom="SG", ligand_smarts="[#6:1]~[#8]",
            target_distance=3.0, distance_tolerance=0.5,
            optimizer="bogus",  # type: ignore[arg-type]
            optimize=False,
        )

    # num_candidates < match_count
    # ligand_smarts "[#6:1]" matched on CCO has 2 matches.
    # If num_candidates = 1, it should fail.
    with pytest.raises(ValueError, match="num_candidates .* is less than the number of ligand anchor matches"):
        dock_interaction(
            protein_pdb=cys_pdb, query_ligand="CCO", output_dir=output_dir,
            receptor_residue="CYS145:A", receptor_atom="SG", ligand_smarts="[#6:1]",
            target_distance=3.0, distance_tolerance=0.5,
            num_candidates=1, preselect_k=1,
        )


def test_dock_interaction_cache_clear(cys_pdb, tmp_path) -> None:
    clear_interaction_context_cache()
    assert len(_INTERACTION_CONTEXT_CACHE) == 0

    output_dir = tmp_path / "cache_test"
    dock_interaction(
        protein_pdb=cys_pdb,
        query_ligand="CCO",
        output_dir=output_dir,
        receptor_residue="CYS145:A",
        receptor_atom="SG",
        ligand_smarts="[#6:1]~[#8]",
        target_distance=3.0,
        distance_tolerance=0.5,
        num_confs=2,
        num_candidates=4,
        preselect_k=2,
        optimize=False,
        device="cpu",
        verbose=False,
    )

    assert len(_INTERACTION_CONTEXT_CACHE) == 1
    clear_interaction_context_cache()
    assert len(_INTERACTION_CONTEXT_CACHE) == 0


def test_dock_interaction_context_cache_is_bounded(cys_pdb, tmp_path) -> None:
    clear_interaction_context_cache()
    for index in range(INTERACTION_CONTEXT_CACHE_MAXSIZE + 2):
        _INTERACTION_CONTEXT_CACHE[(str(index), "R", "A", 1.0, True, "cpu", "v")] = object()

    dock_interaction(
        protein_pdb=cys_pdb,
        query_ligand="CCO",
        output_dir=tmp_path / "bounded_cache",
        receptor_residue="CYS145:A",
        receptor_atom="SG",
        ligand_smarts="[#6:1]~[#8]",
        target_distance=3.0,
        distance_tolerance=0.5,
        num_confs=1,
        num_candidates=1,
        preselect_k=1,
        optimize=False,
        device="cpu",
        verbose=False,
    )
    assert len(_INTERACTION_CONTEXT_CACHE) == INTERACTION_CONTEXT_CACHE_MAXSIZE
    clear_interaction_context_cache()


def test_dock_interaction_rejects_altloc_in_scored_pocket(cys_pdb, tmp_path) -> None:
    lines = cys_pdb.read_text().splitlines(keepends=True)
    lines[4] = f"{lines[4][:16]}A{lines[4][17:]}"
    ambiguous = tmp_path / "pocket-altloc.pdb"
    ambiguous.write_text("".join(lines))

    with pytest.raises(ValueError, match="alternate locations.*scored receptor pocket"):
        dock_interaction(
            protein_pdb=ambiguous,
            query_ligand="CCO",
            output_dir=tmp_path / "altloc",
            receptor_residue="CYS145:A",
            receptor_atom="SG",
            ligand_smarts="[#6:1]~[#8]",
            target_distance=3.0,
            distance_tolerance=0.5,
            num_confs=1,
            num_candidates=1,
            preselect_k=1,
            optimize=False,
            device="cpu",
            verbose=False,
        )


def test_dock_interaction_supports_explicit_blank_chain(cys_pdb, tmp_path) -> None:
    lines = cys_pdb.read_text().splitlines(keepends=True)
    blank_chain = tmp_path / "blank-chain.pdb"
    blank_chain.write_text("".join(
        f"{line[:21]} {line[22:]}" if line.startswith("ATOM") else line
        for line in lines
    ))
    result = dock_interaction(
        protein_pdb=blank_chain,
        query_ligand="CCO",
        output_dir=tmp_path / "blank-chain",
        receptor_residue="CYS145:",
        receptor_atom="SG",
        ligand_smarts="[#6:1]~[#8]",
        target_distance=3.0,
        distance_tolerance=0.5,
        num_confs=1,
        num_candidates=1,
        preselect_k=1,
        optimize=False,
        device="cpu",
        verbose=False,
    )
    assert result["receptor_residue"] == "CYS145:"
