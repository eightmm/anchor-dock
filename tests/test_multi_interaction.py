from __future__ import annotations

import json

import pytest
import torch
from rdkit import Chem

from anchor_dock import InteractionConstraint, dock_interaction
from anchor_dock.interaction.hypotheses import (
    enumerate_joint_hypotheses,
    pairwise_shell_feasible,
)
from anchor_dock.interaction.pipeline import (
    _allocate_candidate_states,
    _candidate_state_cycle,
    _coarse_selection_energies,
    _inclusive_distance_window_mask,
)
from anchor_dock.interaction.restraint import (
    flat_bottom_distance_restraint,
    flat_bottom_distance_restraint_matrix,
    interaction_distance_matrix,
    mean_flat_bottom_distance_restraint,
)
from anchor_dock.interaction.selectors import LigandAnchorMatch, MatchLimitExceededError
from anchor_dock.interaction.spec import MAX_INTERACTIONS, normalize_interactions


def _constraint(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "receptor_residue": "CYS145:A",
        "receptor_atom": "SG",
        "ligand_smarts": "[O:1]",
        "target_distance": 3.0,
        "distance_tolerance": 0.5,
    }
    value.update(updates)
    return value


def _anchor(match_index: int, atom_index: int) -> LigandAnchorMatch:
    return LigandAnchorMatch(
        match_index=match_index,
        ligand_atom_index=atom_index,
        representative_match=(atom_index,),
        element="C",
        formal_charge=0,
    )


def test_interaction_constraint_normalizes_and_serializes() -> None:
    item = InteractionConstraint(" CYS145:A ", " sg ", " [O:1] ", 3, 0.5, 2)
    assert item.receptor_residue == "CYS145:A"
    assert item.receptor_atom == "SG"
    assert item.as_dict() == {
        "receptor_residue": "CYS145:A",
        "receptor_atom": "SG",
        "ligand_smarts": "[O:1]",
        "target_distance": 3.0,
        "distance_tolerance": 0.5,
        "restraint_weight": 2.0,
    }


def test_normalize_interactions_preserves_legacy_and_canonical_forms() -> None:
    legacy = normalize_interactions(
        interactions=None,
        receptor_residue="CYS145:A",
        receptor_atom="SG",
        ligand_smarts="[O:1]",
        target_distance=3.0,
        distance_tolerance=0.5,
        default_restraint_weight=7.0,
    )
    canonical = normalize_interactions(
        interactions=[_constraint(), _constraint(receptor_atom="O", restraint_weight=4.0)],
        default_restraint_weight=7.0,
    )
    assert len(legacy) == 1
    assert legacy[0].restraint_weight == 7.0
    assert [item.restraint_weight for item in canonical] == [7.0, 4.0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"interactions": []}, "non-empty"),
        (
            {
                "interactions": [_constraint()],
                "receptor_residue": "CYS145:A",
            },
            "cannot be combined",
        ),
        ({"interactions": [_constraint(), _constraint()]}, "duplicates"),
        ({"interactions": [_constraint(extra=True)]}, "unsupported fields"),
        ({"interactions": [{"receptor_atom": "SG"}]}, "requires fields"),
        (
            {"interactions": [_constraint() for _ in range(MAX_INTERACTIONS + 1)]},
            "at most",
        ),
    ],
)
def test_normalize_interactions_rejects_ambiguous_or_unbounded_input(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        normalize_interactions(default_restraint_weight=10.0, **kwargs)


def test_joint_hypotheses_are_ordered_bounded_and_allow_shared_atoms() -> None:
    groups = [(_anchor(0, 1), _anchor(1, 2)), (_anchor(0, 1), _anchor(1, 3))]
    hypotheses = enumerate_joint_hypotheses(groups, max_joint_matches=4)
    assert [item.ligand_atom_indices for item in hypotheses] == [
        (1, 1),
        (1, 3),
        (2, 1),
        (2, 3),
    ]
    with pytest.raises(MatchLimitExceededError, match="4 joint hypotheses"):
        enumerate_joint_hypotheses(groups, max_joint_matches=3)


def test_pairwise_shell_feasibility_is_conservative_and_supports_shared_atom() -> None:
    receptor = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    targets = torch.tensor([2.0, 2.0])
    tolerances = torch.tensor([0.5, 0.5])
    impossible = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    feasible = torch.tensor([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    assert not pairwise_shell_feasible(impossible, (0, 1), receptor, targets, tolerances)
    assert pairwise_shell_feasible(feasible, (0, 1), receptor, targets, tolerances)

    bridging_receptor = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    assert pairwise_shell_feasible(
        impossible,
        (0, 0),
        bridging_receptor,
        targets,
        tolerances,
    )


def test_pairwise_shell_feasibility_does_not_reject_float32_boundary() -> None:
    receptor = torch.tensor([[0.0, 0.0, 0.0], [32.044, 0.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([2.727, 4.94], dtype=torch.float32)
    tolerances = torch.tensor([0.05, 0.694], dtype=torch.float32)
    ligand = torch.tensor([[0.0, 0.0, 0.0], [40.455, 0.0, 0.0]], dtype=torch.float32)

    # These decimal values meet the upper shell boundary exactly; independent
    # float32 rounding differs by about 3.8e-6 Angstrom.
    assert pairwise_shell_feasible(ligand, (0, 1), receptor, targets, tolerances)


def test_candidate_state_cycle_is_round_robin_across_viable_hypotheses() -> None:
    active, states = _candidate_state_cycle([[0, 2], [], [1], [0, 1, 2]])
    assert active == [0, 2, 3]
    assert states == [
        (0, 0, 0),
        (1, 2, 1),
        (2, 3, 0),
        (0, 0, 2),
        (2, 3, 1),
        (2, 3, 2),
    ]


def test_candidate_allocation_is_equal_across_hypotheses() -> None:
    active, states = _allocate_candidate_states([[0], list(range(10))], 128)
    counts = [sum(state[0] == group for state in states) for group in range(len(active))]

    assert active == [0, 1]
    assert max(counts) - min(counts) <= 1
    assert [state[2] for state in states if state[0] == 0] == [0] * counts[0]
    assert [state[2] for state in states if state[0] == 1][:12] == [
        *range(10),
        0,
        1,
    ]


def test_final_window_gate_includes_decimal_boundary() -> None:
    ligand = torch.tensor([[-12.1034, 0.0, 0.0]], dtype=torch.float64)
    receptor = torch.tensor([-7.8407, 0.0, 0.0], dtype=torch.float64)
    distances = torch.linalg.vector_norm(ligand - receptor, dim=1)[:, None]
    targets = torch.tensor([3.5427], dtype=torch.float64)
    tolerances = torch.tensor([0.72], dtype=torch.float64)

    assert distances.item() > (targets + tolerances).item()
    assert _inclusive_distance_window_mask(distances, targets, tolerances).item()


def test_multi_restraint_matrix_preserves_single_behavior_and_gradients() -> None:
    coords = torch.tensor(
        [[[4.0, 0.0, 0.0], [0.0, 5.0, 0.0]]],
        requires_grad=True,
    )
    receptor = torch.zeros((2, 3))
    distances = interaction_distance_matrix(coords, (0, 1), receptor)
    matrix = flat_bottom_distance_restraint_matrix(
        distances,
        [3.0, 3.0],
        [0.5, 0.5],
        [2.0, 4.0],
    )
    assert distances.tolist() == [[4.0, 5.0]]
    assert matrix.tolist() == [[0.5, 9.0]]
    assert mean_flat_bottom_distance_restraint(distances, [3.0, 3.0], [0.5, 0.5], [2.0, 4.0]).item() == pytest.approx(
        4.75
    )
    assert torch.allclose(
        matrix[:, :1].squeeze(1),
        flat_bottom_distance_restraint(distances[:, 0], 3.0, 0.5, 2.0),
    )
    matrix.mean().backward()
    assert coords.grad is not None
    assert torch.any(coords.grad.abs() > 0)


def test_secondary_restraints_change_coarse_preselection_energy() -> None:
    physical = torch.tensor([1.0, 1.0])
    penalties = torch.tensor([[0.0, 8.0], [0.0, 2.0]])
    assert _coarse_selection_energies(physical, penalties).tolist() == [5.0, 2.0]


@pytest.mark.parametrize("optimize", [False, True])
def test_multi_interaction_exports_all_constraint_provenance(
    cys_pdb,
    tmp_path,
    optimize: bool,
) -> None:
    interactions = [
        _constraint(),
        _constraint(
            receptor_residue="ALA146:A",
            receptor_atom="O",
            ligand_smarts="[C:1][O]",
            target_distance=20.0,
            distance_tolerance=19.9,
            restraint_weight=3.0,
        ),
    ]
    result = dock_interaction(
        protein_pdb=cys_pdb,
        query_ligand="CCO",
        output_dir=tmp_path / f"multi-{optimize}",
        interactions=interactions,
        num_confs=1,
        num_candidates=4,
        preselect_k=2,
        optimize=optimize,
        opt_steps=2,
        release_steps=1,
        opt_lr=0.01,
        opt_batch_size=2,
        top_k=2,
        device="cpu",
        verbose=False,
    )
    assert result["num_interactions"] == 2
    assert result["interaction_logic"] == "all"
    assert result["primary_interaction_index"] == 0
    assert result["num_joint_hypotheses"] == 1
    assert result["receptor_structure_scope"] == "residue_union_pocket"
    assert result["optimization_applied"] is optimize
    assert len(result["pose_interactions"]) == result["num_poses"]
    assert all(len(pose_record["interactions"]) == 2 for pose_record in result["pose_interactions"])

    poses = [pose for pose in Chem.SDMolSupplier(result["output_file"], removeHs=False) if pose]
    assert poses
    for pose in poses:
        assert pose.GetProp("AnchorDock_Output_Schema") == "4"
        assert pose.GetProp("AnchorDock_Interaction_Logic") == "all"
        assert pose.GetProp("AnchorDock_Restraint_Aggregation") == "mean"
        values = json.loads(pose.GetProp("AnchorDock_Interaction_Distances"))
        assert len(values) == 2
        assert all(value["satisfied"] for value in values)
        assert 2.5 <= values[0]["final_distance"] <= 3.5
        assert 0.1 <= values[1]["final_distance"] <= 39.9


def test_dock_interaction_rejects_mixed_multi_and_legacy_before_io(tmp_path) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        dock_interaction(
            protein_pdb=tmp_path / "missing.pdb",
            query_ligand="CCO",
            interactions=[_constraint()],
            receptor_residue="CYS145:A",
        )
