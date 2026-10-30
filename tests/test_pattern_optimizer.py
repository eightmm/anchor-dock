"""Black-box pose objectives must be usable without inventing gradients."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from rdkit import Chem

from anchor_dock.core.grid import TrilinearGrid
from anchor_dock.core.native_vina import native_vina_score
from anchor_dock.core.optimization import SE3PoseModel, optimize_pose_module, optimize_torsions
from anchor_dock.core.prepared import PreparedLigand
from anchor_dock.core.scoring import PairwiseScorer


class CartesianPose(torch.nn.Module):
    def __init__(self, xyz):
        super().__init__()
        self.xyz = torch.nn.Parameter(torch.as_tensor(xyz, dtype=torch.float64).clone())

    def forward(self):
        return self.xyz


def run(model, energy, **options):
    kwargs = dict(optimizer="pattern", learning_rate=.2, num_steps=50, min_delta=0., early_stopping=False)
    kwargs.update(options)
    return optimize_pose_module(model, energy, **kwargs)


def quantized_energy(x):
    assert not x.requires_grad
    return torch.floor(25. * x.square()).sum((1, 2)).detach()


def test_quantized_black_box_reaches_zero_without_coordinate_or_weight_gradients():
    model = CartesianPose([[[1.2, -.8, .3]]])
    weight = torch.nn.Parameter(torch.tensor(2.))
    output, stats = run(model, lambda x: weight * quantized_energy(x))
    assert quantized_energy(output.detach()).item() == 0.
    assert stats.initial_best_energy > stats.final_best_energy == 0.
    assert model.xyz.grad is weight.grad is None


def test_pattern_accepts_only_decreases_and_bounds_actual_displacement():
    model = CartesianPose([[[1.3, -.9, .3]]])
    previous = model().detach().clone()
    value = quantized_energy(previous)
    for _ in range(20):
        result, _ = run(model, quantized_energy, num_steps=1, max_atom_displacement=.15)
        updated = result.detach().clone()
        assert quantized_energy(updated) <= value
        assert torch.linalg.vector_norm(updated - previous, dim=2).max() <= .15000001
        previous, value = updated, quantized_energy(updated)


def test_pattern_has_independent_batch_rows_and_per_pose_budgets():
    initial = torch.tensor([[[1.1, .6, -.4]], [[100., 80., -20.]]], dtype=torch.float64)
    batch, stats = run(CartesianPose(initial), quantized_energy, num_steps=12, pattern_max_evaluations=48)
    assert stats.num_poses == 2
    for row in range(2):
        single, _ = run(CartesianPose(initial[row:row + 1]), quantized_energy,
                        num_steps=12, pattern_max_evaluations=48)
        torch.testing.assert_close(batch[row], single[0], rtol=0, atol=0)


def test_failed_domain_trials_backtrack_without_shrinking_neighbor_step():
    values = -torch.arange(2, dtype=torch.float64).reshape(1, 1, 1, 2).expand(1, 2, 2, 2).clone()
    field = TrilinearGrid(values, origin=[0., 0., 0.], spacing=1.)
    initial = torch.tensor([[[.9, .5, .5]], [[.1, .5, .5]]], dtype=torch.float64)

    def energy(x):
        return field(x, [0]).sum(1)

    batch, _ = run(CartesianPose(initial), energy, num_steps=1, pattern_step=.25)
    assert .9 < batch[0, 0, 0] <= 1.
    assert batch[1, 0, 0] == .35
    for row in range(2):
        single, _ = run(CartesianPose(initial[row:row + 1]), energy, num_steps=1, pattern_step=.25)
        torch.testing.assert_close(batch[row], single[0], rtol=0, atol=0)


def test_nonfinite_trial_energy_is_rejected_and_backtracked():
    model = CartesianPose([[[.9, .5, .5]]])

    def energy(x):
        return torch.where(x[:, 0, 0] <= 1., -x[:, 0, 0], float("nan"))

    result, _ = run(model, energy, num_steps=1, pattern_step=.25)
    assert .9 < result[0, 0, 0] <= 1.


def test_unexpected_scorer_failure_restores_last_accepted_state():
    model = CartesianPose([[[1., 0., 0.]]])
    original = model().detach().clone()

    def energy(x):
        if x[0, 0, 0] < .9:
            raise ValueError("broken external scorer")
        return x.square().sum((1, 2)).detach()

    with pytest.raises(ValueError, match="broken external scorer"):
        run(model, energy, num_steps=1)
    torch.testing.assert_close(model(), original, rtol=0, atol=0)


def test_budget_counts_initial_and_trial_calls_exactly():
    calls = []

    def energy(x):
        calls.append(x.detach().clone())
        return x.square().sum((1, 2)).detach()

    _, stats = run(CartesianPose([[[1., 2., 3.]]]), energy, pattern_max_evaluations=7)
    assert len(calls) == 7
    assert stats.maximum_steps == 1
    calls.clear()
    model = CartesianPose([[[1., 2., 3.]]])
    original = model().detach().clone()
    result, stats = run(model, energy, pattern_max_evaluations=1)
    assert len(calls) == 1
    assert stats.maximum_steps == 0
    torch.testing.assert_close(result, original, rtol=0, atol=0)


def test_detached_view_objective_has_correct_initial_and_best_statistics():
    model = CartesianPose([[[1., 0., 0.]]])
    result, stats = run(model, lambda x: x[:, 0, 0], num_steps=2)
    assert stats.initial_best_energy == 1.
    assert stats.final_best_energy == result.detach()[0, 0, 0]
    assert stats.final_best_energy < 1.


def test_integer_initial_value_does_not_truncate_later_floating_point_scores():
    model = CartesianPose([[[2., 0., 0.]]])
    first = True

    def energy(x):
        nonlocal first
        if first:
            first = False
            return torch.tensor([2])
        return x[:, 0, 0]

    result, stats = run(model, energy, num_steps=2)
    assert result.detach()[0, 0, 0] < 1.7
    assert stats.initial_best_energy == 2.
    assert stats.final_best_energy == result.detach()[0, 0, 0]


def test_fixed_redundant_directions_do_not_spend_objective_calls():
    class RedundantPose(CartesianPose):
        def __init__(self):
            super().__init__([[[0., 0., 0.]]])
            self.unused = torch.nn.Parameter(torch.zeros(1, 8, dtype=torch.float64))

    calls = []

    def energy(x):
        calls.append(1)
        return x.square().sum((1, 2)).detach()

    run(RedundantPose(), energy, num_steps=1)
    assert len(calls) == 7  # initial + 2 signs * 3 Cartesian actions


def _prepared(payload):
    molecule = Chem.MolFromMolBlock(payload["molblock"], removeHs=False)
    return PreparedLigand.from_pdbqt(molecule, payload["pdbqt"], serial_to_atom_index={
        int(key): value for key, value in payload["serial_to_atom_index"].items()})


def test_nondifferentiable_native_final_energy_can_refine_se3_and_torsions():
    oracle = json.loads((Path(__file__).parent / "fixtures/native_scoring.json").read_text())
    ligand, receptor = _prepared(oracle["cases"]["secondary_amine"]), _prepared(oracle["receptor"])
    scorer = PairwiseScorer("vina").prepare(
        ligand.features(), receptor.coordinates(), receptor.features(),
        num_rotatable_bonds=ligand.vina_num_tors, intramolecular_mask=ligand.intramolecular_mask(),
    )
    xyz = ligand.coordinates()[None]
    model = SE3PoseModel(ligand.mol, xyz, 0, xyz[:, 0], torch.zeros(1, 3), "cpu")

    def energy(x):
        return native_vina_score(scorer, x, intramolecular_reference=0.).score

    result, stats = run(model, energy, num_steps=3, pattern_max_evaluations=120)
    assert stats.final_best_energy < stats.initial_best_energy
    assert torch.isfinite(result).all()
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in ligand.mol.GetBonds()]
    left, right = torch.tensor(bonds).T
    torch.testing.assert_close(
        torch.linalg.vector_norm(result[:, left] - result[:, right], dim=-1),
        torch.linalg.vector_norm(xyz[:, left] - xyz[:, right], dim=-1),
        rtol=2e-6, atol=2e-6,
    )


def test_torsion_entry_point_preserves_frozen_anchor_and_shape():
    mol = Chem.MolFromSmiles("CCCC")
    xyz = torch.tensor([[[0., 0., 0.], [1., .1, 0.], [2., 1., .2], [3., 1., 1.]]])

    class Scorer:
        def search_energy(self, x):
            return x[:, -1, 2].square().detach()

    result, stats = optimize_torsions(mol, [0], xyz.repeat(2, 1, 1), Scorer(), "cpu",
                                     optimizer="pattern", num_steps=3, learning_rate=.2)
    assert result.shape == (2, 4, 3)
    torch.testing.assert_close(result[:, 0], xyz[:, 0].repeat(2, 1))
    assert stats.num_poses == 2


@pytest.mark.parametrize("options,match", [
    ({"pattern_step": 0.}, "pattern_step"), ({"pattern_step": True}, "pattern_step"),
    ({"pattern_step": float("nan")}, "pattern_step"),
    ({"pattern_min_step": 0.}, "pattern_min_step"), ({"pattern_min_step": .3}, "pattern_min_step"),
    ({"pattern_shrink": 0.}, "pattern_shrink"), ({"pattern_shrink": 1.}, "pattern_shrink"),
    ({"pattern_shrink": True}, "pattern_shrink"), ({"pattern_shrink": float("inf")}, "pattern_shrink"),
    ({"pattern_max_evaluations": 0}, "pattern_max_evaluations"),
    ({"pattern_max_evaluations": 1.5}, "pattern_max_evaluations"),
    ({"pattern_max_evaluations": True}, "pattern_max_evaluations"),
    ({"max_atom_displacement": 0.}, "max_atom_displacement"),
    ({"max_atom_displacement": float("nan")}, "max_atom_displacement"),
    ({"max_backtracks": -1}, "max_backtracks"), ({"max_backtracks": True}, "max_backtracks"),
])
def test_pattern_invalid_options_fail_before_noop(options, match):
    with pytest.raises(ValueError, match=match):
        run(CartesianPose([[[1., 0., 0.]]]), quantized_energy, num_steps=0, **options)


def test_pattern_rejects_nonfinite_initial_energy_and_unbounded_chart_size():
    with pytest.raises(FloatingPointError):
        run(CartesianPose([[[1., 0., 0.]]]), lambda x: x.new_tensor([float("nan")]))
    with pytest.raises(ValueError, match="256"):
        run(CartesianPose(torch.zeros(1, 100, 3)), quantized_energy)


def test_pattern_invalid_initial_grid_position_is_not_hidden():
    field = TrilinearGrid(torch.zeros(1, 2, 2, 2, dtype=torch.float64), origin=[0., 0., 0.], spacing=1.)
    with pytest.raises(ValueError, match="bounds"):
        run(CartesianPose([[[1.1, .5, .5]]]), lambda x: field(x, [0]).sum(1))


def test_finite_coordinates_do_not_hide_overflowing_trial_parameter_state():
    class SaturatedPose(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q = torch.nn.Parameter(torch.zeros(1, 1, 3, dtype=torch.float32))

        def forward(self):
            return torch.sigmoid(self.q)

    model = SaturatedPose()
    result, _ = run(model, lambda x: -x.sum((1, 2)), num_steps=1,
                    pattern_step=1e100, max_atom_displacement=2.)
    assert torch.isfinite(model.q).all()
    torch.testing.assert_close(result, torch.full_like(result, .5))


def test_mixed_parameter_dtypes_are_checked_before_copy_downcast():
    class MixedPose(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q = torch.nn.Parameter(torch.zeros(1, 1, 3, dtype=torch.float32))
            self.unused = torch.nn.Parameter(torch.zeros(1, 1, dtype=torch.float64))

        def forward(self):
            return torch.sigmoid(self.q)

    model = MixedPose()
    run(model, lambda x: -x.sum((1, 2)), num_steps=1,
        pattern_step=1e100, max_atom_displacement=2.)
    assert torch.isfinite(model.q).all()
    assert torch.isfinite(model.unused).all()


def test_nonfinite_initial_parameters_are_rejected_even_with_finite_model_output():
    class SaturatedPose(CartesianPose):
        def forward(self):
            return torch.sigmoid(self.xyz)

    with pytest.raises(ValueError, match="initial parameters"):
        run(SaturatedPose([[[float("inf"), 0., 0.]]]), lambda x: -x.sum((1, 2)))
