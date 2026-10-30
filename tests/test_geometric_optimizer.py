"""Independent geometric and objective contracts for Cartesian pullback steps."""

from __future__ import annotations

import pytest
import torch
from rdkit import Chem

from anchor_dock.core.grid import TrilinearGrid
from anchor_dock.core.kinematics import rotation_vector_to_matrix
from anchor_dock.core.optimization import (
    SE3PoseModel,
    _batch_parameters,
    _cartesian_pose_jacobian,
    _copy_flat_pose_state_,
    _flat_pose_state,
    optimize_pose_module,
)


class CartesianPose(torch.nn.Module):
    def __init__(self, coordinates):
        super().__init__()
        self.xyz = torch.nn.Parameter(torch.as_tensor(coordinates, dtype=torch.float64).clone())

    def forward(self):
        return self.xyz


class RigidPose(torch.nn.Module):
    def __init__(self, coordinates, translation, rotation):
        super().__init__()
        self.register_buffer("base_coords", coordinates)
        self.translations = torch.nn.Parameter(translation.clone())
        self.rotation_vectors = torch.nn.Parameter(rotation.clone())

    def forward(self):
        return self.base_coords @ rotation_vector_to_matrix(self.rotation_vectors).transpose(1, 2) + self.translations[:, None]


class RedundantPose(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left = torch.nn.Parameter(torch.tensor([[2., -1., 3.]], dtype=torch.float64))
        self.right = torch.nn.Parameter(torch.zeros(1, 3, dtype=torch.float64))
        self.frozen_action = torch.nn.Parameter(torch.zeros(1, 2, dtype=torch.float64))

    def forward(self):
        return (self.left + self.right + 0. * self.frozen_action.sum(dim=1, keepdim=True))[:, None]


def cartesian_run(model, energy, **kwargs):
    return optimize_pose_module(
        model, energy, optimizer="cartesian", learning_rate=1., min_delta=0., early_stopping=False, **kwargs,
    )


def test_cartesian_direction_matches_known_quadratic_step():
    model = CartesianPose([[[0.1, -0.1, 0.1]]])
    initial = model().detach().clone()
    result, stats = cartesian_run(model, lambda x: .5 * x.square().sum((1, 2)), num_steps=1)
    torch.testing.assert_close(result, initial * (1. - 1. / 1.001), atol=1e-14, rtol=1e-12)
    assert stats.final_best_energy < stats.initial_best_energy


def test_cartesian_solves_redundant_and_frozen_actions_without_singular_solve():
    model = RedundantPose()
    result, stats = cartesian_run(model, lambda x: .5 * x.square().sum((1, 2)), num_steps=25)
    assert result.detach().abs().max() < 1e-10
    assert stats.final_best_energy < 1e-20
    torch.testing.assert_close(model.frozen_action, torch.zeros_like(model.frozen_action))


def test_cartesian_is_independent_of_high_gradient_batch_neighbor():
    coordinates = torch.tensor([[[2., .3, -.5]], [[100., 80., -20.]]], dtype=torch.float64)
    target = torch.zeros_like(coordinates)
    batched, _ = cartesian_run(CartesianPose(coordinates), lambda x: .5 * (x - target).square().sum((1, 2)), num_steps=12)
    for row in range(2):
        single, _ = cartesian_run(CartesianPose(coordinates[row:row + 1]), lambda x: .5 * x.square().sum((1, 2)), num_steps=12)
        torch.testing.assert_close(batched[row], single[0], rtol=1e-13, atol=1e-13)


def test_pullback_cartesian_jacobian_matches_finite_differences_and_parameter_vjp():
    mol = Chem.MolFromSmiles("CCCC")
    coordinates = torch.tensor([[[0., 0., 0.], [1., .1, 0.], [2., 1., .2], [3., 1., 1.]]])
    model = SE3PoseModel(mol, coordinates, 0, torch.tensor([[.2, -.1, .4]]), torch.tensor([[.3, -.2, .4]]), "cpu").double()
    with torch.no_grad():
        model.kinematics.thetas.fill_(.35)
    parameters = _batch_parameters(model, 1)
    flat = _flat_pose_state(parameters)
    jacobian = _cartesian_pose_jacobian(model, parameters, flat, 0)
    columns = []
    epsilon = 1e-6
    for column in range(flat.shape[1]):
        offset = torch.zeros_like(flat)
        offset[0, column] = epsilon
        _copy_flat_pose_state_(parameters, flat + offset)
        positive = model().detach().reshape(-1)
        _copy_flat_pose_state_(parameters, flat - offset)
        negative = model().detach().reshape(-1)
        columns.append((positive - negative) / (2 * epsilon))
    _copy_flat_pose_state_(parameters, flat)
    torch.testing.assert_close(jacobian, torch.stack(columns, dim=1), rtol=2e-8, atol=2e-9)
    coords = model()
    weights = torch.arange(1, coords.numel() + 1, dtype=coords.dtype).reshape_as(coords)
    gradients = torch.autograd.grad((coords * weights).sum(), parameters)
    parameter_gradient = torch.cat([value.reshape(-1) for value in gradients])
    torch.testing.assert_close(jacobian.T @ weights.reshape(-1), parameter_gradient, rtol=1e-12, atol=1e-12)


def test_cartesian_rigid_steps_are_rotation_and_translation_equivariant():
    base = torch.tensor([[[0., 0., 0.], [1., .2, .3], [-.2, 1.1, .1], [.1, -.4, 1.3]]], dtype=torch.float64)
    centers = torch.tensor([[.5, -.3, 1.]], dtype=torch.float64)
    rotation = torch.tensor([[.4, -.2, .3]], dtype=torch.float64)
    target = base + torch.tensor([[[.7, -.6, .2]]], dtype=torch.float64)
    world_rotation = rotation_vector_to_matrix(torch.tensor([[.7, -.5, .2]], dtype=torch.float64))[0]
    world_translation = torch.tensor([20., -10., 30.], dtype=torch.float64)
    first = RigidPose(base, centers, rotation)
    second = RigidPose(base @ world_rotation.T, centers @ world_rotation.T + world_translation, rotation @ world_rotation.T)
    transformed_target = target @ world_rotation.T + world_translation
    output, _ = cartesian_run(first, lambda x: .5 * (x - target).square().sum((1, 2)), num_steps=8)
    transformed, _ = cartesian_run(second, lambda x: .5 * (x - transformed_target).square().sum((1, 2)), num_steps=8)
    torch.testing.assert_close(transformed, output @ world_rotation.T + world_translation, atol=2e-11, rtol=2e-12)


def test_cartesian_nonlinear_steps_decrease_objective_and_bound_actual_atom_displacement():
    base = torch.tensor([[[0., 0., 0.], [8., .1, 0.], [0., 5., .1]]], dtype=torch.float64)
    model = RigidPose(base, torch.zeros(1, 3, dtype=torch.float64), torch.zeros(1, 3, dtype=torch.float64))
    target = -base + torch.tensor([[[2., 3., .5]]], dtype=torch.float64)
    def energy(x):
        return .5 * (x - target).square().sum((1, 2))
    previous = model().detach().clone()
    value = energy(previous)
    for _ in range(15):
        result, _ = cartesian_run(model, energy, num_steps=1, max_atom_displacement=.15)
        updated = result.detach().clone()
        assert energy(updated) <= value
        assert torch.linalg.vector_norm(updated - previous, dim=2).max() <= .15000001
        previous, value = updated, energy(updated)


def test_cartesian_backtracks_grid_boundary_per_pose():
    values = -torch.arange(2, dtype=torch.float64).reshape(1, 1, 1, 2).expand(1, 2, 2, 2).clone()
    field = TrilinearGrid(values, origin=[0., 0., 0.], spacing=1.)
    def energy(x):
        return field(x, [0]).sum(dim=1)
    initial = torch.tensor([[[.9, .5, .5]], [[.1, .5, .5]]], dtype=torch.float64)
    batched, _ = cartesian_run(CartesianPose(initial), energy, num_steps=1)
    assert .9 < batched[0, 0, 0] <= 1.
    assert batched[1, 0, 0] > .3
    for row in range(2):
        single, _ = cartesian_run(CartesianPose(initial[row:row + 1]), energy, num_steps=1)
        torch.testing.assert_close(batched[row], single[0], atol=1e-14, rtol=1e-14)


def test_cartesian_rejects_nonfinite_trial_energy():
    model = CartesianPose([[[.9, .5, .5]]])

    def energy(x):
        return torch.where(x[:, 0, 0] <= 1., -x[:, 0, 0], float("nan"))

    output, _ = cartesian_run(model, energy, num_steps=1)
    assert .9 < output[0, 0, 0] <= 1.


def test_cartesian_backtracking_checks_true_objective_beyond_geometry_metric():
    model = CartesianPose([[[1., 0., 0.]]])
    calls = []

    def energy(x):
        value = 5000. * x.square().sum((1, 2))
        calls.append(float(value.detach()[0]))
        return value

    output, stats = cartesian_run(model, energy, num_steps=1, max_atom_displacement=100.)
    assert max(calls) > calls[0]  # The geometry metric is not an energy Hessian.
    assert stats.final_best_energy < stats.initial_best_energy
    assert abs(output[0, 0, 0]) < 1.


def test_cartesian_exhausted_backtracking_keeps_accepted_state():
    model = CartesianPose([[[1., 0., 0.]]])
    original = model().detach().clone()
    output, stats = cartesian_run(model, lambda x: 5000. * x.square().sum((1, 2)), num_steps=1,
                                  max_atom_displacement=100., max_backtracks=0)
    torch.testing.assert_close(output, original)
    assert stats.final_best_energy == stats.initial_best_energy


def test_cartesian_rejected_or_failed_trials_restore_live_state():
    model = CartesianPose([[[1., 0., 0.]]])
    original = model().detach().clone()

    def energy(x):
        if x[0, 0, 0] < .9:
            raise ValueError("broken scorer")
        return x.square().sum((1, 2))

    with pytest.raises(ValueError, match="broken scorer"):
        cartesian_run(model, energy, num_steps=1)
    torch.testing.assert_close(model(), original)


def test_cartesian_does_not_accumulate_scorer_weight_gradients():
    model = CartesianPose([[[1., 0., 0.]]])
    scorer_weight = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))
    cartesian_run(model, lambda x: scorer_weight * x.square().sum((1, 2)), num_steps=2)
    assert scorer_weight.grad is None
    assert model.xyz.grad is None


@pytest.mark.parametrize("keyword,value", [
    ("geometry_damping", 0.), ("geometry_damping", float("nan")),
    ("max_atom_displacement", -1.), ("max_atom_displacement", float("inf")),
    ("max_backtracks", -1), ("max_backtracks", True), ("max_backtracks", 1.5),
])
def test_cartesian_rejects_invalid_geometry_options(keyword, value):
    with pytest.raises(ValueError, match=keyword):
        cartesian_run(CartesianPose([[[1., 0., 0.]]]), lambda x: x.square().sum((1, 2)), num_steps=1, **{keyword: value})


def test_cartesian_bounds_dense_jacobian_size():
    model = CartesianPose(torch.ones(1, 100, 3))
    with pytest.raises(ValueError, match="bounded size"):
        cartesian_run(model, lambda x: x.square().sum((1, 2)), num_steps=1)


def test_cartesian_initial_grid_failure_is_not_silently_accepted():
    field = TrilinearGrid(torch.zeros(1, 2, 2, 2, dtype=torch.float64), origin=[0., 0., 0.], spacing=1.)
    with pytest.raises(ValueError, match="bounds"):
        cartesian_run(CartesianPose([[[1.1, .5, .5]]]), lambda x: field(x, [0]).sum(dim=1), num_steps=1)


@pytest.mark.parametrize("kwargs,match", [
    ({"num_steps": True}, "num_steps"), ({"num_steps": 1.5}, "num_steps"),
    ({"num_steps": -1}, "num_steps"), ({"patience": False}, "patience"),
    ({"patience": 1.5}, "patience"), ({"learning_rate": float("nan")}, "learning_rate"),
    ({"learning_rate": float("inf")}, "learning_rate"), ({"learning_rate": True}, "learning_rate"),
    ({"min_delta": float("nan")}, "min_delta"), ({"min_delta": float("inf")}, "min_delta"),
    ({"min_delta": False}, "min_delta"), ({"gradient_clip": True}, "gradient_clip"),
    ({"gradient_clip": float("nan")}, "gradient_clip"), ({"optimizer": "bogus"}, "optimizer"),
    ({"optimizer": []}, "optimizer"),
    ({"optimizer": "cartesian", "geometry_damping": float("nan")}, "geometry_damping"),
    ({"optimizer": "cartesian", "geometry_damping": True}, "geometry_damping"),
    ({"optimizer": "cartesian", "max_atom_displacement": float("inf")}, "max_atom_displacement"),
    ({"optimizer": "cartesian", "max_atom_displacement": True}, "max_atom_displacement"),
    ({"optimizer": "cartesian", "max_backtracks": True}, "max_backtracks"),
])
@pytest.mark.parametrize("has_parameters", [True, False])
def test_options_fail_before_zero_step_or_parameterless_shortcuts(kwargs, match, has_parameters):
    model = CartesianPose([[[1., 0., 0.]]])
    if not has_parameters:
        model.xyz.requires_grad_(False)
    options = dict(num_steps=0, optimizer="adam")
    options.update(kwargs)
    with pytest.raises(ValueError, match=match):
        optimize_pose_module(model, lambda x: x.square().sum((1, 2)), **options)


def test_cartesian_invalid_options_fail_before_model_evaluation():
    class UnevaluatedModel(torch.nn.Module):
        def forward(self):
            raise AssertionError("invalid options should be rejected first")

    with pytest.raises(ValueError, match="geometry_damping"):
        optimize_pose_module(UnevaluatedModel(), lambda x: x, optimizer="cartesian", num_steps=0, geometry_damping=0.)
