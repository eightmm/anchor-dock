"""SO(3), torsion and Cartesian-to-pose derivative regressions."""
from __future__ import annotations

import math

import pytest
import torch
from rdkit import Chem

from anchor_dock.core.kinematics import (
    LigandKinematics,
    get_batched_rotation_matrix,
    rotation_vector_to_matrix,
)
from anchor_dock.core.optimization import SE3PoseModel, _clip_pose_gradients_, optimize_pose_module


def _skew(vectors):
    x, y, z = vectors.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(-1, 3, 3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rotation_vector_identity_has_the_infinitesimal_rotation_jacobian(dtype):
    vector = torch.zeros((1, 3), dtype=dtype, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(rotation_vector_to_matrix, vector)[0, :, :, 0, :]
    generators = _skew(torch.eye(3, dtype=dtype)).permute(1, 2, 0)
    torch.testing.assert_close(rotation_vector_to_matrix(vector), torch.eye(3, dtype=dtype)[None], rtol=0, atol=0)
    torch.testing.assert_close(jacobian, generators, rtol=0, atol=0)


@pytest.mark.parametrize("vector", [[0., 0., 0.], [1e-12, -2e-12, 3e-12], [.2, -.3, .1]])
def test_rotation_vector_first_and_second_derivatives_pass_gradcheck(vector):
    value = torch.tensor([vector], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(rotation_vector_to_matrix, (value,), atol=1e-7, rtol=1e-5)
    assert torch.autograd.gradgradcheck(rotation_vector_to_matrix, (value,), atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rotation_vector_matches_matrix_exponential_and_preserves_so3(dtype):
    vectors = torch.tensor([
        [0., 0., 0.], [1e-8, -2e-8, 3e-8], [.003, .002, -.001],
        [.01, 0., 0.], [.2, -.3, .7], [math.pi, 0., 0.], [0., -4., 0.], [0., 0., 2*math.pi],
    ], dtype=dtype)
    actual = rotation_vector_to_matrix(vectors)
    expected = torch.linalg.matrix_exp(_skew(vectors))
    tolerance = 2e-6 if dtype == torch.float32 else 2e-14
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    identity = torch.eye(3, dtype=dtype).expand(len(vectors), -1, -1)
    torch.testing.assert_close(actual @ actual.transpose(1, 2), identity, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(torch.linalg.det(actual), torch.ones(len(vectors), dtype=dtype), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_finite_rotations_agree_with_axis_angle_and_degenerate_axis_contract_remains(dtype):
    vectors = torch.tensor([[.3, -.4, .1], [1.2, 1.1, -.2], [-2., .1, .8]], dtype=dtype)
    expected = get_batched_rotation_matrix(vectors, torch.linalg.vector_norm(vectors, dim=1))
    torch.testing.assert_close(rotation_vector_to_matrix(vectors), expected)
    # The axis-angle API deliberately treats a missing geometric axis as no rotation.
    torch.testing.assert_close(get_batched_rotation_matrix(torch.zeros(1, 3, dtype=dtype),
                                                          torch.tensor([1.2], dtype=dtype)),
                               torch.eye(3, dtype=dtype)[None], rtol=0, atol=0)


def _rigid_model():
    return SE3PoseModel(Chem.MolFromSmiles("CC"),
                        torch.tensor([[[0., 0., 0.], [1., 0., 0.]]]), 0,
                        torch.zeros(1, 3), torch.zeros(1, 3), "cpu")


def test_se3_rotation_at_identity_agrees_with_finite_difference():
    model = _rigid_model()
    derivative = torch.autograd.grad(model()[0, 1, 1], model.rotation_vectors)[0]
    torch.testing.assert_close(derivative, torch.tensor([[0., 0., 1.]]), rtol=0, atol=0)
    epsilon = 1e-3
    with torch.no_grad():
        model.rotation_vectors[0, 2] = epsilon
        plus = model()[0, 1, 1].item()
        model.rotation_vectors[0, 2] = -epsilon
        minus = model()[0, 1, 1].item()
    assert derivative[0, 2].item() == pytest.approx((plus-minus)/(2*epsilon), abs=1e-6)


def test_zero_initialized_rotation_can_optimize_when_translation_is_fixed():
    model = _rigid_model()
    model.translations.requires_grad_(False)
    target = torch.tensor([0., 1., 0.])
    final, stats = optimize_pose_module(model, lambda xyz: (xyz[:, 1] - target).square().sum(dim=1),
                                        num_steps=60, learning_rate=.05, early_stopping=False,
                                        gradient_clip=None)
    assert stats.initial_best_energy == pytest.approx(2.)
    assert stats.final_best_energy < .01
    assert final[0, 1, 1].item() > .99
    torch.testing.assert_close(final[0, 0], torch.zeros(3), rtol=0, atol=0)


def test_cartesian_gradient_projects_to_translation_and_rotation_torque():
    model = _rigid_model().double()
    cartesian_gradient = torch.tensor([[[.2, -.4, .1], [.7, 1.3, -.2]]], dtype=torch.float64)
    coordinates = model()
    translation, rotation = torch.autograd.grad((coordinates * cartesian_gradient).sum(),
                                               (model.translations, model.rotation_vectors))
    torch.testing.assert_close(translation, cartesian_gradient.sum(dim=1), rtol=0, atol=0)
    lever = coordinates.detach() - coordinates[:, :1].detach()
    expected_torque = torch.cross(lever, cartesian_gradient, dim=2).sum(dim=1)
    torch.testing.assert_close(rotation, expected_torque, rtol=0, atol=0)


def test_finite_rotation_vjp_includes_so3_left_jacobian():
    model = _rigid_model().double()
    vector = torch.tensor([[.4, -.3, .2]], dtype=torch.float64)
    with torch.no_grad():
        model.rotation_vectors.copy_(vector)
    coordinates = model()
    gradient = torch.tensor([[[.2, -.4, .1], [.7, 1.3, -.2]]], dtype=torch.float64)
    actual = torch.autograd.grad((coordinates * gradient).sum(), model.rotation_vectors)[0]
    angle = torch.linalg.vector_norm(vector, dim=1)
    skew = _skew(vector)
    left_jacobian = (torch.eye(3, dtype=vector.dtype)[None]
                     + ((1-torch.cos(angle))/angle.square())[:, None, None] * skew
                     + ((angle-torch.sin(angle))/angle.pow(3))[:, None, None] * (skew @ skew))
    lever = coordinates.detach() - model.translations.detach()[:, None]
    torque = torch.cross(lever, gradient, dim=2).sum(dim=1)
    expected = (left_jacobian.transpose(1, 2) @ torque[:, :, None]).squeeze(2)
    torch.testing.assert_close(actual, expected, atol=1e-15, rtol=1e-15)


def test_torsion_jacobian_and_cartesian_vjp_match_axis_cross_lever_formula():
    molecule = Chem.MolFromSmiles("CCCCCC")
    initial = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1.7, 1., .2],
                            [2.8, 1.1, 1.], [3.6, 2.1, 1.4], [4.8, 2.2, .6]])
    model = LigandKinematics(molecule, [0], initial[None], "cpu").double()
    theta = torch.linspace(-.31, .23, model.num_torsions, dtype=torch.float64)[None].requires_grad_()
    def forward(angles):
        return torch.func.functional_call(model, {"thetas": angles})
    assert torch.autograd.gradcheck(forward, (theta,), atol=1e-7, rtol=1e-5)
    coordinates = forward(theta)
    gradient = torch.linspace(-.8, 1.2, coordinates.numel(), dtype=torch.float64).reshape_as(coordinates)
    actual = torch.autograd.grad((coordinates * gradient).sum(), theta)[0]
    expected = torch.zeros_like(actual)
    for index, atoms in enumerate(model.atoms_to_rotate):
        origin = coordinates.detach()[:, model.parent_atoms[index]]
        axis = coordinates.detach()[:, model.child_atoms[index]] - origin
        unit = axis / torch.linalg.vector_norm(axis, dim=1, keepdim=True)
        lever = coordinates.detach()[:, atoms] - origin[:, None]
        tangent = torch.cross(unit[:, None].expand_as(lever), lever, dim=2)
        expected[:, index] = (tangent * gradient[:, atoms]).sum(dim=(1, 2))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(coordinates[:, 0], initial[None, 0].double(), rtol=0, atol=0)


def test_rotation_vector_input_contract():
    with pytest.raises(ValueError, match="shape"):
        rotation_vector_to_matrix(torch.zeros(3))
    with pytest.raises(TypeError, match="floating"):
        rotation_vector_to_matrix(torch.zeros(1, 3, dtype=torch.long))
    assert rotation_vector_to_matrix(torch.empty(0, 3)).shape == (0, 3, 3)


@pytest.mark.parametrize("optimizer", ["adam", "adamw"])
def test_per_pose_clipping_is_independent_of_a_high_gradient_neighbor(optimizer):
    class IndependentPoses(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.positions = torch.nn.Parameter(torch.tensor(values).reshape(-1, 1, 1))

        def forward(self):
            return self.positions.expand(-1, -1, 3)

    def run(values):
        final, _ = optimize_pose_module(
            IndependentPoses(values), lambda xyz: xyz[:, 0, 0].square(),
            num_steps=30, learning_rate=.05, early_stopping=False,
            optimizer=optimizer, gradient_clip=1.,
        )
        return final.detach()[0]

    torch.testing.assert_close(run([2.]), run([2., 100.]), atol=1e-7, rtol=0)


def test_per_pose_clip_norm_combines_all_parameter_blocks():
    translation = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.float64))
    torsion = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))
    unused = torch.nn.Parameter(torch.zeros(2, 1, dtype=torch.float64))
    translation.grad = torch.tensor([[3., 4., 0.], [.3, .4, 0.]], dtype=torch.float64)
    torsion.grad = torch.tensor([[0., 12.], [0., 0.]], dtype=torch.float64)
    before_translation, before_torsion = translation.grad.clone(), torsion.grad.clone()
    norms = _clip_pose_gradients_([translation, torsion, unused], 2, 5.)
    torch.testing.assert_close(norms, torch.tensor([13., .5], dtype=norms.dtype))
    scale = torch.tensor([5./(13.+1e-6), 1.], dtype=norms.dtype)[:, None]
    torch.testing.assert_close(translation.grad, before_translation * scale, atol=0, rtol=0)
    torch.testing.assert_close(torsion.grad, before_torsion * scale, atol=0, rtol=0)
    assert unused.grad is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_pose_gradient_is_rejected_before_mutation(bad):
    parameter = torch.nn.Parameter(torch.zeros(2, 1))
    parameter.grad = torch.tensor([[1.], [bad]])
    with pytest.raises(FloatingPointError, match="gradients"):
        _clip_pose_gradients_([parameter], 2, 1.)
    assert parameter.grad[0].item() == 1.


def test_finite_loss_with_infinite_gradient_is_rejected_before_optimizer_step():
    class Pose(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.position = torch.nn.Parameter(torch.zeros(1, 1, 3))

        def forward(self):
            return self.position

    model = Pose()
    with pytest.raises(FloatingPointError, match="gradients"):
        optimize_pose_module(model, lambda xyz: torch.sqrt(xyz[:, 0, 0]),
                             num_steps=1, gradient_clip=1.)
    torch.testing.assert_close(model.position, torch.zeros_like(model.position), rtol=0, atol=0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.])
def test_gradient_clip_requires_a_finite_positive_limit(bad):
    with pytest.raises(ValueError, match="finite"):
        optimize_pose_module(_rigid_model(), lambda xyz: xyz.square().sum((1, 2)),
                             gradient_clip=bad)
