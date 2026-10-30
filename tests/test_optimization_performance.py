"""Row-mask regressions for the synchronization-free accelerator path."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from anchor_dock.core.optimization import (
    _clear_adam_rows,
    _restore_rows,
    _update_best_rows,
    optimize_pose_module,
)

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("selection", ["none", "some", "all"])
def test_row_updates_preserve_unselected_values_and_parameter_gradients(device, selection):
    rows = torch.tensor([False, True, False], device=device)
    if selection == "none":
        rows.zero_()
    elif selection == "all":
        rows.fill_(True)
    # Include different ranks and a noncontiguous view, like different pose DOFs.
    parameters = [
        nn.Parameter(torch.arange(3.0, device=device)),
        nn.Parameter(torch.arange(18.0, device=device).reshape(2, 3, 3).transpose(0, 1)),
    ]
    snapshots = [torch.full_like(parameter, -7.0) for parameter in parameters]
    original = [parameter.detach().clone() for parameter in parameters]
    gradients = [torch.full_like(parameter, 13.0) for parameter in parameters]
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient.clone()

    _restore_rows(parameters, snapshots, rows)
    for parameter, saved, before, gradient in zip(parameters, snapshots, original, gradients, strict=True):
        expected = before.clone()
        expected[rows] = saved[rows]
        assert torch.equal(parameter, expected)
        assert torch.equal(parameter.grad, gradient)

    for parameter, before in zip(parameters, original, strict=True):
        with torch.no_grad():
            parameter.copy_(before)
    _update_best_rows(parameters, snapshots, rows)
    for parameter, snapshot, before in zip(parameters, snapshots, original, strict=True):
        expected = torch.full_like(before, -7.0)
        expected[rows] = before[rows]
        assert torch.equal(snapshot, expected)
        assert torch.equal(parameter, before)


@pytest.mark.parametrize("device", DEVICES)
def test_row_restore_does_not_spread_nan_from_unselected_snapshot(device):
    parameter = nn.Parameter(torch.tensor([[1.0], [2.0]], device=device))
    snapshot = torch.tensor([[float("nan")], [5.0]], device=device)
    _restore_rows([parameter], [snapshot], torch.tensor([False, True], device=device))
    assert torch.equal(parameter, torch.tensor([[1.0], [5.0]], device=device))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("optimizer_cls", [torch.optim.Adam, torch.optim.AdamW])
def test_clear_adam_rows_preserves_other_moments_and_scalar_step(device, optimizer_cls):
    parameter = nn.Parameter(torch.arange(1.0, 19.0, device=device).reshape(3, 2, 3))
    optimizer = optimizer_cls([parameter], lr=0.1, amsgrad=True)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]
    before = {name: value.clone() for name, value in state.items()}
    rows = torch.tensor([False, True, False], device=device)
    _clear_adam_rows(optimizer, [parameter], rows)
    assert torch.equal(state["step"], before["step"])
    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        expected = before[name].clone()
        expected[rows] = 0
        assert torch.equal(state[name], expected)


class CartesianPose(nn.Module):
    def __init__(self, values: torch.Tensor):
        super().__init__()
        self.xyz = nn.Parameter(values.clone())

    def forward(self):
        return self.xyz


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("optimizer", ["adam", "adamw"])
def test_partially_converged_rows_stay_at_their_best_coordinates(device, optimizer):
    initial = torch.tensor([[[0.0, 0.0, 0.0]], [[3.0, -1.0, 2.0]], [[-2.0, 2.0, 1.0]]], device=device)
    model = CartesianPose(initial)
    result, stats = optimize_pose_module(
        model,
        lambda xyz: xyz.square().sum(dim=(1, 2)),
        num_steps=30,
        learning_rate=0.1,
        optimizer=optimizer,
        patience=3,
        min_delta=1e-9,
    )
    assert torch.equal(result[0], initial[0])
    assert stats.minimum_steps == 3
    assert stats.maximum_steps == 30
    assert 3 < stats.average_steps < 30
    assert torch.all(result[1:].square().sum(dim=(1, 2)) < initial[1:].square().sum(dim=(1, 2)))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("optimizer", ["adam", "adamw", "lbfgs"])
def test_best_state_restoration_survives_overshooting_and_repeated_phases(device, optimizer):
    initial = torch.tensor([[[5.0, 0.0, 0.0]]], device=device)
    model = CartesianPose(initial)
    def energy(xyz):
        return (xyz[:, 0, 0] - 1.0).square()
    guided, guided_stats = optimize_pose_module(
        model, energy, num_steps=8, learning_rate=4.0, optimizer=optimizer, early_stopping=False
    )
    guided = guided.detach().clone()
    assert energy(guided).item() == pytest.approx(guided_stats.final_best_energy)
    released, release_stats = optimize_pose_module(
        model, energy, num_steps=5, learning_rate=0.05, optimizer=optimizer, early_stopping=False
    )
    assert energy(released).item() == pytest.approx(release_stats.final_best_energy)
    assert release_stats.initial_best_energy == pytest.approx(guided_stats.final_best_energy)
    assert release_stats.final_best_energy <= release_stats.initial_best_energy


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_row_helpers_do_not_read_mask_scalars_on_host():
    parameter = nn.Parameter(torch.ones(3, 2, device="cuda"))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    snapshot = parameter.detach().clone()
    rows = torch.tensor([False, True, False], device="cuda")
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        _restore_rows([parameter], [snapshot], rows)
        _update_best_rows([parameter], [snapshot], rows)
        _clear_adam_rows(optimizer, [parameter], rows)
    assert "aten::_local_scalar_dense" not in {event.key for event in profile.key_averages()}
