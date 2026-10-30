"""Scorer-independent differentiable and derivative-free pose optimization."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import torch
import torch.nn as nn
from rdkit import Chem

from .grid import GridBoundsError
from .kinematics import LigandKinematics, rotation_vector_to_matrix
from .scoring import PreparedScorer

OptimizerName = Literal["adam", "adamw", "lbfgs", "cartesian", "pattern"]
OPTIMIZATION_CONTRACT = "so3-exp-v2-per-pose-clip-v1-objective-consistent-v1-cartesian-pullback-v2-pattern-v1"


@dataclass(frozen=True)
class OptimizationStats:
    average_steps: float
    minimum_steps: int
    maximum_steps: int
    num_poses: int
    initial_best_energy: float
    final_best_energy: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "average_steps": self.average_steps,
            "minimum_steps": self.minimum_steps,
            "maximum_steps": self.maximum_steps,
            "num_poses": self.num_poses,
            "initial_best_energy": self.initial_best_energy,
            "final_best_energy": self.final_best_energy,
        }


def _batch_parameters(model: nn.Module, batch_size: int) -> list[nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.numel() > 0]
    for parameter in parameters:
        if parameter.ndim == 0 or parameter.shape[0] != batch_size:
            raise ValueError("all optimizable pose parameters must use the pose batch as their first dimension")
    return parameters


def _restore_rows(parameters: list[nn.Parameter], snapshots: list[torch.Tensor], rows: torch.Tensor) -> None:
    # Keep the cheap empty-mask shortcut on CPU, but never read a CUDA mask on
    # the host here: these helpers run several times per optimization step.
    if rows.device.type == "cpu" and not rows.any():
        return
    with torch.no_grad():
        for parameter, snapshot in zip(parameters, snapshots, strict=True):
            mask = rows.reshape((rows.shape[0],) + (1,) * (parameter.ndim - 1))
            parameter.copy_(torch.where(mask, snapshot, parameter))


def _update_best_rows(
    parameters: list[nn.Parameter],
    snapshots: list[torch.Tensor],
    improved: torch.Tensor,
) -> None:
    if improved.device.type == "cpu" and not improved.any():
        return
    with torch.no_grad():
        for parameter, snapshot in zip(parameters, snapshots, strict=True):
            mask = improved.reshape((improved.shape[0],) + (1,) * (parameter.ndim - 1))
            snapshot.copy_(torch.where(mask, parameter.detach(), snapshot))


def _clear_adam_rows(optimizer: torch.optim.Optimizer, parameters: list[nn.Parameter], rows: torch.Tensor) -> None:
    if rows.device.type == "cpu" and not rows.any():
        return
    for parameter in parameters:
        state = optimizer.state.get(parameter, {})
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(key)
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == rows.shape[0]:
                mask = rows.reshape((rows.shape[0],) + (1,) * (value.ndim - 1))
                value.masked_fill_(mask, 0)


def _checked_losses(values: torch.Tensor, batch_size: int) -> torch.Tensor:
    if values.ndim == 0 and batch_size == 1:
        values = values.reshape(1)
    if values.shape != (batch_size,):
        raise ValueError(f"energy function must return shape [{batch_size}], got {tuple(values.shape)}")
    if not torch.isfinite(values).all():
        raise FloatingPointError("energy function returned NaN or infinity")
    return values


def _assign_pose_gradients(loss: torch.Tensor, parameters: list[nn.Parameter]) -> None:
    """Populate only pose-parameter gradients, leaving scorer parameters untouched."""
    if not loss.requires_grad:
        raise RuntimeError("energy function is not differentiable with respect to pose parameters")
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    if not any(gradient is not None for gradient in gradients):
        raise RuntimeError("energy function is not differentiable with respect to any pose parameter")
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient


def _clip_pose_gradients_(parameters: list[nn.Parameter], batch_size: int, max_norm: float) -> torch.Tensor:
    """Clip each pose across its parameter blocks, without coupling batch rows.

    This is a numerical Euclidean norm of the chosen parameterization, not a
    physical Cartesian metric: translations and angular parameters still need
    separately chosen scales if such a metric is desired.
    """
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("gradient_clip must be finite and positive")
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("no pose gradients to clip")
    norm_device = gradients[0].device
    norms = torch.stack([
        torch.linalg.vector_norm(gradient.reshape(batch_size, -1), dim=1).to(norm_device)
        for gradient in gradients
    ])
    total_norm = torch.linalg.vector_norm(norms, dim=0)
    if not torch.isfinite(total_norm).all():
        raise FloatingPointError("pose gradients contain NaN, infinity or an overflowing norm")
    scale = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    with torch.no_grad():
        for gradient in gradients:
            mask_shape = (batch_size,) + (1,) * (gradient.ndim - 1)
            gradient.mul_(scale.to(device=gradient.device, dtype=gradient.dtype).reshape(mask_shape))
    return total_norm


def _as_pose_batch(coords: torch.Tensor) -> torch.Tensor:
    if coords.ndim == 2:
        coords = coords.unsqueeze(0)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError("pose module must return [N,3] or [B,N,3] coordinates")
    return coords


def _flat_pose_state(parameters: list[nn.Parameter]) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(parameter.shape[0], -1) for parameter in parameters], dim=1)


def _copy_flat_pose_state_(parameters: list[nn.Parameter], values: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            size = parameter[0].numel()
            parameter.copy_(values[:, offset:offset + size].reshape_as(parameter))
            offset += size


def _finite_parameter_row(parameters: list[nn.Parameter], values: torch.Tensor) -> bool:
    """Check finiteness after each block's actual parameter dtype conversion."""
    offset = 0
    for parameter in parameters:
        size = parameter[0].numel()
        if not torch.isfinite(values[offset:offset + size].to(dtype=parameter.dtype)).all():
            return False
        offset += size
    return True


def _cartesian_pose_jacobian(
    model: nn.Module,
    parameters: list[nn.Parameter],
    flat_state: torch.Tensor,
    row: int,
) -> torch.Tensor:
    """Reference per-pose Cartesian Jacobian without mutating live parameters.

    The full module is evaluated so registered buffers and arbitrary parameter
    names retain their meaning. Dense Jacobians are intentionally restricted to
    this opt-in reference optimizer; normal Adam/LBFGS paths use only VJPs.
    As with the other optimizers, batch rows must represent independent poses.
    """
    parameter_names = {id(value): name for name, value in model.named_parameters()}
    row_index = torch.tensor([row], dtype=torch.long, device=flat_state.device)

    def coordinates_at(values: torch.Tensor) -> torch.Tensor:
        all_values = flat_state.index_copy(0, row_index, values.unsqueeze(0))
        replacements = {}
        offset = 0
        for parameter in parameters:
            size = parameter[0].numel()
            replacements[parameter_names[id(parameter)]] = all_values[:, offset:offset + size].reshape_as(parameter)
            offset += size
        coords = torch.func.functional_call(model, replacements, ())
        return _as_pose_batch(coords)[row].reshape(-1)

    return torch.func.jacrev(coordinates_at)(flat_state[row]).detach()


def _pullback_direction(
    jacobian: torch.Tensor,
    cartesian_gradient: torch.Tensor,
    parameters: list[nn.Parameter],
    damping: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve a damped Cartesian metric, not an energy Hessian approximation.

    Block-isotropic damping uses each parameter tensor's mean metric diagonal.
    This keeps translation/rotation vector blocks isotropic and gives a unit-
    aware regularizer instead of mixing Angstrom and radian gradient norms.
    Zero-motion columns are regularized and receive a zero step. Solving in
    FP64 makes redundant/frozen torsions reliable even for FP32 pose modules.
    """
    matrix = jacobian.to(dtype=torch.float64)
    gradient = matrix.T @ cartesian_gradient.reshape(-1).to(dtype=torch.float64)
    if not torch.isfinite(matrix).all() or not torch.isfinite(gradient).all():
        raise FloatingPointError("Cartesian Jacobian or energy gradient contains NaN or infinity")
    metric = matrix.T @ matrix
    diagonal = metric.diagonal()
    scale_floor = diagonal.max().clamp_min(1.0) * torch.finfo(torch.float64).eps
    regularizer = torch.empty_like(diagonal)
    offset = 0
    for parameter in parameters:
        size = parameter[0].numel()
        regularizer[offset:offset + size] = diagonal[offset:offset + size].mean().clamp_min(scale_floor)
        offset += size
    regularized = metric + damping * torch.diag(regularizer)
    direction = torch.linalg.solve(regularized, -gradient)
    if not torch.isfinite(direction).all():
        raise FloatingPointError("damped Cartesian metric solve returned a nonfinite direction")
    return direction.to(dtype=jacobian.dtype), torch.dot(gradient, direction)


def _optimize_cartesian_pullback(
    model: nn.Module,
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    parameters: list[nn.Parameter],
    initial_losses: torch.Tensor,
    *,
    num_steps: int,
    learning_rate: float,
    early_stopping: bool,
    patience: int,
    min_delta: float,
    damping: float,
    max_atom_displacement: float,
    max_backtracks: int,
) -> tuple[torch.Tensor, OptimizationStats]:
    """Geometry-aware reference optimizer with exact objective acceptance.

    Every trial changes just one pose, allowing a strict map-domain exception
    to reject that pose without shrinking another pose's step. Nonfinite trial
    energies are also rejected. Programming errors propagate after restoring
    the last accepted state. The actual, nonlinear atom displacement is checked
    in addition to the linearized bound. Raw rotation-vector coordinates remain
    the existing chart; this is not a Lie-retraction optimizer or a fused GPU
    kernel.
    """
    batch_size = initial_losses.shape[0]
    best_losses = initial_losses.clone()
    best_parameters = [parameter.detach().clone() for parameter in parameters]
    stale = torch.zeros(batch_size, dtype=torch.long, device=initial_losses.device)
    converged = torch.zeros(batch_size, dtype=torch.bool, device=initial_losses.device)
    step_counts = torch.zeros(batch_size, dtype=torch.long, device=initial_losses.device)

    for _ in range(num_steps):
        active = ~converged if early_stopping else torch.ones_like(converged)
        if not active.any():
            break
        coords = _as_pose_batch(model())
        losses = _checked_losses(energy_fn(coords), batch_size)
        if not losses.requires_grad:
            raise RuntimeError("energy function is not differentiable with respect to coordinates")
        cartesian_gradient = torch.autograd.grad(losses[active].sum(), coords, allow_unused=True)[0]
        if cartesian_gradient is None:
            raise RuntimeError("energy function is not differentiable with respect to coordinates")
        if not torch.isfinite(cartesian_gradient).all():
            raise FloatingPointError("Cartesian energy gradient contains NaN or infinity")
        baseline_coords = coords.detach().clone()
        baseline_losses = losses.detach().clone()
        flat_state = _flat_pose_state(parameters)
        new_losses = losses.detach().clone()

        for row in range(batch_size):
            if not bool(active[row]):
                continue
            jacobian = _cartesian_pose_jacobian(model, parameters, flat_state, row)
            direction, directional_derivative = _pullback_direction(
                jacobian, cartesian_gradient[row], parameters, damping,
            )
            if not bool(directional_derivative < 0):
                continue
            predicted = (jacobian @ direction).reshape(-1, 3)
            largest_displacement = torch.linalg.vector_norm(predicted, dim=1).max()
            alpha = min(learning_rate, float(max_atom_displacement / largest_displacement.clamp_min(1e-30)))
            accepted = False
            try:
                for _attempt in range(max_backtracks + 1):
                    trial_state = flat_state.clone()
                    trial_state[row] = flat_state[row] + alpha * direction
                    _copy_flat_pose_state_(parameters, trial_state)
                    with torch.no_grad():
                        trial_coords = _as_pose_batch(model())
                        displacement = torch.linalg.vector_norm(trial_coords[row] - baseline_coords[row], dim=1).max()
                        if (not torch.isfinite(trial_coords[row]).all()
                                or bool(displacement > max_atom_displacement * (1.0 + 1e-6))):
                            alpha *= 0.5
                            continue
                        try:
                            trial_losses = energy_fn(trial_coords)
                        except (GridBoundsError, FloatingPointError):
                            alpha *= 0.5
                            continue
                        if trial_losses.ndim == 0 and batch_size == 1:
                            trial_losses = trial_losses.reshape(1)
                        if trial_losses.shape != (batch_size,):
                            raise ValueError(f"energy function must return shape [{batch_size}]")
                        value = trial_losses[row]
                        threshold = baseline_losses[row] + 1e-4 * alpha * directional_derivative
                        if bool(torch.isfinite(value) & (value <= threshold)):
                            flat_state = trial_state
                            new_losses[row] = value
                            accepted = True
                            break
                    alpha *= 0.5
            finally:
                # flat_state changes only after acceptance, including when a
                # scorer programming error interrupts a trial.
                _copy_flat_pose_state_(parameters, flat_state)
            if not accepted:
                new_losses[row] = baseline_losses[row]

        step_counts.add_(active)
        improved = active & (new_losses < best_losses - min_delta)
        _update_best_rows(parameters, best_parameters, improved)
        best_losses = torch.where(improved, new_losses, best_losses)
        stale = torch.where(improved, torch.zeros_like(stale), torch.where(active, stale + 1, stale))
        if early_stopping:
            converged |= stale >= patience
            _restore_rows(parameters, best_parameters, converged)

    _restore_rows(parameters, best_parameters, torch.ones_like(converged))
    return model(), OptimizationStats(
        average_steps=float(step_counts.float().mean().cpu()),
        minimum_steps=int(step_counts.min().cpu()),
        maximum_steps=int(step_counts.max().cpu()),
        num_poses=batch_size,
        initial_best_energy=float(initial_losses.min().cpu()),
        final_best_energy=float(best_losses.min().cpu()),
    )


def _optimize_pattern(
    model: nn.Module,
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    parameters: list[nn.Parameter],
    initial_losses: torch.Tensor,
    *,
    num_steps: int,
    early_stopping: bool,
    patience: int,
    min_delta: float,
    initial_step: float,
    shrink: float,
    min_step: float,
    max_evaluations: int,
    max_atom_displacement: float,
    max_backtracks: int,
) -> tuple[torch.Tensor, OptimizationStats]:
    """Deterministic coordinate polling, without derivatives of any kind.

    A poll evaluates both signs of every parameter-chart coordinate from one
    accepted state and accepts the best strict decrease. The chart step mixes
    translation/angular units, so every trial also checks its *actual* maximum
    Cartesian atom displacement. Invalid-domain/nonfinite trials backtrack;
    unexpected scorer exceptions restore the last accepted state and propagate.

    ``max_evaluations`` counts initial and trial objective calls per pose. A
    batched objective is evaluated with unchanged neighboring poses, so the
    high-level engine uses single-pose chunks to avoid redundant scoring.
    Rows of both the pose model and objective must be independent, as in the
    other optimizers. Dense polling can be costly and gives no global-search
    or smooth/quantized-objective convergence guarantee.
    """
    batch_size = initial_losses.shape[0]
    # A black-box objective may return a detached view of a live parameter.
    # Snapshot it before trials mutate that parameter's storage.
    initial_values = initial_losses.clone()
    # Score-only wrappers may initially return integer/low-precision values.
    # Keep accepted values independently in FP64 rather than truncating a
    # later floating-point improvement into the initial tensor's dtype.
    losses = initial_values.to(dtype=torch.float64).clone()
    flat_state = _flat_pose_state(parameters)
    if not torch.isfinite(flat_state).all():
        raise ValueError("pattern initial parameters must be finite")
    steps = [initial_step] * batch_size
    evaluations = [1] * batch_size
    stale = [0] * batch_size
    converged = [max_evaluations == 1] * batch_size
    step_counts = [0] * batch_size

    with torch.no_grad():
        for _ in range(num_steps):
            if all(converged):
                break
            for row in range(batch_size):
                if converged[row]:
                    continue
                baseline_coords = _as_pose_batch(model()).detach().clone()
                baseline_value = float(losses[row])
                best_value = baseline_value
                best_state = flat_state
                try:
                    for column in range(flat_state.shape[1]):
                        for sign in (1.0, -1.0):
                            if evaluations[row] >= max_evaluations:
                                break
                            trial_step = steps[row]
                            for _attempt in range(max_backtracks + 1):
                                trial_state = flat_state.clone()
                                trial_state[row, column] += sign * trial_step
                                if not _finite_parameter_row(parameters, trial_state[row]):
                                    trial_step *= shrink
                                    continue
                                _copy_flat_pose_state_(parameters, trial_state)
                                trial_coords = _as_pose_batch(model()).detach()
                                delta = trial_coords[row] - baseline_coords[row]
                                displacement = torch.linalg.vector_norm(delta, dim=1).max()
                                if (not torch.isfinite(trial_coords[row]).all()
                                        or bool(displacement > max_atom_displacement * (1.0 + 1e-6))):
                                    trial_step *= shrink
                                    continue
                                if torch.equal(trial_coords[row], baseline_coords[row]):
                                    # Frozen/redundant chart direction: do not
                                    # spend a scorer call on identical atoms.
                                    break
                                evaluations[row] += 1
                                try:
                                    trial_losses = _checked_losses(energy_fn(trial_coords), batch_size)
                                except (GridBoundsError, FloatingPointError):
                                    trial_step *= shrink
                                    if evaluations[row] >= max_evaluations:
                                        break
                                    continue
                                value = float(trial_losses[row])
                                if value < best_value:
                                    best_value, best_state = value, trial_state
                                # Backtracking enforces the domain/displacement
                                # bound, not a fabricated gradient line search.
                                break
                        if evaluations[row] >= max_evaluations:
                            break
                finally:
                    # best_state is a candidate until the whole poll succeeds.
                    # A programming error must not leave any trial installed.
                    _copy_flat_pose_state_(parameters, flat_state)
                flat_state = best_state
                _copy_flat_pose_state_(parameters, flat_state)
                losses[row] = best_value
                step_counts[row] += 1
                if best_value < baseline_value:
                    stale[row] = 0 if best_value < baseline_value - min_delta else stale[row] + 1
                else:
                    steps[row] *= shrink
                    stale[row] += 1
                converged[row] = (
                    steps[row] < min_step
                    or evaluations[row] >= max_evaluations
                    or (early_stopping and stale[row] >= patience)
                )
    return model(), OptimizationStats(
        average_steps=sum(step_counts) / batch_size,
        minimum_steps=min(step_counts),
        maximum_steps=max(step_counts),
        num_poses=batch_size,
        initial_best_energy=float(initial_values.min().cpu()),
        final_best_energy=float(losses.min().cpu()),
    )


def optimize_pose_module(
    model: nn.Module,
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    num_steps: int = 100,
    learning_rate: float = 0.05,
    optimizer: OptimizerName = "adam",
    early_stopping: bool = True,
    patience: int = 30,
    min_delta: float = 1e-5,
    gradient_clip: float | None = 1.0,
    geometry_damping: float = 1e-3,
    max_atom_displacement: float = 0.25,
    max_backtracks: int = 12,
    pattern_step: float | None = None,
    pattern_shrink: float = 0.5,
    pattern_min_step: float = 1e-4,
    pattern_max_evaluations: int = 10_000,
) -> tuple[torch.Tensor, OptimizationStats]:
    """Optimize a pose module and restore the best state for every pose.

    ``cartesian`` is an opt-in dense Cartesian pullback reference optimizer.
    Its damping, actual atom displacement limit (Angstrom) and backtracking
    affect only that mode. ``learning_rate=1`` is a useful initial trial scale;
    objective acceptance and the displacement limit still bound every step.
    It does not use numerical parameter-gradient clipping.

    ``pattern`` polls parameter-chart coordinates and requires no gradient.
    Its initial chart step is ``pattern_step`` or ``learning_rate`` if None;
    failed polls shrink it until ``pattern_min_step``. Per-pose objective calls
    include the initial evaluation and are bounded by
    ``pattern_max_evaluations``. Actual atom-displacement/backtracking options
    also apply. This is a bounded local black-box search, not finite-difference
    gradient descent or an emulation of a native docking search.
    """
    # Validate options before inspecting a module or taking an otherwise valid
    # zero-step/no-parameter shortcut. Invalid configurations must not look
    # successful merely because no optimization happened on that input.
    for name, value, minimum in (("num_steps", num_steps, 0), ("patience", patience, 1)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name, value, positive in (("learning_rate", learning_rate, True), ("min_delta", min_delta, False)):
        if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
                or (value <= 0 if positive else value < 0)):
            raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    if not isinstance(optimizer, str) or optimizer not in {"adam", "adamw", "lbfgs", "cartesian", "pattern"}:
        raise ValueError(f"unknown optimizer: {optimizer!r}")
    if gradient_clip is not None and (isinstance(gradient_clip, bool) or not isinstance(gradient_clip, Real)
                                      or not math.isfinite(gradient_clip) or gradient_clip <= 0):
        raise ValueError("gradient_clip must be finite and positive or None")
    if optimizer in {"cartesian", "pattern"}:
        settings = [("max_atom_displacement", max_atom_displacement)]
        if optimizer == "cartesian":
            settings.append(("geometry_damping", geometry_damping))
        for name, value in settings:
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(max_backtracks, bool) or not isinstance(max_backtracks, Integral) or max_backtracks < 0:
            raise ValueError("max_backtracks must be a non-negative integer")
    if optimizer == "pattern":
        pattern_step = learning_rate if pattern_step is None else pattern_step
        for name, value in (("pattern_step", pattern_step), ("pattern_min_step", pattern_min_step)):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (isinstance(pattern_shrink, bool) or not isinstance(pattern_shrink, Real)
                or not math.isfinite(pattern_shrink) or not 0 < pattern_shrink < 1):
            raise ValueError("pattern_shrink must be finite and strictly between zero and one")
        if pattern_min_step > pattern_step:
            raise ValueError("pattern_min_step must not exceed the initial pattern_step/learning_rate")
        if (isinstance(pattern_max_evaluations, bool) or not isinstance(pattern_max_evaluations, Integral)
                or pattern_max_evaluations < 1):
            raise ValueError("pattern_max_evaluations must be a positive integer")
    initial_coords = model()
    if initial_coords.ndim == 2:
        initial_coords = initial_coords.unsqueeze(0)
    if optimizer == "pattern":
        initial_coords = _as_pose_batch(initial_coords).detach()
        if min(initial_coords.shape[:2]) < 1 or not torch.isfinite(initial_coords).all():
            raise ValueError("pattern initial coordinates must be nonempty and finite")
    batch_size = initial_coords.shape[0]
    parameters = _batch_parameters(model, batch_size)

    with torch.no_grad():
        initial_losses = _checked_losses(energy_fn(initial_coords), batch_size).detach()
    if not parameters or num_steps == 0:
        stats = OptimizationStats(
            0.0,
            0,
            0,
            batch_size,
            float(initial_losses.min().cpu()),
            float(initial_losses.min().cpu()),
        )
        return model(), stats

    if optimizer == "pattern":
        degrees_of_freedom = sum(parameter[0].numel() for parameter in parameters)
        if degrees_of_freedom > 256:
            raise ValueError("pattern search is bounded to 256 parameter-chart coordinates per pose")
        return _optimize_pattern(
            model, energy_fn, parameters, initial_losses,
            num_steps=num_steps, early_stopping=early_stopping, patience=patience, min_delta=min_delta,
            initial_step=float(pattern_step), shrink=float(pattern_shrink), min_step=float(pattern_min_step),
            max_evaluations=int(pattern_max_evaluations), max_atom_displacement=max_atom_displacement,
            max_backtracks=int(max_backtracks),
        )

    if optimizer == "cartesian":
        degrees_of_freedom = sum(parameter[0].numel() for parameter in parameters)
        if degrees_of_freedom > 256 or initial_coords.numel() * degrees_of_freedom > 2_000_000:
            raise ValueError(
                "cartesian reference Jacobian exceeds its bounded size (256 parameters per pose or "
                "2,000,000 batched Cartesian-Jacobian elements); reduce batch size or use adam/lbfgs"
            )
        return _optimize_cartesian_pullback(
            model, energy_fn, parameters, initial_losses,
            num_steps=num_steps,
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
            damping=geometry_damping,
            max_atom_displacement=max_atom_displacement,
            max_backtracks=max_backtracks,
        )

    if optimizer == "lbfgs" and batch_size != 1:
        raise ValueError("LBFGS optimization is performed one pose at a time")

    best_losses = initial_losses.clone()
    best_parameters = [parameter.detach().clone() for parameter in parameters]
    stale = torch.zeros(batch_size, dtype=torch.long, device=initial_losses.device)
    converged = torch.zeros(batch_size, dtype=torch.bool, device=initial_losses.device)
    step_counts = torch.zeros(batch_size, dtype=torch.long, device=initial_losses.device)

    if optimizer == "adam":
        torch_optimizer: torch.optim.Optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    elif optimizer == "adamw":
        # Pose coordinates/angles have no physical preference for chart zero.
        # A nonzero decay would add an origin-dependent force not in energy_fn.
        torch_optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    elif optimizer == "lbfgs":
        torch_optimizer = torch.optim.LBFGS(
            parameters,
            lr=learning_rate,
            max_iter=1,
            max_eval=5,
            history_size=10,
            line_search_fn="strong_wolfe",
        )
    else:
        raise ValueError(f"unknown optimizer: {optimizer}")

    for _ in range(num_steps):
        active = ~converged if early_stopping else torch.ones_like(converged)
        if not active.any():
            break

        if optimizer == "lbfgs":

            def closure() -> torch.Tensor:
                torch_optimizer.zero_grad(set_to_none=True)
                losses = _checked_losses(energy_fn(model()), batch_size)
                loss = losses[active].sum()
                _assign_pose_gradients(loss, parameters)
                # Strong-Wolfe line search needs the actual derivative of the
                # returned objective. Gradient clipping is an Adam-only guard.
                return loss

            torch_optimizer.step(closure)
        else:
            torch_optimizer.zero_grad(set_to_none=True)
            losses = _checked_losses(energy_fn(model()), batch_size)
            loss = losses[active].sum()
            _assign_pose_gradients(loss, parameters)
            for parameter in parameters:
                if parameter.grad is not None:
                    mask = (~active).reshape((batch_size,) + (1,) * (parameter.ndim - 1))
                    parameter.grad.masked_fill_(mask, 0)
            if gradient_clip is not None:
                _clip_pose_gradients_(parameters, batch_size, gradient_clip)
            torch_optimizer.step()
            _restore_rows(parameters, best_parameters, ~active)
            _clear_adam_rows(torch_optimizer, parameters, ~active)

        step_counts.add_(active)
        with torch.no_grad():
            new_losses = _checked_losses(energy_fn(model()), batch_size).detach()
        improved = active & (new_losses < best_losses - min_delta)
        _update_best_rows(parameters, best_parameters, improved)
        best_losses = torch.where(improved, new_losses, best_losses)
        stale = torch.where(improved, torch.zeros_like(stale), torch.where(active, stale + 1, stale))
        if early_stopping:
            converged |= stale >= patience

    _restore_rows(parameters, best_parameters, torch.ones_like(converged))
    final_coords = model()
    stats = OptimizationStats(
        average_steps=float(step_counts.float().mean().cpu()),
        minimum_steps=int(step_counts.min().cpu()),
        maximum_steps=int(step_counts.max().cpu()),
        num_poses=batch_size,
        initial_best_energy=float(initial_losses.min().cpu()),
        final_best_energy=float(best_losses.min().cpu()),
    )
    return final_coords, stats


def optimize_torsions(
    mol: Chem.Mol,
    anchor_indices: Iterable[int],
    init_coords: torch.Tensor,
    scorer: PreparedScorer,
    device: torch.device | str,
    *,
    num_steps: int = 100,
    learning_rate: float = 0.05,
    batch_size: int = 128,
    optimizer: OptimizerName = "adam",
    early_stopping: bool = True,
    patience: int = 30,
    min_delta: float = 1e-5,
    freeze_anchor: bool = True,
) -> tuple[torch.Tensor, OptimizationStats]:
    """Optimize torsions for one molecule topology in memory-bounded chunks."""
    device = torch.device(device)
    coords = init_coords.to(device=device, dtype=torch.float32)
    single_pose = coords.ndim == 2
    if single_pose:
        coords = coords.unsqueeze(0)
    if coords.ndim != 3:
        raise ValueError("init_coords must have shape [N,3] or [B,N,3]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    output = torch.empty_like(coords)
    all_steps: list[float] = []
    minimum_steps: list[int] = []
    maximum_steps: list[int] = []
    initial_best: list[float] = []
    final_best: list[float] = []
    for start in range(0, coords.shape[0], batch_size):
        stop = min(start + batch_size, coords.shape[0])
        chunk = coords[start:stop]
        if optimizer in {"lbfgs", "cartesian", "pattern"}:
            chunk_outputs: list[torch.Tensor] = []
            chunk_stats: list[OptimizationStats] = []
            for pose in chunk:
                model = LigandKinematics(mol, anchor_indices, pose, device, freeze_anchor=freeze_anchor)
                optimized, stats = optimize_pose_module(
                    model,
                    lambda values: scorer.search_energy(values),
                    num_steps=num_steps,
                    learning_rate=learning_rate,
                    optimizer=optimizer,
                    early_stopping=early_stopping,
                    patience=patience,
                    min_delta=min_delta,
                )
                chunk_outputs.append(optimized)
                chunk_stats.append(stats)
            output[start:stop] = torch.stack(chunk_outputs)
            stats_for_chunk = chunk_stats
        else:
            model = LigandKinematics(mol, anchor_indices, chunk, device, freeze_anchor=freeze_anchor)
            optimized, stats = optimize_pose_module(
                model,
                scorer.search_energy,
                num_steps=num_steps,
                learning_rate=learning_rate,
                optimizer=optimizer,
                early_stopping=early_stopping,
                patience=patience,
                min_delta=min_delta,
            )
            output[start:stop] = optimized
            stats_for_chunk = [stats]

        for stats in stats_for_chunk:
            all_steps.extend([stats.average_steps] * stats.num_poses)
            minimum_steps.append(stats.minimum_steps)
            maximum_steps.append(stats.maximum_steps)
            initial_best.append(stats.initial_best_energy)
            final_best.append(stats.final_best_energy)

    aggregate = OptimizationStats(
        average_steps=float(sum(all_steps) / len(all_steps)) if all_steps else 0.0,
        minimum_steps=min(minimum_steps, default=0),
        maximum_steps=max(maximum_steps, default=0),
        num_poses=coords.shape[0],
        initial_best_energy=min(initial_best, default=0.0),
        final_best_energy=min(final_best, default=0.0),
    )
    result = output[0] if single_pose else output
    return result, aggregate


class SE3PoseModel(nn.Module):
    """Torsion + rigid SE(3) parameters that keep one ligand pivot atom fixed at a per-row center.

    The pivot atom is frozen by the kinematics, so torsions and the global
    rotation only ever reposition atoms around it; only the per-row
    translation can move the pivot itself.
    """

    def __init__(
        self,
        mol: Chem.Mol,
        base_coords: torch.Tensor,
        pivot_atom_index: int,
        initial_centers: torch.Tensor,
        initial_rotation_vectors: torch.Tensor,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        device = torch.device(device)
        coords = base_coords.to(device=device, dtype=torch.float32)
        if coords.ndim != 3:
            raise ValueError("base_coords must have shape [B,N,3]")
        if not torch.isfinite(coords).all():
            raise ValueError("base_coords must be finite")
        batch_size, num_atoms, _ = coords.shape
        if not (0 <= int(pivot_atom_index) < num_atoms):
            raise ValueError("pivot_atom_index must index an atom in base_coords")

        centers = initial_centers.to(device=device, dtype=torch.float32)
        rotation_vectors = initial_rotation_vectors.to(device=device, dtype=torch.float32)
        if centers.shape != (batch_size, 3):
            raise ValueError("initial_centers must have shape [B,3]")
        if rotation_vectors.shape != (batch_size, 3):
            raise ValueError("initial_rotation_vectors must have shape [B,3]")
        if not torch.isfinite(centers).all():
            raise ValueError("initial_centers must be finite")
        if not torch.isfinite(rotation_vectors).all():
            raise ValueError("initial_rotation_vectors must be finite")

        self.pivot_atom_index = int(pivot_atom_index)
        self.kinematics = LigandKinematics(mol, (self.pivot_atom_index,), coords, device, freeze_anchor=True)
        self.translations = nn.Parameter(centers.clone())
        self.rotation_vectors = nn.Parameter(rotation_vectors.clone())

    def forward(self) -> torch.Tensor:
        flexible = self.kinematics()
        pivot = flexible[:, self.pivot_atom_index, :]
        centered = flexible - pivot[:, None, :]
        rotation = rotation_vector_to_matrix(self.rotation_vectors)
        rotated = torch.matmul(centered, rotation.transpose(1, 2))
        return rotated + self.translations[:, None, :]

