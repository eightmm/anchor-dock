"""Physical-coordinate regression for an experimental map-only component."""

from __future__ import annotations

import pytest
import torch

from anchor_dock.core.grid import AD4IntermolecularGrid, TrilinearGrid

ORIGIN = (-1.3, 2.1, -0.7)
SPACING = (0.375, 0.5, 0.8)


def _maps(*functions, dtype=torch.float64, origin=ORIGIN, spacing=SPACING):
    x = origin[0] + spacing[0] * torch.arange(6, dtype=dtype)
    y = origin[1] + spacing[1] * torch.arange(5, dtype=dtype)
    z = origin[2] + spacing[2] * torch.arange(4, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack([function(xx, yy, zz) for function in functions])


def _coordinates(dtype=torch.float64):
    position = torch.tensor([
        [[0.3, 0.7, 0.6], [1.3, 2.2, 1.4], [4.3, 3.2, 2.6]],
        [[1.2, 1.7, 2.3], [2.4, 0.3, 1.2], [3.1, 2.3, 0.4]],
    ], dtype=dtype)
    return position * torch.tensor(SPACING, dtype=dtype) + torch.tensor(ORIGIN, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batched_channels", [False, True])
def test_linear_fields_preserve_physical_units_axis_order_and_batched_channels(dtype, batched_channels):
    maps = _maps(lambda x, y, z: 2 * x - 3 * y + 5 * z + 7,
                 lambda x, y, z: -4 * x + 0.5 * y + z - 2, dtype=dtype)
    field = TrilinearGrid(maps, origin=ORIGIN, spacing=SPACING)
    xyz = _coordinates(dtype).requires_grad_()
    channels = torch.tensor([[0, 1, 0], [1, 0, 1]]) if batched_channels else torch.tensor([0, 1, 0])
    actual = field(xyz, channels)
    selected = channels if batched_channels else channels.expand(2, -1)
    coefficients = torch.tensor([[2., -3., 5.], [-4., .5, 1.]], dtype=dtype)[selected]
    intercept = torch.tensor([7., -2.], dtype=dtype)[selected]
    expected = (xyz * coefficients).sum(-1) + intercept
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    gradient = torch.autograd.grad(actual.sum(), xyz)[0]
    torch.testing.assert_close(gradient, coefficients, rtol=tolerance, atol=tolerance)


def test_multilinear_cross_field_values_and_gradients_are_exact():
    field = TrilinearGrid(_maps(lambda x, y, z: x * y * z + 2 * x * y - z), origin=ORIGIN, spacing=SPACING)
    xyz = _coordinates().requires_grad_()
    x, y, z = xyz.unbind(-1)
    actual = field(xyz, [0, 0, 0])
    expected = x * y * z + 2 * x * y - z
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    gradient = torch.autograd.grad(actual.sum(), xyz)[0]
    expected_gradient = torch.stack((y * z + 2 * y, x * z + 2 * x, x * y - 1), dim=-1)
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-12, atol=1e-12)


def test_fp64_gradcheck_away_from_cell_faces():
    field = TrilinearGrid(_maps(lambda x, y, z: x * y * z + 2 * x * y - z), origin=ORIGIN, spacing=SPACING)
    xyz = _coordinates().requires_grad_()
    assert torch.autograd.gradcheck(lambda values: field(values, [0, 0, 0]), (xyz,), eps=1e-6, atol=1e-7)


def test_outer_nodes_are_allowed_and_use_the_adjacent_cell_derivative():
    maps = _maps(lambda x, y, z: x + 2 * y + 3 * z)
    field = TrilinearGrid(maps, origin=ORIGIN, spacing=SPACING)
    origin = torch.tensor(ORIGIN, dtype=torch.float64)
    upper = field.origin + field.spacing * origin.new_tensor([5, 4, 3])
    xyz = torch.stack((origin, upper)).unsqueeze(0).requires_grad_()
    value = field(xyz, [0, 0])
    torch.testing.assert_close(value, xyz[..., 0] + 2 * xyz[..., 1] + 3 * xyz[..., 2], rtol=0, atol=1e-14)
    torch.testing.assert_close(torch.autograd.grad(value.sum(), xyz)[0], xyz.new_tensor([1, 2, 3]).expand_as(xyz),
                               rtol=0, atol=1e-13)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("upper", [False, True])
def test_out_of_grid_points_fail_instead_of_returning_flat_padding_energy(axis, upper):
    field = TrilinearGrid(_maps(lambda x, y, z: x + y + z), origin=ORIGIN, spacing=SPACING)
    xyz = _coordinates()[:1, :1].clone()
    limit = field.origin + field.spacing * xyz.new_tensor([5, 4, 3]) if upper else field.origin
    xyz[0, 0, axis] = limit[axis] + (1e-7 if upper else -1e-7)
    with pytest.raises(ValueError, match="outside"):
        field(xyz, [0])


@pytest.mark.parametrize("channels", [[-1, 0, 0], [0, 1, 0], [0., 0., 0.], [True, False, False], [[0, 0]]])
def test_invalid_channel_indices_and_shapes_are_rejected(channels):
    field = TrilinearGrid(_maps(lambda x, y, z: x + y + z), origin=ORIGIN, spacing=SPACING)
    with pytest.raises(ValueError):
        field(_coordinates(), channels)


@pytest.mark.parametrize("kind", ["dtype", "nan", "shape", "empty"])
def test_invalid_coordinate_contract_is_rejected(kind):
    field = TrilinearGrid(_maps(lambda x, y, z: x + y + z), origin=ORIGIN, spacing=SPACING)
    xyz = _coordinates()
    if kind == "dtype":
        xyz = xyz.float()
    elif kind == "nan":
        xyz[0, 0, 0] = float("nan")
    elif kind == "shape":
        xyz = xyz[0]
    else:
        xyz = xyz[:, :0]
    with pytest.raises(ValueError):
        field(xyz, [0, 0, 0])


@pytest.mark.parametrize("bad_spacing", [0, -1, [1, -1, 1], [1, float("inf"), 1], [1, 1]])
def test_invalid_spacing_is_rejected(bad_spacing):
    with pytest.raises(ValueError):
        TrilinearGrid(torch.zeros(1, 2, 2, 2), origin=[0, 0, 0], spacing=bad_spacing)


def test_fixed_map_buffers_are_independent_snapshots_and_can_be_cast():
    maps = _maps(lambda x, y, z: x + y + z).requires_grad_()
    field = TrilinearGrid(maps, origin=ORIGIN, spacing=SPACING)
    xyz = _coordinates().requires_grad_()
    before = field(xyz, [0, 0, 0])
    with torch.no_grad():
        maps.add_(100)
    torch.testing.assert_close(field(xyz, [0, 0, 0]), before, rtol=0, atol=0)
    assert not field.values.requires_grad
    assert not list(field.parameters())
    torch.autograd.grad(before.sum(), xyz)
    assert maps.grad is None
    torch.testing.assert_close(field.float()(xyz.float(), [0, 0, 0]), before.float(), rtol=1e-5, atol=1e-5)


def test_ad4_grid_component_has_correct_signed_charge_terms_and_gradients():
    maps = _maps(lambda x, y, z: x + 2 * y + 3 * z,
                 lambda x, y, z: 4 * x - y + .5 * z,
                 lambda x, y, z: -.4 * x + .2 * y - .1 * z,
                 lambda x, y, z: .3 * x + .5 * y + .7 * z)
    component = AD4IntermolecularGrid(maps[:2], maps[2], maps[3], atom_types=["C", "OA"],
                                    origin=ORIGIN, spacing=SPACING)
    xyz = _coordinates().requires_grad_()
    charges = torch.tensor([.4, -.7, .2], dtype=xyz.dtype, requires_grad=True)
    channels = torch.tensor([[0, 1, 0], [1, 0, 1]])
    atom_energy = component.per_atom(xyz, channels, charges)
    coefficients = xyz.new_tensor([[1, 2, 3], [4, -1, .5]])[channels]
    electro_coefficients = xyz.new_tensor([-.4, .2, -.1])
    desolv_coefficients = xyz.new_tensor([.3, .5, .7])
    electro = (xyz * electro_coefficients).sum(-1)
    desolv = (xyz * desolv_coefficients).sum(-1)
    expected = (xyz * coefficients).sum(-1) + charges * electro + charges.abs() * desolv
    torch.testing.assert_close(atom_energy, expected, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(component(xyz, channels, charges), expected.sum(-1), rtol=1e-13, atol=1e-13)
    actual_xyz, actual_charge = torch.autograd.grad(atom_energy.sum(), (xyz, charges))
    expected_xyz = coefficients + charges.detach()[None, :, None] * electro_coefficients
    expected_xyz = expected_xyz + charges.detach().abs()[None, :, None] * desolv_coefficients
    torch.testing.assert_close(actual_xyz, expected_xyz, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_charge, (electro + charges.detach().sign() * desolv).sum(0), rtol=1e-12, atol=1e-12)


def test_ad4_grid_coordinate_and_charge_gradcheck():
    maps = _maps(lambda x, y, z: x * y * z, lambda x, y, z: x * y, lambda x, y, z: y * z)
    component = AD4IntermolecularGrid(maps[:1], maps[1], maps[2], atom_types=["C"], origin=ORIGIN, spacing=SPACING)
    xyz = _coordinates().requires_grad_()
    charges = torch.tensor([[.3, -.4, .5], [-.1, .2, -.7]], dtype=xyz.dtype, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, q: component(x, [0, 0, 0], q), (xyz, charges), eps=1e-6, atol=1e-7)


def test_ad4_grid_rejects_bad_atom_map_charge_and_map_geometry():
    maps = torch.zeros(2, 3, 4, 5, dtype=torch.float64)
    component = AD4IntermolecularGrid(maps, maps[0], maps[0], atom_types=["C", "OA"], origin=[0, 0, 0], spacing=.375)
    xyz = torch.full((1, 2, 3), .1, dtype=torch.float64)
    with pytest.raises(ValueError, match="outside the supplied map set"):
        component(xyz, [0, 2], torch.zeros(2, dtype=xyz.dtype))
    with pytest.raises(ValueError, match="charges must have shape"):
        component(xyz, [0, 1], torch.zeros(3, dtype=xyz.dtype))
    with pytest.raises(ValueError, match="finite"):
        component(xyz, [0, 1], torch.tensor([0., float("nan")], dtype=xyz.dtype))
    with pytest.raises(ValueError, match="same.*shape"):
        AD4IntermolecularGrid(maps, maps[0, :, :, :2], maps[0], atom_types=["C", "OA"], origin=[0, 0, 0], spacing=.375)
    with pytest.raises(ValueError, match="unique"):
        AD4IntermolecularGrid(maps, maps[0], maps[0], atom_types=["C", "C"], origin=[0, 0, 0], spacing=.375)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("count", [1, 3])
def test_shared_geometry_matches_independent_channels_at_interior_and_outer_nodes(dtype, count):
    generator = torch.Generator().manual_seed(1729)
    field = TrilinearGrid(torch.randn(4, 4, 5, 6, generator=generator, dtype=dtype),
                          origin=ORIGIN, spacing=SPACING)
    xyz = _coordinates(dtype)
    xyz[0, 0] = field.origin
    xyz[1, -1] = field.origin + field.spacing * xyz.new_tensor([5, 4, 3])
    xyz.requires_grad_()
    channels = torch.randint(4, (2, 3, count), generator=generator)
    actual = field.sample_channels(xyz, channels)
    expected = torch.stack([field(xyz, channels[..., k]) for k in range(count)], dim=-1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weights = torch.randn(actual.shape, generator=generator, dtype=dtype)
    actual_gradient, = torch.autograd.grad((actual * weights).sum(), xyz, retain_graph=True)
    expected_gradient, = torch.autograd.grad((expected * weights).sum(), xyz)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-13
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=tolerance, atol=tolerance)


def test_shared_geometry_fp64_gradcheck_uses_anisotropic_physical_spacing():
    field = TrilinearGrid(_maps(lambda x, y, z: x * y * z, lambda x, y, z: x * y,
                                lambda x, y, z: y * z), origin=ORIGIN, spacing=SPACING)
    channels = torch.tensor([[[0, 1, 2], [2, 0, 1], [1, 2, 0]],
                             [[2, 1, 0], [1, 0, 2], [0, 2, 1]]])
    xyz = _coordinates().requires_grad_()
    assert torch.autograd.gradcheck(lambda values: field.sample_channels(values, channels),
                                    (xyz,), eps=1e-6, atol=1e-7)


@pytest.mark.parametrize("bad", ["shape", "empty", "float", "bool", "range", "coordinate_dtype", "nan", "bounds"])
def test_shared_geometry_rejects_invalid_channels_and_coordinates(bad):
    field = TrilinearGrid(_maps(lambda x, y, z: x + y + z), origin=ORIGIN, spacing=SPACING)
    xyz, channels = _coordinates(), torch.zeros((2, 3, 1), dtype=torch.long)
    if bad == "shape":
        channels = channels[..., 0]
    elif bad == "empty":
        channels = channels[..., :0]
    elif bad == "float":
        channels = channels.double()
    elif bad == "bool":
        channels = channels.bool()
    elif bad == "range":
        channels[0, 0, 0] = 1
    elif bad == "coordinate_dtype":
        xyz = xyz.float()
    elif bad == "nan":
        xyz[0, 0, 0] = float("nan")
    else:
        xyz[0, 0, 0] = field.origin[0] - 1e-7
    with pytest.raises(ValueError):
        field.sample_channels(xyz, channels)


@pytest.mark.parametrize("map_curl", [None, 1000.])
def test_ad4_shared_channels_preserve_repeat_baseline_coordinate_and_charge_gradients(map_curl):
    generator = torch.Generator().manual_seed(1741)
    maps = torch.randn(4, 4, 5, 6, generator=generator, dtype=torch.float64) * 3.
    component = AD4IntermolecularGrid(maps[:2], maps[2], maps[3], atom_types=["C", "OA"],
                                     origin=ORIGIN, spacing=SPACING, map_curl=map_curl, upper_exclusive=True)
    xyz = _coordinates().requires_grad_()
    charges = torch.tensor([[.3, -.4, .5], [-.1, .2, -.7]], dtype=xyz.dtype, requires_grad=True)
    channels = torch.tensor([[0, 1, 0], [1, 0, 1]])
    actual = component.per_atom(xyz, channels, charges)
    old_channels = torch.cat((channels, torch.full_like(channels, 2), torch.full_like(channels, 3)), dim=1)
    old_samples = component.field(xyz.repeat(1, 3, 1), old_channels)
    if map_curl is not None:
        old_samples = old_samples / (1. + old_samples.clamp_min(0.) / map_curl)
    affinity, electrostatic, desolvation = old_samples.split(3, dim=1)
    expected = affinity + charges * electrostatic + charges.abs() * desolvation
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_gradients = torch.autograd.grad(actual.sum(), (xyz, charges), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.sum(), (xyz, charges))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-13, atol=2e-13)
    upper = component.field.origin + component.field.spacing * xyz.new_tensor([5, 4, 3])
    outside = xyz.detach().clone()
    outside[0, 0] = upper
    with pytest.raises(ValueError, match="upper-exclusive"):
        component(outside, channels, charges.detach())
