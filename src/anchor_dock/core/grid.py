"""Experimental fixed-map interpolation, not a complete AutoDock4 scorer.

Map storage is ``[channel, z, y, x]`` and physical coordinates are ``[x,y,z]``
in angstroms. Trilinear values are continuous but generally not C1 across cell
faces. Autograd differentiates the selected cell and includes inverse spacing.
This correctness reference rejects every out-of-bounds point; an optimizer's
restoring penalty must be an explicit, separately defined search objective.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _vector3(value, reference: torch.Tensor, name: str, *, scalar: bool = False) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device).detach().clone()
    if scalar and result.ndim == 0:
        result = result.repeat(3)
    if result.shape != (3,) or not torch.isfinite(result).all():
        raise ValueError(f"{name} must be a finite [x,y,z] vector")
    return result


def _atom_channels(value, batch: int, atoms: int, count: int, device: torch.device) -> torch.Tensor:
    indices = torch.as_tensor(value, device=device)
    if indices.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("atom channels must be integer indices")
    if indices.shape == (atoms,):
        indices = indices.unsqueeze(0).expand(batch, -1)
    elif indices.shape != (batch, atoms):
        raise ValueError("atom channels must have shape [N] or [B,N]")
    if ((indices < 0) | (indices >= count)).any():
        raise ValueError("atom channel index is outside the supplied map set")
    return indices.to(dtype=torch.long)


class TrilinearGrid(nn.Module):
    """Sample fixed finite maps at atom coordinates with exact cell derivatives.

    ``values`` must be floating point ``[C,Z,Y,X]`` with at least two nodes
    per spatial axis. ``origin`` is the physical first node, not a box center;
    ``spacing`` is a positive scalar or ``[sx,sy,sz]`` in angstroms. Module
    buffers are independent snapshots. Move/cast the module explicitly before
    scoring; coordinate dtype and device must match them.

    The strict bounds/finite checks intentionally synchronize CUDA. This is a
    small correctness reference, not a fused performance kernel.
    """

    def __init__(self, values: torch.Tensor, *, origin, spacing) -> None:
        super().__init__()
        if not isinstance(values, torch.Tensor) or values.ndim != 4:
            raise ValueError("grid values must have shape [C,Z,Y,X]")
        if values.dtype not in (torch.float32, torch.float64):
            raise ValueError("grid values must use float32 or float64")
        if values.shape[0] < 1 or any(size < 2 for size in values.shape[1:]):
            raise ValueError("grid needs at least one channel and two nodes per axis")
        if not torch.isfinite(values).all():
            raise ValueError("grid values must be finite")
        first = _vector3(origin, values, "origin")
        step = _vector3(spacing, values, "spacing", scalar=True)
        if (step <= 0).any():
            raise ValueError("grid spacing must be positive")
        self.register_buffer("values", values.detach().clone().contiguous())
        self.register_buffer("origin", first)
        self.register_buffer("spacing", step)

    @property
    def num_channels(self) -> int:
        return self.values.shape[0]

    def forward(self, coordinates: torch.Tensor, atom_channels) -> torch.Tensor:
        """Return ``[B,N]`` values for ``coordinates=[B,N,3]`` and atom channels.

        Points on the outer nodes are allowed. Their derivative is taken from
        the adjacent inner cell. Internal cell faces use the cell chosen by
        ``floor``; no claim of a unique derivative on those faces is made.
        """
        if not isinstance(coordinates, torch.Tensor) or coordinates.ndim != 3 or coordinates.shape[-1] != 3:
            raise ValueError("coordinates must have shape [B,N,3]")
        batch, atoms, _ = coordinates.shape
        if batch < 1 or atoms < 1:
            raise ValueError("coordinates require a nonempty batch and atom set")
        if coordinates.dtype != self.values.dtype or coordinates.device != self.values.device:
            raise ValueError("coordinate dtype/device must match the grid; cast/move the module explicitly")
        if not torch.isfinite(coordinates).all():
            raise ValueError("coordinates must be finite")
        channels = _atom_channels(atom_channels, batch, atoms, self.num_channels, coordinates.device)
        nz, ny, nx = self.values.shape[1:]
        last_indices = coordinates.new_tensor((nx - 1, ny - 1, nz - 1))
        upper = self.origin + self.spacing * last_indices
        if ((coordinates < self.origin) | (coordinates > upper)).any():
            raise ValueError("coordinates are outside the closed physical grid bounds")

        position = (coordinates - self.origin) / self.spacing
        lower = torch.floor(position).to(dtype=torch.long)
        last_cell = torch.tensor((nx - 2, ny - 2, nz - 2), dtype=torch.long, device=coordinates.device)
        # This only chooses the adjacent cell for an already-validated boundary
        # point. It never clamps out-of-grid coordinates into a flat energy field.
        lower = torch.minimum(torch.maximum(lower, torch.zeros_like(lower)), last_cell)
        fraction = position - lower.to(dtype=position.dtype)
        flat = self.values.reshape(-1)
        result = torch.zeros_like(coordinates[..., 0])
        for dz in (0, 1):
            wz = fraction[..., 2] if dz else 1.0 - fraction[..., 2]
            for dy in (0, 1):
                wy = fraction[..., 1] if dy else 1.0 - fraction[..., 1]
                for dx in (0, 1):
                    wx = fraction[..., 0] if dx else 1.0 - fraction[..., 0]
                    index = (((channels * nz + lower[..., 2] + dz) * ny + lower[..., 1] + dy) * nx
                             + lower[..., 0] + dx)
                    result = result + wx * wy * wz * flat[index]
        return result


class AD4IntermolecularGrid(nn.Module):
    """Experimental AD4 map-only intermolecular energy component.

    Uses authoritative, already-weighted affinity, electrostatic and
    charge-dependent desolvation maps:

    ``sum_i affinity[type_i](x_i) + q_i*electrostatic(x_i) + abs(q_i)*desolvation(x_i)``.

    Charges are in elementary-charge units. This component does not prepare
    atom types/charges, parse AutoGrid files, add intramolecular/torsional terms,
    subtract an unbound reference, or reproduce AutoDock-GPU search heuristics.
    Therefore its output is not a complete AD4 binding-affinity score. Maps are
    fixed buffers; coordinates and supplied charges remain differentiable.
    """

    def __init__(
        self,
        affinity_maps: torch.Tensor,
        electrostatic_map: torch.Tensor,
        desolvation_map: torch.Tensor,
        *,
        atom_types: Sequence[str],
        origin,
        spacing,
    ) -> None:
        super().__init__()
        if not isinstance(affinity_maps, torch.Tensor) or affinity_maps.ndim != 4:
            raise ValueError("affinity maps must have shape [C,Z,Y,X]")
        if isinstance(atom_types, (str, bytes)):
            raise ValueError("atom_types must be a sequence of type names, not one string")
        types = tuple(atom_types)
        if len(types) != affinity_maps.shape[0] or any(not isinstance(t, str) or not t for t in types):
            raise ValueError("one nonempty atom type name per affinity map is required")
        if len(set(types)) != len(types):
            raise ValueError("affinity map atom type names must be unique")
        spatial = affinity_maps.shape[1:]
        others = []
        for name, value in (("electrostatic", electrostatic_map), ("desolvation", desolvation_map)):
            if not isinstance(value, torch.Tensor) or value.shape != spatial:
                raise ValueError(f"{name} map must have the same [Z,Y,X] shape as affinity maps")
            if value.dtype != affinity_maps.dtype or value.device != affinity_maps.device:
                raise ValueError(f"{name} map dtype/device must match affinity maps")
            others.append(value.unsqueeze(0))
        self.atom_types = types
        self.field = TrilinearGrid(torch.cat((affinity_maps, *others), dim=0), origin=origin, spacing=spacing)

    def per_atom(self, coordinates: torch.Tensor, atom_channels, charges: torch.Tensor) -> torch.Tensor:
        """Return the map-only component per atom, shape ``[B,N]``."""
        if not isinstance(coordinates, torch.Tensor) or coordinates.ndim != 3 or coordinates.shape[-1] != 3:
            raise ValueError("coordinates must have shape [B,N,3]")
        batch, atoms, _ = coordinates.shape
        channels = _atom_channels(atom_channels, batch, atoms, len(self.atom_types), coordinates.device)
        if not isinstance(charges, torch.Tensor) or charges.dtype != coordinates.dtype or charges.device != coordinates.device:
            raise ValueError("charges must be a tensor with coordinate dtype/device")
        if charges.shape == (atoms,):
            charges = charges.unsqueeze(0).expand(batch, -1)
        elif charges.shape != (batch, atoms):
            raise ValueError("charges must have shape [N] or [B,N]")
        if not torch.isfinite(charges).all():
            raise ValueError("charges must be finite")
        electrostatic = torch.full_like(channels, len(self.atom_types))
        desolvation = electrostatic + 1
        all_channels = torch.cat((channels, electrostatic, desolvation), dim=1)
        sampled = self.field(coordinates.repeat(1, 3, 1), all_channels)
        affinity, electro, desolv = sampled.split(atoms, dim=1)
        return affinity + charges * electro + charges.abs() * desolv

    def forward(self, coordinates: torch.Tensor, atom_channels, charges: torch.Tensor) -> torch.Tensor:
        """Return the intermolecular map contribution per pose, shape ``[B]``."""
        return self.per_atom(coordinates, atom_channels, charges).sum(dim=1)
