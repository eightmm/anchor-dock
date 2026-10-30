"""Strict AutoGrid map ingestion with explicit physical grid geometry.

NELEMENTS counts intervals; each axis has NELEMENTS + 1 nodes. File samples
vary x fastest. This module never infers receptor chemistry or generates maps.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from .grid import AD4IntermolecularGrid


@dataclass(frozen=True)
class AutoGridMap:
    values: torch.Tensor
    spacing: float
    nelements: tuple[int, int, int]
    center: tuple[float, float, float]
    source_sha256: str
    source_path: str
    receptor: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.spacing) or self.spacing <= 0 or len(self.nelements) != 3 or any(
            type(v) is not int or v < 2 or v % 2 for v in self.nelements
        ) or len(self.center) != 3 or any(not math.isfinite(v) for v in self.center):
            raise ValueError("invalid canonical AutoGrid geometry")
        expected = tuple(v + 1 for v in reversed(self.nelements))
        if not isinstance(self.values, torch.Tensor) or self.values.shape != expected:
            raise ValueError("AutoGrid values must match NELEMENTS+1 in [Z,Y,X] order")
        if self.values.dtype not in (torch.float32, torch.float64) or not torch.isfinite(self.values).all():
            raise ValueError("AutoGrid values must be finite float32 or float64")
        if not isinstance(self.source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ValueError("AutoGrid source_sha256 must identify the original map content")

    @property
    def origin(self) -> tuple[float, float, float]:
        return tuple(c - n * self.spacing / 2 for c, n in zip(self.center, self.nelements, strict=True))


def read_autogrid_map(path: str | Path, *, dtype: torch.dtype = torch.float32,
                      device: torch.device | str = "cpu") -> AutoGridMap:
    """Read a canonical six-header AutoGrid ASCII map, failing on malformed data."""
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("AutoGrid maps require float32 or float64")
    path = Path(path)
    raw = path.read_bytes()
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("AutoGrid maps must be UTF-8 ASCII-compatible text") from exc
    names = ("GRID_PARAMETER_FILE", "GRID_DATA_FILE", "MACROMOLECULE", "SPACING", "NELEMENTS", "CENTER")
    if len(lines) < 6:
        raise ValueError("AutoGrid map is missing its six-line header")
    headers = []
    for line, expected in zip(lines[:6], names, strict=True):
        fields = line.split()
        if not fields or fields[0] != expected or len(fields) < 2:
            raise ValueError(f"expected AutoGrid header {expected}")
        headers.append(fields[1:])
    if len(headers[3]) != 1 or len(headers[4]) != 3 or len(headers[5]) != 3:
        raise ValueError("AutoGrid spacing, interval counts and center have invalid dimensions")
    try:
        spacing = float(headers[3][0])
        nelements = tuple(int(v) for v in headers[4])
        center = tuple(float(v) for v in headers[5])
    except ValueError as exc:
        raise ValueError("invalid numeric AutoGrid geometry") from exc
    if not math.isfinite(spacing) or spacing <= 0 or not all(math.isfinite(v) for v in center):
        raise ValueError("AutoGrid geometry must be finite with positive spacing")
    if any(v < 2 or v % 2 for v in nelements):
        raise ValueError("AutoGrid NELEMENTS must contain positive even interval counts")
    count = math.prod(v + 1 for v in nelements)
    samples = " ".join(lines[6:]).split()
    if len(samples) != count:
        raise ValueError(f"AutoGrid map has {len(samples)} samples; expected {count}")
    try:
        values = torch.tensor([float(v) for v in samples], dtype=dtype, device=device)
    except ValueError as exc:
        raise ValueError("AutoGrid map contains a nonnumeric sample") from exc
    if not torch.isfinite(values).all():
        raise ValueError("AutoGrid map samples must be finite in the requested dtype")
    nx, ny, nz = (v + 1 for v in nelements)
    return AutoGridMap(values.reshape(nz, ny, nx), spacing, nelements, center,
                       hashlib.sha256(raw).hexdigest(), str(path), " ".join(headers[2]))


@dataclass(frozen=True)
class AutoGridMapSet:
    """Caller-ordered affinity channels with electrostatic/desolvation maps."""

    atom_types: tuple[str, ...]
    affinity: tuple[AutoGridMap, ...]
    electrostatic: AutoGridMap
    desolvation: AutoGridMap

    def __post_init__(self) -> None:
        if not self.atom_types or len(self.atom_types) != len(self.affinity):
            raise ValueError("one affinity map is required per atom type")
        if len(set(self.atom_types)) != len(self.atom_types) or any(
            not isinstance(t, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", t) or t in {"e", "d"}
            for t in self.atom_types
        ):
            raise ValueError("AutoGrid affinity map types must be unique safe channel names")
        reference = self.electrostatic
        for field in (*self.affinity, self.desolvation):
            if field.nelements != reference.nelements or field.spacing != reference.spacing or field.center != reference.center:
                raise ValueError("all AutoGrid maps must have identical intervals, spacing and center")
            if field.values.dtype != reference.values.dtype or field.values.device != reference.values.device:
                raise ValueError("all AutoGrid maps must have the same dtype/device")
            if field.receptor != reference.receptor:
                raise ValueError("AutoGrid maps refer to different receptor sources")

    @property
    def source_sha256(self) -> dict[str, str]:
        return {kind: field.source_sha256 for kind, field in zip(
            (*self.atom_types, "e", "d"), (*self.affinity, self.electrostatic, self.desolvation), strict=True)}

    def to_inter_grid(self, *, map_curl: float | None = None, upper_exclusive: bool = False) -> AD4IntermolecularGrid:
        """Snapshot maps into a module. Supplied map coefficients are already weighted."""
        return AD4IntermolecularGrid(torch.stack([v.values for v in self.affinity]),
                                     self.electrostatic.values, self.desolvation.values,
                                     atom_types=self.atom_types, origin=self.electrostatic.origin,
                                     spacing=self.electrostatic.spacing, map_curl=map_curl,
                                     upper_exclusive=upper_exclusive)


def load_autogrid_maps(prefix: str | Path, atom_types: Sequence[str], *, dtype: torch.dtype = torch.float32,
                       device: torch.device | str = "cpu") -> AutoGridMapSet:
    """Load prefix.TYPE.map, prefix.e.map and prefix.d.map in explicit type order."""
    if isinstance(atom_types, (str, bytes)):
        raise ValueError("atom_types must be an explicit sequence")
    types = tuple(atom_types)
    if not types or len(set(types)) != len(types) or any(
        not isinstance(t, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", t) or t in {"e", "d"} for t in types
    ):
        raise ValueError("atom_types must be unique safe affinity channel names")
    fields = [read_autogrid_map(f"{prefix}.{kind}.map", dtype=dtype, device=device) for kind in (*types, "e", "d")]
    return AutoGridMapSet(types, tuple(fields[:-2]), fields[-2], fields[-1])
