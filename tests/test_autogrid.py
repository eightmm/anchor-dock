import hashlib
from pathlib import Path

import pytest
import torch

from anchor_dock.core.autogrid import load_autogrid_maps, read_autogrid_map


def _write(path, values=None, *, spacing=.5, nelements=(2, 4, 6), center=(1, 2, 3), receptor="receptor.pdbqt"):
    if values is None:
        nx, ny, nz = (n + 1 for n in nelements)
        z, y, x = torch.meshgrid(torch.arange(nz), torch.arange(ny), torch.arange(nx), indexing="ij")
        values = x + 10 * y + 100 * z
    header = ("GRID_PARAMETER_FILE receptor.gpf\nGRID_DATA_FILE receptor.maps.fld\n"
              f"MACROMOLECULE {receptor}\nSPACING {spacing}\nNELEMENTS {' '.join(map(str, nelements))}\n"
              f"CENTER {' '.join(map(str, center))}\n")
    Path(path).write_text(header + "\n".join(str(v) for v in values.reshape(-1).tolist()) + "\n")


def test_interval_count_origin_axis_order_and_physical_gradient(tmp_path):
    path = tmp_path / "receptor.C.map"
    _write(path)
    field = read_autogrid_map(path, dtype=torch.float64)
    assert field.nelements == (2, 4, 6)
    assert field.origin == (.5, 1., 1.5)
    assert field.values.shape == (7, 5, 3)
    assert field.values[6, 4, 2] == 642
    assert field.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    for kind in ("e", "d"):
        _write(tmp_path / f"receptor.{kind}.map", torch.zeros(7, 5, 3))
    maps = load_autogrid_maps(tmp_path / "receptor", ["C"], dtype=torch.float64)
    grid = maps.to_inter_grid()
    xyz = torch.tensor([[[.75, 1.75, 2.75]]], dtype=torch.float64, requires_grad=True)
    energy = grid(xyz, [0], torch.zeros(1, dtype=torch.float64))
    assert energy.item() == pytest.approx(.5 + 15 + 250)
    gradient, = torch.autograd.grad(energy.sum(), xyz)
    torch.testing.assert_close(gradient, torch.tensor([[[2., 20., 200.]]], dtype=torch.float64))


@pytest.mark.parametrize("mutation", ["truncate", "extra", "nan", "spacing", "odd", "header", "numeric"])
def test_reject_malformed_maps(tmp_path, mutation):
    path = tmp_path / "bad.map"
    _write(path)
    lines = path.read_text().splitlines()
    if mutation == "truncate":
        lines.pop()
    elif mutation == "extra":
        lines.append("0")
    elif mutation == "nan":
        lines[-1] = "nan"
    elif mutation == "spacing":
        lines[3] = "SPACING -1"
    elif mutation == "odd":
        lines[4] = "NELEMENTS 3 4 6"
    elif mutation == "header":
        lines[4] = "ELEMENTS 2 4 6"
    elif mutation == "numeric":
        lines[-1] = "text"
    path.write_text("\n".join(lines))
    with pytest.raises(ValueError):
        read_autogrid_map(path)


@pytest.mark.parametrize("mismatch", ["center", "spacing", "nelements", "receptor"])
def test_mapset_rejects_mixed_geometry_or_receptor(tmp_path, mismatch):
    for kind in ("C", "e"):
        _write(tmp_path / f"receptor.{kind}.map")
    changes = dict(center=(1., 2., 3.1)) if mismatch == "center" else {
        "spacing": dict(spacing=.6), "nelements": dict(nelements=(4, 4, 6)),
        "receptor": dict(receptor="different.pdbqt"),
    }.get(mismatch, {})
    _write(tmp_path / "receptor.d.map", **changes)
    with pytest.raises(ValueError, match="identical|different receptor"):
        load_autogrid_maps(tmp_path / "receptor", ["C"])


@pytest.mark.parametrize("types", ["C", [], ["C", "C"], ["../C"], ["e"]])
def test_map_channel_order_is_explicit_and_safe(tmp_path, types):
    with pytest.raises(ValueError):
        load_autogrid_maps(tmp_path / "receptor", types)


def test_preserve_channel_order_and_map_snapshot(tmp_path):
    for kind, value in (("OA", 3.), ("C", 2.), ("e", 0.), ("d", 0.)):
        _write(tmp_path / f"receptor.{kind}.map", torch.full((7, 5, 3), value))
    maps = load_autogrid_maps(tmp_path / "receptor", ["OA", "C"])
    assert maps.atom_types == ("OA", "C")
    assert set(maps.source_sha256) == {"OA", "C", "e", "d"}
    grid = maps.to_inter_grid()
    maps.affinity[0].values.fill_(50.)
    assert grid(torch.tensor([[[1., 2., 3.]]]), [0], torch.zeros(1)).item() == 3.
