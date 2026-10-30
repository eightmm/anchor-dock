"""RMSD benchmark retains the receptor frame and molecular symmetry."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

_PATH = Path(__file__).resolve().parents[1] / "examples/benchmark_redocking.py"
_SPEC = importlib.util.spec_from_file_location("anchor_benchmark_redocking", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
fixed_frame_symmetry_rmsd = _MODULE.fixed_frame_symmetry_rmsd


def test_fixed_frame_rmsd_does_not_fit_out_translation_or_rotation():
    mol = Chem.MolFromSmiles("CCO")
    reference = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]])
    translated = reference + np.array([1.25, 0., 0.])
    rotated = reference @ np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    before = np.stack([translated, rotated])
    measured = fixed_frame_symmetry_rmsd(mol, reference, before)
    assert measured[0] == pytest.approx(1.25)
    assert measured[1] > 1.0
    np.testing.assert_array_equal(before, np.stack([translated, rotated]))


def test_fixed_frame_rmsd_handles_graph_symmetry():
    mol = Chem.MolFromSmiles("CC")
    reference = np.array([[-1., 0., 0.], [1., 0., 0.]])
    assert fixed_frame_symmetry_rmsd(mol, reference, reference[::-1])[0] == pytest.approx(0.)


def test_fixed_frame_rmsd_symmetrizes_carboxylate_terminal_oxygen():
    mol = Chem.MolFromSmiles("CC(=O)[O-]")
    reference = np.array([[0., 0., 0.], [1., 0., 0.], [2., 1., 0.], [2., -1., 0.]])
    assert fixed_frame_symmetry_rmsd(mol, reference, reference[[0, 1, 3, 2]])[0] == pytest.approx(0.)


def test_fixed_frame_rmsd_rejects_truncated_symmetry():
    mol = Chem.MolFromSmiles("CC(=O)[O-]")
    coordinates = np.zeros((mol.GetNumAtoms(), 3))
    with pytest.raises(ValueError, match="symmetry exceeds"):
        fixed_frame_symmetry_rmsd(mol, coordinates, coordinates, max_matches=1)


@pytest.mark.parametrize("bad", [np.zeros((2, 3)), np.full((3, 3), np.nan), np.zeros((0, 3, 3))])
def test_fixed_frame_rmsd_rejects_misaligned_or_nonfinite_inputs(bad):
    with pytest.raises(ValueError):
        fixed_frame_symmetry_rmsd(Chem.MolFromSmiles("CCO"), np.zeros((3, 3)), bad)


def test_fixed_frame_rmsd_rejects_mapping_change_by_h_removal():
    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    coordinates = np.zeros((mol.GetNumAtoms(), 3))
    with pytest.raises(ValueError, match="heavy atoms only"):
        fixed_frame_symmetry_rmsd(mol, coordinates, coordinates)
