from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from rdkit import Chem

from anchor_dock.core import features


@pytest.mark.parametrize(
    ("smiles", "types", "donors", "acceptors"),
    [
        ("CCO", ("C_H", "C_P", "O_DA"), [2], [2]),
        ("CC(=O)[O-]", ("C_H", "C_P", "O_A", "O_A"), [], [2, 3]),
        ("c1ccncc1", ("C_H", "C_H", "C_P", "N_A", "C_P", "C_H"), [], [3]),
        ("NC(=O)N", ("N_D", "C_P", "O_A", "N_D"), [0, 3], [2]),
        ("CS", ("C_P", "S_P"), [], []),
        ("CS(=O)(=O)N", ("C_P", "S_P", "O_A", "O_A", "N_DA"), [4], [2, 3, 4]),
        ("[Zn+2]", ("Met_D",), [0], []),
        ("[Na+].[Cl-]", ("Met_D", "Cl_H"), [0], []),
        ("B(O)O", ("X", "O_DA", "O_DA"), [1, 2], [1, 2]),
    ],
)
def test_feature_chemistry_is_preserved(smiles, types, donors, acceptors):
    molecule = Chem.MolFromSmiles(smiles)
    actual = features.compute_atom_features(molecule, "cpu")
    assert actual["xs_types"] == types
    assert features.infer_xs_atom_types(molecule) == list(types)
    assert actual["donor"].tolist() == [float(i in donors) for i in range(len(types))]
    assert actual["acceptor"].tolist() == [float(i in acceptors) for i in range(len(types))]
    assert actual["typing_version"] == features.ATOM_TYPING_VERSION


def test_compute_extracts_graph_roles_and_pdb_fallbacks_once(monkeypatch):
    graph_roles = Mock(wraps=features._feature_atom_sets)
    fallbacks = Mock(wraps=features._apply_pdb_fallbacks)
    monkeypatch.setattr(features, "_feature_atom_sets", graph_roles)
    monkeypatch.setattr(features, "_apply_pdb_fallbacks", fallbacks)
    features.compute_atom_features(Chem.MolFromSmiles("NCC(=O)O"), "cpu")
    graph_roles.assert_called_once()
    fallbacks.assert_called_once()


def test_unsanitized_protein_fallback_and_element_overrides(monkeypatch):
    # Misleading graph-role assignments cannot override declared protein
    # chemistry, unresolved HIS, or the XS sulfur/hydrogen element semantics.
    molecule = Chem.RWMol()
    entries = [
        (8, "HOH", "O"),
        (7, "LYS", "NZ"),
        (7, "HIS", "ND1"),
        (16, "LYS", "NZ"),
        (7, "PRO", "N"),
        (8, "GLY", "OXT"),
        (1, "LYS", "NZ"),
        (30, "HOH", "O"),
    ]
    for element, residue, name in entries:
        atom = Chem.Atom(element)
        atom.SetMonomerInfo(Chem.AtomPDBResidueInfo(atomName=f"{name:>4}", residueName=residue))
        molecule.AddAtom(atom)
    graph_roles = Mock(return_value=(set(range(len(entries))), set(range(len(entries)))))
    monkeypatch.setattr(features, "_feature_atom_sets", graph_roles)
    actual = features.compute_atom_features(molecule.GetMol(), "cpu")
    assert actual["xs_types"] == ("O_DA", "N_D", "N_P", "S_P", "N_P", "O_A", "H", "Met_D")
    assert actual["donor"].tolist() == [1, 1, 0, 0, 0, 0, 0, 1]
    assert actual["acceptor"].tolist() == [1, 0, 0, 0, 0, 1, 0, 0]
    assert actual["active"].tolist() == [True, True, True, True, True, True, False, True]
    graph_roles.assert_called_once()
    assert len(actual["typing_ambiguities"]) == 1


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
))])
def test_explicit_hydrogens_and_device_contract(device):
    molecule = Chem.AddHs(Chem.MolFromSmiles("NCS"))
    actual = features.compute_atom_features(molecule, device)
    hydrogen_indices = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 1]
    for name in ("radius_vina", "radius_vinardo", "hydrophobic", "donor", "acceptor", "active"):
        tensor = actual[name]
        assert tensor.shape == (molecule.GetNumAtoms(),)
        assert tensor.device.type == device
        assert tensor.dtype == (torch.bool if name == "active" else torch.float32)
        assert not tensor[hydrogen_indices].any()
