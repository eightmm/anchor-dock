"""Ambiguous chemistry must remain visible at the scoring entry point."""
from __future__ import annotations

import pytest
import torch
from rdkit import Chem

from anchor_dock.core.engine import AtomTypingWarning, DockingEngine
from anchor_dock.core.io import receptor_context_from_mol


def _ambiguous_receptor(residue="HIS"):
    mol = Chem.RWMol()
    atom = Chem.Atom(7)
    atom.SetNoImplicit(True)
    atom.SetMonomerInfo(Chem.AtomPDBResidueInfo(atomName=' ND1', residueName=residue, residueNumber=10, chainId='A'))
    mol.AddAtom(atom)
    mol = mol.GetMol()
    conf = Chem.Conformer(1)
    conf.SetAtomPosition(0, (0., 0., 0.))
    mol.AddConformer(conf)
    return receptor_context_from_mol(mol, 'cpu')


def test_permissive_engine_exposes_selected_pocket_ambiguity():
    receptor = _ambiguous_receptor()
    assert receptor.features['xs_types'] == ('N_P',)
    with pytest.warns(AtomTypingWarning, match='unresolved HIS'):
        problem = DockingEngine(device='cpu').prepare(Chem.MolFromSmiles('CCO'), receptor)
    assert problem.scorer.receptor_features['typing_ambiguities']


def test_strict_engine_rejects_ambiguous_selected_receptor():
    with pytest.raises(ValueError, match='ambiguous receptor atom typing'):
        DockingEngine(device='cpu', strict_atom_typing=True).prepare(Chem.MolFromSmiles('CCO'), _ambiguous_receptor())


def test_strict_engine_accepts_unambiguous_graph_and_types():
    receptor = _ambiguous_receptor('HID')
    problem = DockingEngine(device='cpu', strict_atom_typing=True).prepare(Chem.MolFromSmiles('CN'), receptor)
    assert torch.equal(problem.scorer.ligand_features['acceptor'], torch.tensor([0., 1.]))
