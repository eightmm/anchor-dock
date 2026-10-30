"""Force-field typing regressions, independent of RDKit pharmacophore labels.

Expected ordinary ligand XS types are frozen from Meeko AD4 preset + official
Vina AD-to-XS conventions. They are not a generic medicinal-chemistry HBA label.
"""
from __future__ import annotations

import pytest
import torch
from rdkit import Chem

from anchor_dock.core import features as typing_module
from anchor_dock.core.features import (
    ATOM_TYPING_VERSION,
    AUTHORITATIVE_TYPING_VERSION,
    compute_atom_features,
    compute_atom_features_from_ad4,
)


@pytest.mark.parametrize(('smiles', 'expected'), [
    ('CN', ('C_P', 'N_DA')),
    ('CNC', ('C_P', 'N_DA', 'C_P')),
    ('CN(C)C', ('C_P', 'N_A', 'C_P', 'C_P')),
    ('C[NH3+]', ('C_P', 'N_D')),
    ('C[N+](C)(C)C', ('C_P', 'N_P', 'C_P', 'C_P', 'C_P')),
    ('CC(=O)[O-]', ('C_H', 'C_P', 'O_A', 'O_A')),
    ('CC(=O)O', ('C_H', 'C_P', 'O_A', 'O_DA')),
    ('C[N+](=O)[O-]', ('C_P', 'N_P', 'O_A', 'O_A')),
    ('NC(=O)N', ('N_D', 'C_P', 'O_A', 'N_D')),
    ('CN(C)C=O', ('C_P', 'N_P', 'C_P', 'C_P', 'O_A')),
    ('Nc1ccccc1', ('N_D', 'C_P', 'C_H', 'C_H', 'C_H', 'C_H', 'C_H')),
    ('c1ncc[nH]1', ('C_P', 'N_A', 'C_P', 'C_P', 'N_D')),
    ('c1cc[nH]c1', ('C_H', 'C_H', 'C_P', 'N_D', 'C_P')),
    ('CS', ('C_P', 'S_P')),
    ('CSC', ('C_P', 'S_P', 'C_P')),
])
@pytest.mark.parametrize('explicit_h', [False, True])
def test_graph_typing_matches_frozen_prepared_forcefield_types(smiles, expected, explicit_h):
    mol = Chem.MolFromSmiles(smiles)
    if explicit_h:
        mol = Chem.AddHs(mol)
    features = compute_atom_features(mol, 'cpu', strict=True)
    heavy = tuple(t for atom, t in zip(mol.GetAtoms(), features['xs_types'], strict=True) if atom.GetAtomicNum() != 1)
    assert heavy == expected
    assert features['typing_version'] == ATOM_TYPING_VERSION == 'inferred-xs-v3'
    assert features['typing_ambiguities'] == ()


def _histidine(residue='HIS', protonated=()):
    mol = Chem.RWMol()
    for name in ('ND1', 'NE2'):
        atom = Chem.Atom(7)
        atom.SetNoImplicit(True)
        atom.SetMonomerInfo(Chem.AtomPDBResidueInfo(atomName=f'{name:>4}', residueName=residue,
                                                   residueNumber=10, chainId='A'))
        mol.AddAtom(atom)
    for name in protonated:
        idx = ('ND1', 'NE2').index(name)
        hydrogen = mol.AddAtom(Chem.Atom(1))
        mol.AddBond(idx, hydrogen, Chem.BondType.SINGLE)
    return mol.GetMol()


def test_unspecified_histidine_is_conservative_and_strictly_rejectable():
    mol = _histidine()
    features = compute_atom_features(mol, 'cpu')
    assert features['xs_types'] == ('N_P', 'N_P')
    assert features['donor'].sum().item() == features['acceptor'].sum().item() == 0
    assert len(features['typing_ambiguities']) == 2
    with pytest.raises(ValueError, match='unresolved HIS'):
        compute_atom_features(mol, 'cpu', strict=True)


@pytest.mark.parametrize(('residue', 'expected'), [
    ('HID', ('N_D', 'N_A')), ('HSD', ('N_D', 'N_A')),
    ('HIE', ('N_A', 'N_D')), ('HSE', ('N_A', 'N_D')),
    ('HIP', ('N_D', 'N_D')), ('HSP', ('N_D', 'N_D')),
])
def test_declared_histidine_tautomers_are_not_guessed(residue, expected):
    features = compute_atom_features(_histidine(residue), 'cpu', strict=True)
    assert features['xs_types'] == expected


@pytest.mark.parametrize(('protonated', 'expected'), [
    (('ND1',), ('N_D', 'N_A')), (('NE2',), ('N_A', 'N_D')),
    (('ND1', 'NE2'), ('N_D', 'N_D')),
])
def test_explicit_histidine_hydrogens_determine_roles(protonated, expected):
    features = compute_atom_features(_histidine(protonated=protonated), 'cpu', strict=True)
    assert features['xs_types'][:2] == expected


def test_histidine_alias_conflicting_with_explicit_h_is_not_silently_overridden():
    mol = _histidine('HID', ('NE2',))
    actual = compute_atom_features(mol, 'cpu')
    assert actual['xs_types'][:2] == ('N_P', 'N_P')
    assert all('conflicts' in message for message in actual['typing_ambiguities'])
    with pytest.raises(ValueError, match='conflicts'):
        compute_atom_features(mol, 'cpu', strict=True)


def test_graph_failure_is_explicit_and_only_pdb_templates_can_fallback(monkeypatch):
    def fail(_):
        raise RuntimeError('malformed graph')
    monkeypatch.setattr(typing_module, '_feature_atom_sets', fail)
    with pytest.raises(ValueError, match='valid bond/protonation'):
        compute_atom_features(Chem.MolFromSmiles('CN'), 'cpu')
    actual = compute_atom_features(_histidine('HID'), 'cpu')
    assert actual['xs_types'] == ('N_D', 'N_A')
    assert actual['typing_ambiguities'] == ('graph-role inference unavailable; PDB residue-only inference used',)


def test_unvalidated_metals_are_not_given_a_generic_donor_radius():
    features = compute_atom_features(Chem.MolFromSmiles('[Li+]'), 'cpu')
    assert features['xs_types'] == ('X',)
    assert 'unsupported element Li' in features['typing_ambiguities'][0]
    with pytest.raises(ValueError, match='unsupported element'):
        compute_atom_features(Chem.MolFromSmiles('[Al+3]'), 'cpu', strict=True)


def test_authoritative_ad4_conversion_uses_explicit_hd_and_supplied_acceptor_types():
    mol = Chem.AddHs(Chem.MolFromSmiles('CN'))
    # Input order is RDKit C,N followed by C H,H,H and N H,H.
    ad4 = ['C', 'NA'] + ['H'] * 3 + ['HD'] * 2
    features = compute_atom_features_from_ad4(mol, ad4, 'cpu')
    assert features['xs_types'][:2] == ('C_P', 'N_DA')
    assert features['typing_version'] == AUTHORITATIVE_TYPING_VERSION
    assert features['typing_source'] == 'authoritative-ad4-input'
    assert features['typing_ambiguities'] == ()
    ad4[1] = 'N'
    assert compute_atom_features_from_ad4(mol, ad4, 'cpu')['xs_types'][1] == 'N_D'


def test_authoritative_ad4_rejects_missing_hydrogens_bad_mapping_and_extended_types():
    with pytest.raises(ValueError, match='explicit polar-H'):
        compute_atom_features_from_ad4(Chem.MolFromSmiles('CN'), ['C', 'NA'], 'cpu')
    with pytest.raises(ValueError, match='per graph atom'):
        compute_atom_features_from_ad4(Chem.MolFromSmiles('CO'), ['C'], 'cpu')
    with pytest.raises(ValueError, match='element/order'):
        compute_atom_features_from_ad4(Chem.MolFromSmiles('CO'), ['OA', 'C'], 'cpu')
    with pytest.raises(ValueError, match='unsupported authoritative'):
        compute_atom_features_from_ad4(Chem.MolFromSmiles('C'), ['CG0'], 'cpu')
    with pytest.raises(ValueError, match='bonded polar'):
        compute_atom_features_from_ad4(Chem.AddHs(Chem.MolFromSmiles('C')), ['C', 'HD', 'H', 'H', 'H'], 'cpu')


def test_ad4_sulfur_acceptor_does_not_turn_into_an_xs_hbond_acceptor():
    mol = Chem.MolFromSmiles('CSC')
    features = compute_atom_features_from_ad4(mol, ['C', 'SA', 'C'], 'cpu')
    assert features['xs_types'][1] == 'S_P'
    assert features['acceptor'][1].item() == 0


def test_forcefield_oxygen_convention_is_not_a_medchem_label():
    # The ordinary AD4 preset labels all O as OA; retain that explicit contract.
    mol = Chem.MolFromSmiles('[OH3+]')
    features = compute_atom_features(mol, 'cpu')
    assert features['xs_types'] == ('O_DA',)
    assert features['acceptor'].item() == 1
    assert features['active'].dtype == torch.bool
