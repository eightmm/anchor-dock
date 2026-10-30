"""Prepared mappings preserve chemistry and fail closed on incompatible data."""
from __future__ import annotations

import pytest
import torch
from rdkit import Chem

from anchor_dock.core.engine import DockingEngine
from anchor_dock.core.io import receptor_context_from_mol
from anchor_dock.core.kinematics import LigandKinematics
from anchor_dock.core.prepared import PreparedLigand, parse_pdbqt, receptor_context_from_prepared_pdbqt
from anchor_dock.core.topology import PREPARED_ROTORS_PROPERTY, find_rotatable_bonds


def _mol(smiles, hydrogens=False):
    mol = Chem.MolFromSmiles(smiles)
    if hydrogens:
        mol = Chem.AddHs(mol)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (1.3*i, .2*(i % 3), .3*(i % 2)))
    mol.AddConformer(conf)
    return mol


def _atom(mol, index, serial, kind, charge=0.):
    xyz = mol.GetConformer().GetAtomPosition(index)
    chars = list(' '*80)
    chars[:6] = 'ATOM  '
    chars[6:11] = f'{serial:5d}'
    chars[12:16] = f'{mol.GetAtomWithIdx(index).GetSymbol():>4}'
    for start, value in zip((30, 38, 46), xyz, strict=True):
        chars[start:start+8] = f'{value:8.3f}'
    chars[70:76] = f'{charge:6.3f}'
    return ''.join(chars[:77]) + kind


def _rigid(mol, kinds=None, order=None):
    kinds = ['C']*mol.GetNumAtoms() if kinds is None else kinds
    order = list(range(mol.GetNumAtoms())) if order is None else order
    serials = {index: 11 + 3*index for index in order}
    text = '\n'.join(['ROOT', *[_atom(mol, i, serials[i], kinds[i], .1*i) for i in order],
                      'ENDROOT', 'TORSDOF 0'])
    return text, {serials[i]: i for i in order}


def test_reordered_noncontiguous_mapping_preserves_graph_order_and_copy():
    mol = _mol('CC#N')
    text, mapping = _rigid(mol, ['C', 'C', 'NA'], [2, 0, 1])
    data = PreparedLigand.from_pdbqt(mol, text, serial_to_atom_index=mapping)
    assert data.ad4_types == ('C', 'C', 'NA')
    assert data.pdbqt_serials == (11, 14, 17)
    assert data.original_atom_indices == (0, 1, 2)
    assert data.features()['xs_types'] == ('C_H', 'C_P', 'N_A')
    torch.testing.assert_close(data.coordinates(dtype=torch.float64),
                               torch.tensor(mol.GetConformer().GetPositions()))
    copied = data.mol
    copied.GetAtomWithIdx(0).SetFormalCharge(1)
    assert data.mol.GetAtomWithIdx(0).GetFormalCharge() == 0


def test_omitted_nonpolar_h_keeps_polar_h_and_donor_assignment():
    mol = _mol('CO', hydrogens=True)
    polar = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1
                 and a.GetNeighbors()[0].GetAtomicNum() == 8)
    atoms = [(0, 30, 'C'), (1, 10, 'OA'), (polar, 42, 'HD')]
    text = '\n'.join(['ROOT', *[_atom(mol, i, serial, kind) for i, serial, kind in atoms],
                      'ENDROOT', 'TORSDOF 0'])
    data = PreparedLigand.from_pdbqt(mol, text, serial_to_atom_index={s: i for i, s, _ in atoms})
    assert data.mol.GetNumAtoms() == 3
    assert data.original_atom_indices == (0, 1, polar)
    assert data.features()['xs_types'] == ('C_P', 'O_DA', 'H')
    assert data.features()['active'].tolist() == [True, True, False]
    missing = text.replace(_atom(mol, polar, 42, 'HD')+'\n', '')
    with pytest.raises(ValueError, match='omit only nonpolar'):
        PreparedLigand.from_pdbqt(mol, missing, serial_to_atom_index={30: 0, 10: 1})


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'wrong_coords', 'wrong_element'])
def test_mapping_rejects_missing_duplicate_or_misaligned_atoms(change):
    mol = _mol('CC#N')
    text, mapping = _rigid(mol, ['C', 'C', 'NA'])
    if change == 'missing':
        mapping.pop(11)
    elif change == 'duplicate':
        mapping[14] = 0
    elif change == 'wrong_coords':
        mapping[11], mapping[14] = mapping[14], mapping[11]
    else:
        text = text.replace('NA', 'OA')
    with pytest.raises(ValueError):
        PreparedLigand.from_pdbqt(mol, text, serial_to_atom_index=mapping)


@pytest.mark.parametrize('extra', ['MODEL 1', 'BEGIN_RES ALA A 1', 'ROOT', 'TORSDOF 0', 'BRANCH 11 99'])
def test_unsupported_or_malformed_tree_records_fail(extra):
    text, _ = _rigid(_mol('CC'))
    with pytest.raises(ValueError):
        parse_pdbqt(text+'\n'+extra)


def test_authoritative_rotors_drive_kinematics_and_half_integral_vina_count():
    mol = _mol('CCCCCC')
    data = PreparedLigand.from_arrays(mol, ['C']*6, [0.]*6, rotatable_bonds=[(0, 1), (2, 3)], torsdof=7)
    assert data.vina_num_tors == 1.5
    assert data.torsdof == 7
    assert find_rotatable_bonds(data.mol) == [(0, 1), (2, 3)]
    model = LigandKinematics(data.mol, [0], data.coordinates(), 'cpu')
    assert model.num_torsions == 2
    receptor = receptor_context_from_mol(_mol('C'), 'cpu')
    problem = DockingEngine(device='cpu').prepare(data.mol, receptor, prepared_ligand=data)
    assert problem.num_rotatable_bonds == 1.5
    assert problem.prepared_input_fingerprint == data.fingerprint
    assert problem.scorer.ligand_features['typing_source'] == 'authoritative-ad4-input'
    with pytest.raises(ValueError, match='conflicts'):
        DockingEngine(device='cpu').prepare(data.mol, receptor, prepared_ligand=data, num_rotatable_bonds=2)


def test_prepared_mobility_excludes_fixed_axis_endpoint_distances_beyond_graph_14():
    mol = _mol('CCCCCCC')
    data = PreparedLigand.from_arrays(mol, ['C']*7, [0.]*7, rotatable_bonds=[(1, 2)])
    mask = data.intramolecular_mask()
    assert not mask[1, 6]
    assert mask[0, 6]
    assert not mask[0, 3]
    assert torch.equal(mask, mask.T)
    assert (0, 6) in data.intramolecular_pairs


def test_fingerprint_covers_charge_types_tree_and_stereochemistry():
    mol = _mol('CCCC')
    base = PreparedLigand.from_arrays(mol, ['C']*4, [0.]*4)
    charge = PreparedLigand.from_arrays(mol, ['C']*4, [0., .1, 0., 0.])
    tree = PreparedLigand.from_arrays(mol, ['C']*4, [0.]*4, rotatable_bonds=[(1, 2)])
    type_change = PreparedLigand.from_arrays(mol, ['A', 'C', 'C', 'C'], [0.]*4)
    assert len({data.fingerprint for data in (base, charge, tree, type_change)}) == 4
    cis, trans = _mol('F/C=C/F'), _mol('F/C=C\\F')
    data = PreparedLigand.from_arrays(cis, ['F', 'C', 'C', 'F'], [0.]*4)
    other = PreparedLigand.from_arrays(trans, ['F', 'C', 'C', 'F'], [0.]*4)
    assert data.fingerprint != other.fingerprint
    with pytest.raises(ValueError, match='graph/order'):
        data.validate_mol(trans)


def test_rigid_prepared_receptor_accepts_disconnected_graph_without_templates():
    mol = _mol('[Na+].[Cl-]')
    text, mapping = _rigid(mol, ['Na', 'Cl'])
    rigid = '\n'.join(line for line in text.splitlines() if line.startswith('ATOM'))
    receptor = receptor_context_from_prepared_pdbqt(mol, rigid, serial_to_atom_index=mapping)
    assert receptor.features['typing_source'] == 'authoritative-ad4-input'
    assert receptor.atom_typing_version == 'prepared-ad4-xs-v1'
    assert receptor.features['ad4_types'] == ('Na', 'Cl')
    assert 'prepared_intramolecular_pairs' not in receptor.features


@pytest.mark.parametrize('kwargs', [dict(learning_rate=float('nan')), dict(min_delta=float('inf')),
                                   dict(num_steps=True), dict(patience=1.5), dict(batch_size=0),
                                   dict(optimizer='bogus'), dict(strict_atom_typing=1)])
def test_engine_rejects_invalid_optimizer_configuration(kwargs):
    with pytest.raises(ValueError):
        DockingEngine(device='cpu', **kwargs)


def test_bad_prepared_tree_or_graph_contract_fails():
    ring = _mol('C1CCCCC1')
    with pytest.raises(ValueError, match='rings'):
        PreparedLigand.from_arrays(ring, ['C']*6, [0.]*6, rotatable_bonds=[(0, 1)])
    mol = _mol('CCC')
    mol.SetProp(PREPARED_ROTORS_PROPERTY, '[[0,9]]')
    with pytest.raises(ValueError, match='prepared rotor'):
        find_rotatable_bonds(mol)
    data = PreparedLigand.from_arrays(_mol('CCC'), ['C']*3, [0.]*3)
    with pytest.raises(ValueError, match='graph/order'):
        data.validate_mol(_mol('CCO'))


def test_pdbqt_branch_tree_is_applied_and_cross_group_bonds_are_checked():
    mol = _mol('CCCCCC')
    atoms = [_atom(mol, i, i+1, 'C') for i in range(6)]
    text = '\n'.join(['ROOT', *atoms[:3], 'ENDROOT', 'BRANCH 3 4', *atoms[3:], 'ENDBRANCH 3 4', 'TORSDOF 1'])
    mapping = {i+1: i for i in range(6)}
    data = PreparedLigand.from_pdbqt(mol, text, serial_to_atom_index=mapping)
    assert data.rotatable_bonds == ((2, 3),)
    assert data.vina_num_tors == 1.
    assert data.intramolecular_pairs == ((0, 4), (0, 5), (1, 5))
    malformed = text.replace('BRANCH 3 4', 'BRANCH 2 4')
    with pytest.raises(ValueError, match='existing graph bond'):
        PreparedLigand.from_pdbqt(mol, malformed, serial_to_atom_index=mapping)
    # The added cross-group closure contradicts a free prepared torsion.
    edited = Chem.RWMol(mol)
    edited.AddBond(0, 5, Chem.BondType.SINGLE)
    with pytest.raises(ValueError, match='rings'):
        PreparedLigand.from_pdbqt(edited.GetMol(), text, serial_to_atom_index=mapping)


@pytest.mark.parametrize('tolerance', [True, -.1, float('nan'), float('inf')])
def test_pdbqt_coordinate_tolerance_contract(tolerance):
    mol = _mol('C')
    text, mapping = _rigid(mol)
    with pytest.raises(ValueError, match='coordinate_tolerance'):
        PreparedLigand.from_pdbqt(mol, text, serial_to_atom_index=mapping, coordinate_tolerance=tolerance)


@pytest.mark.parametrize('kwargs', [dict(charges=[float('nan')]), dict(charges=[float('inf')]),
                                   dict(coordinates=[[0., float('nan'), 1.]]), dict(torsdof=True),
                                   dict(torsdof=-1), dict(rotatable_bonds=[(0, 0)])])
def test_invalid_prepared_array_values_fail(kwargs):
    values = dict(charges=[0.])
    values.update(kwargs)
    with pytest.raises(ValueError):
        PreparedLigand.from_arrays(_mol('C'), ['C'], **values)


def test_implicit_polar_h_and_macrocycle_dummy_types_are_rejected():
    with pytest.raises(ValueError, match='explicit polar-H'):
        PreparedLigand.from_arrays(_mol('CO'), ['C', 'OA'], [0., 0.])
    with pytest.raises(ValueError, match='unsupported authoritative'):
        PreparedLigand.from_arrays(_mol('CC'), ['CG0', 'C'], [0., 0.])
