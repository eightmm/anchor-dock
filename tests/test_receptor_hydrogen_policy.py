"""Supplied receptor H must remain available as protonation evidence."""
from __future__ import annotations

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from anchor_dock.core.features import compute_atom_features
from anchor_dock.core.io import (
    RECEPTOR_PREPARATION_CONTRACT,
    clear_receptor_cache,
    extract_pocket_around_residue,
    load_receptor_context,
)
from anchor_dock.covalent.pipeline import _prepare_covalent_receptor, clear_covalent_context_cache
from anchor_dock.interaction import clear_interaction_context_cache, dock_interaction
from anchor_dock.interaction.pipeline import _INTERACTION_CONTEXT_CACHE


@pytest.fixture
def explicit_his(tmp_path):
    mol = Chem.AddHs(Chem.MolFromSmiles('[nH]1cncc1'))
    options = AllChem.ETKDGv3()
    options.randomSeed = 19
    assert AllChem.EmbedMolecule(mol, options) == 0
    names = ['ND1', 'CE1', 'NE2', 'CD2', 'CG']
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            name = 'HD1' if atom.GetNeighbors()[0].GetAtomicNum() == 7 else f'H{atom.GetIdx()}'
        else:
            name = names[atom.GetIdx()]
        atom.SetMonomerInfo(Chem.AtomPDBResidueInfo(atomName=name.rjust(4), residueName='HIS',
                                                  residueNumber=10, chainId='A'))
    path = tmp_path/'his.pdb'
    path.write_text(Chem.MolToPDBBlock(mol))
    return path, mol


def _assert_his_evidence(mol, features):
    by_name = {a.GetPDBResidueInfo().GetName().strip(): a.GetIdx() for a in mol.GetAtoms()}
    assert features['xs_types'][by_name['ND1']] == 'N_D'
    assert features['xs_types'][by_name['NE2']] == 'N_A'
    assert features['typing_ambiguities'] == ()
    h_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
    assert h_indices
    assert not features['active'][h_indices].any()
    assert mol.GetBondBetweenAtoms(by_name['ND1'], by_name['HD1']) is not None


def test_receptor_file_explicit_h_policy_and_cache(explicit_his, monkeypatch):
    path, source = explicit_his
    clear_receptor_cache()
    actual_reader = Chem.MolFromPDBFile
    calls = []

    def reader(*args, **kwargs):
        calls.append(kwargs)
        return actual_reader(*args, **kwargs)

    monkeypatch.setattr('anchor_dock.core.io.Chem.MolFromPDBFile', reader)
    context = load_receptor_context(path, 'cpu')
    assert context.mol.GetNumAtoms() == source.GetNumAtoms()
    _assert_his_evidence(context.mol, context.features)
    assert calls == [dict(sanitize=False, removeHs=False)]
    assert context.features['receptor_preparation_contract'] == RECEPTOR_PREPARATION_CONTRACT
    assert load_receptor_context(path, 'cpu') is context
    assert len(calls) == 1


def test_residue_pocket_retains_h_bond_evidence(explicit_his):
    path, _ = explicit_his
    full = load_receptor_context(path, 'cpu')
    pocket = extract_pocket_around_residue(full.mol, 'HIS10:A', cutoff=1.)
    features = compute_atom_features(pocket, 'cpu', strict=True)
    _assert_his_evidence(pocket, features)
    assert pocket.GetNumAtoms() == full.mol.GetNumAtoms()
    torch.testing.assert_close(torch.tensor(pocket.GetConformer().GetPositions()),
                               torch.tensor(full.mol.GetConformer().GetPositions()))


@pytest.mark.parametrize('multi', [False, True])
def test_interaction_pockets_preserve_explicit_h(explicit_his, tmp_path, multi):
    path, _ = explicit_his
    clear_interaction_context_cache()
    item = dict(receptor_residue='HIS10:A', receptor_atom='ND1', ligand_smarts='[#8:1]',
                target_distance=3., distance_tolerance=2.5)
    selection = dict(interactions=[item, dict(item, receptor_atom='NE2')]) if multi else item
    result = dock_interaction(path, 'CCO', tmp_path/f'out-{multi}', **selection,
                              num_confs=2, num_candidates=4, preselect_k=2, top_k=1,
                              optimize=False, scorer='vina', device='cpu', verbose=False)
    assert result['num_poses'] > 0
    contexts = list(_INTERACTION_CONTEXT_CACHE.values())
    assert len(contexts) == 1
    _assert_his_evidence(contexts[0].mol, contexts[0].features)
    assert contexts[0].features['receptor_preparation_contract'] == RECEPTOR_PREPARATION_CONTRACT


def test_covalent_receptor_preserves_explicit_h_and_heavy_anchor_selection(explicit_his):
    path, _ = explicit_his
    clear_covalent_context_cache()
    context = _prepare_covalent_receptor(path, 'HIS10:A', 12., True, 'cpu')
    _assert_his_evidence(context.receptor.mol, context.receptor.features)
    assert context.anchor.atom_name == 'NE2'
    assert context.anchor.atomic_number == 7
