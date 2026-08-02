"""Unit tests for intramolecular interaction mask."""

import torch
from rdkit import Chem

from anchor_dock.core.masks import compute_intramolecular_mask


class TestIntramolecularMask:
    def test_shape(self):
        mol = Chem.MolFromSmiles("CCCC")
        mol = Chem.AddHs(mol)
        mol = Chem.RemoveHs(mol)
        mask = compute_intramolecular_mask(mol, torch.device("cpu"))
        n = mol.GetNumAtoms()
        assert mask.shape == (n, n)

    def test_diagonal_is_false(self):
        """Self-interaction should be excluded."""
        mol = Chem.MolFromSmiles("CCCC")
        mol = Chem.AddHs(mol)
        mol = Chem.RemoveHs(mol)
        mask = compute_intramolecular_mask(mol, torch.device("cpu"))
        assert not mask.diagonal().any()

    def test_bonded_pair_excluded(self):
        """Directly bonded atoms (1-2 pairs) should be False."""
        mol = Chem.MolFromSmiles("CCCC")  # 0-1-2-3
        mol = Chem.AddHs(mol)
        mol = Chem.RemoveHs(mol)
        mask = compute_intramolecular_mask(mol, torch.device("cpu"))
        # 0-1 bonded
        assert not mask[0, 1].item()
        assert not mask[1, 0].item()

    def test_13_pair_excluded(self):
        """1-3 pairs (two bonds apart) should be False."""
        mol = Chem.MolFromSmiles("CCCC")  # 0-1-2-3
        mol = Chem.AddHs(mol)
        mol = Chem.RemoveHs(mol)
        mask = compute_intramolecular_mask(mol, torch.device("cpu"))
        # 0 and 2 are 1-3 pair
        assert not mask[0, 2].item()
        assert not mask[2, 0].item()

    def test_14_pair_excluded_and_15_pair_included(self):
        """Fixed 1-4 pairs are excluded; the first variable pair is 1-5."""
        mol = Chem.MolFromSmiles("CCCCC")  # 0-1-2-3-4
        mol = Chem.AddHs(mol)
        mol = Chem.RemoveHs(mol)
        mask = compute_intramolecular_mask(mol, torch.device("cpu"))
        assert not mask[0, 3].item()
        assert not mask[3, 0].item()
        assert mask[0, 4].item()
        assert mask[4, 0].item()

    def test_ring_atoms_excluded(self):
        """Atoms in the same ring should be excluded."""
        mol = Chem.MolFromSmiles("C1CCCCC1")  # cyclohexane
        mol = Chem.AddHs(mol)
        mol = Chem.RemoveHs(mol)
        mask = compute_intramolecular_mask(mol, torch.device("cpu"))
        # All atoms are in the same ring - all pairs should be False
        assert not mask.any()

    def test_symmetric(self):
        mol = Chem.MolFromSmiles("CCCCCCC")
        mol = Chem.AddHs(mol)
        mol = Chem.RemoveHs(mol)
        mask = compute_intramolecular_mask(mol, torch.device("cpu"))
        assert torch.equal(mask, mask.T)
