"""Unit tests for forward kinematics."""

import torch
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from anchor_dock.core.kinematics import (
    get_rotation_matrix,
    get_batched_rotation_matrix,
    LigandKinematics,
    BatchedLigandKinematics,
)


class TestRotationMatrix:
    def test_identity_at_zero_angle(self):
        axis = torch.tensor([1.0, 0.0, 0.0])
        R = get_rotation_matrix(axis, torch.tensor(0.0))
        assert torch.allclose(R, torch.eye(3), atol=1e-6)

    def test_90_degree_rotation_z(self):
        axis = torch.tensor([0.0, 0.0, 1.0])
        theta = torch.tensor(torch.pi / 2)
        R = get_rotation_matrix(axis, theta)
        v = torch.tensor([1.0, 0.0, 0.0])
        rotated = R @ v
        expected = torch.tensor([0.0, 1.0, 0.0])
        assert torch.allclose(rotated, expected, atol=1e-6)

    def test_rotation_is_orthogonal(self):
        axis = torch.tensor([1.0, 1.0, 1.0])
        R = get_rotation_matrix(axis, torch.tensor(1.23))
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-6)
        assert torch.allclose(torch.linalg.det(R), torch.tensor(1.0), atol=1e-6)

    def test_zero_axis_no_nan(self):
        """P1 fix: zero axis should not produce NaN."""
        axis = torch.zeros(3)
        R = get_rotation_matrix(axis, torch.tensor(0.5))
        assert not torch.isnan(R).any()

    def test_differentiable(self):
        axis = torch.tensor([0.0, 0.0, 1.0])
        theta = torch.tensor(0.5, requires_grad=True)
        R = get_rotation_matrix(axis, theta)
        loss = R.sum()
        loss.backward()
        assert theta.grad is not None


class TestBatchedRotationMatrix:
    def test_matches_single(self):
        axes = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        thetas = torch.tensor([0.3, 0.7, 1.2])
        R_batch = get_batched_rotation_matrix(axes, thetas)
        for i in range(3):
            R_single = get_rotation_matrix(axes[i], thetas[i])
            assert torch.allclose(R_batch[i], R_single, atol=1e-6)

    def test_shape(self):
        axes = torch.randn(10, 3)
        thetas = torch.randn(10)
        R = get_batched_rotation_matrix(axes, thetas)
        assert R.shape == (10, 3, 3)


class TestLigandKinematics:
    @pytest.fixture
    def ethane_mol(self):
        """Ethane (C-C) with 3D coords - simplest molecule with a rotatable bond."""
        mol = Chem.MolFromSmiles("CC")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol = Chem.RemoveHs(mol)
        return mol

    @pytest.fixture
    def butane_mol(self):
        """Butane (C-C-C-C) - has rotatable bonds for torsion testing."""
        mol = Chem.MolFromSmiles("CCCC")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol = Chem.RemoveHs(mol)
        return mol

    def test_zero_torsion_returns_base_coords(self, butane_mol):
        coords = torch.tensor(butane_mol.GetConformer().GetPositions(), dtype=torch.float32)
        device = torch.device('cpu')
        model = LigandKinematics(butane_mol, [0], coords, device, freeze_mcs=False)
        output = model()
        assert torch.allclose(output, coords, atol=1e-5)

    def test_torsion_changes_coords(self, butane_mol):
        coords = torch.tensor(butane_mol.GetConformer().GetPositions(), dtype=torch.float32)
        device = torch.device('cpu')
        model = LigandKinematics(butane_mol, [0], coords, device, freeze_mcs=False)
        # Set a nonzero torsion angle
        with torch.no_grad():
            model.thetas.fill_(0.5)
        output = model()
        # Coords should differ from base
        assert not torch.allclose(output, coords, atol=1e-3)

    def test_freeze_mcs_keeps_ref_fixed(self, butane_mol):
        coords = torch.tensor(butane_mol.GetConformer().GetPositions(), dtype=torch.float32)
        device = torch.device('cpu')
        ref_indices = [0, 1]
        model = LigandKinematics(butane_mol, ref_indices, coords, device, freeze_mcs=True)
        with torch.no_grad():
            model.thetas.fill_(1.0)
        output = model()
        # MCS atoms (0, 1) should stay the same
        assert torch.allclose(output[0], coords[0], atol=1e-5)
        assert torch.allclose(output[1], coords[1], atol=1e-5)

    def test_differentiable(self, butane_mol):
        coords = torch.tensor(butane_mol.GetConformer().GetPositions(), dtype=torch.float32)
        device = torch.device('cpu')
        model = LigandKinematics(butane_mol, [0], coords, device, freeze_mcs=False)
        output = model()
        loss = output.sum()
        loss.backward()
        assert model.thetas.grad is not None

    def test_no_rotatable_bonds(self, ethane_mol):
        coords = torch.tensor(ethane_mol.GetConformer().GetPositions(), dtype=torch.float32)
        device = torch.device('cpu')
        # Freeze both atoms -> no torsions
        model = LigandKinematics(ethane_mol, [0, 1], coords, device, freeze_mcs=True)
        assert model.num_torsions == 0


class TestBatchedLigandKinematics:
    def test_matches_single(self):
        mol = Chem.MolFromSmiles("CCCC")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol = Chem.RemoveHs(mol)
        coords = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
        device = torch.device('cpu')
        batch_coords = coords.unsqueeze(0).expand(3, -1, -1).clone()

        single = LigandKinematics(mol, [0], coords, device, freeze_mcs=False)
        batched = BatchedLigandKinematics(mol, [0], batch_coords, device, freeze_mcs=False)

        # Zero torsions -> same output
        s_out = single()
        b_out = batched()
        for i in range(3):
            assert torch.allclose(b_out[i], s_out, atol=1e-5)
