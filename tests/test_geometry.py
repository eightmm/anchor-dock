from __future__ import annotations

import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem

from anchor_dock.core.kinematics import LigandKinematics, build_kinematic_topology, get_rotation_matrix
from anchor_dock.core.masks import compute_intramolecular_mask
from anchor_dock.core.optimization import optimize_pose_module


def _embedded(smiles: str) -> tuple[Chem.Mol, torch.Tensor]:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=5) == 0
    mol = Chem.RemoveHs(mol)
    return mol, torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)


def test_zero_axis_is_identity() -> None:
    matrix = get_rotation_matrix(torch.zeros(3), torch.tensor(1.2))
    assert torch.allclose(matrix, torch.eye(3))


def test_distributed_anchors_disable_intervening_torsion() -> None:
    mol, coords = _embedded("CCCC")
    topology = build_kinematic_topology(mol, [0, 3], freeze_anchor=True)
    assert topology["num_torsions"] == 0
    assert topology["disabled_torsions"]
    model = LigandKinematics(mol, [0, 3], coords, "cpu", freeze_anchor=True)
    assert torch.allclose(model(), coords)


def test_single_side_anchor_remains_fixed_while_branch_moves() -> None:
    mol, coords = _embedded("CCCCC")
    model = LigandKinematics(mol, [0], coords, "cpu", freeze_anchor=True)
    with torch.no_grad():
        model.thetas.fill_(0.8)
    output = model()
    assert torch.allclose(output[0], coords[0], atol=1e-6)
    assert not torch.allclose(output[-1], coords[-1], atol=1e-3)


def test_intramolecular_mask_excludes_1_4_and_rigid_pairs() -> None:
    mol = Chem.MolFromSmiles("CCCCC")
    mask = compute_intramolecular_mask(mol, "cpu")
    assert not mask[0, 1]
    assert not mask[0, 2]
    assert not mask[0, 3]
    assert mask[0, 4]


class ScalarPose(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor([[5.0]]))

    def forward(self) -> torch.Tensor:
        xyz = torch.zeros(1, 1, 3)
        xyz[:, :, 0] = self.value
        return xyz


def test_optimizer_restores_best_state() -> None:
    model = ScalarPose()
    coords, stats = optimize_pose_module(
        model,
        lambda xyz: (xyz[:, 0, 0] - 1.0).square(),
        num_steps=8,
        learning_rate=4.0,
        optimizer="adam",
        early_stopping=False,
    )
    assert stats.final_best_energy <= stats.initial_best_energy
    assert float((coords[:, 0, 0] - 1.0).abs().detach()) < 4.0


def test_zero_torsion_optimization_is_a_noop() -> None:
    from anchor_dock.core.features import compute_atom_features
    from anchor_dock.core.optimization import optimize_torsions
    from anchor_dock.core.scoring import PairwiseScorer

    mol, coords = _embedded("CC")
    features = compute_atom_features(mol, "cpu")
    scorer = PairwiseScorer("softdock").prepare(
        features,
        torch.tensor([[10.0, 0.0, 0.0]]),
        compute_atom_features(Chem.MolFromSmiles("C"), "cpu"),
    )
    output, stats = optimize_torsions(mol, [0, 1], coords, scorer, "cpu", num_steps=5)
    assert torch.allclose(output, coords)
    assert stats.maximum_steps == 0
