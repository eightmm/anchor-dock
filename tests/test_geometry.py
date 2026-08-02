from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from anchor_dock.core.conformers import (
    _condensed_direct_rmsd,
    _condensed_kabsch_rmsd,
    generate_conformers_and_cluster,
)
from anchor_dock.core.io import load_ligand
from anchor_dock.core.kinematics import LigandKinematics, build_kinematic_topology, get_rotation_matrix
from anchor_dock.core.masks import compute_intramolecular_mask
from anchor_dock.core.optimization import optimize_pose_module
from anchor_dock.core.topology import find_rotatable_bonds


def _embedded(smiles: str) -> tuple[Chem.Mol, torch.Tensor]:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=5) == 0
    mol = Chem.RemoveHs(mol)
    return mol, torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)


def test_zero_axis_is_identity() -> None:
    matrix = get_rotation_matrix(torch.zeros(3), torch.tensor(1.2))
    assert torch.allclose(matrix, torch.eye(3))


def test_absolute_anchor_clustering_preserves_receptor_frame() -> None:
    first = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    second = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
    coords = torch.stack([first, second])
    assert _condensed_kabsch_rmsd(coords)[0] == pytest.approx(0.0, abs=1e-5)
    assert _condensed_direct_rmsd(coords)[0] == pytest.approx(2.0**0.5 / 2.0)


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("CC(=O)NC", []),
        ("CCOC(=O)NCC", [(1, 2), (5, 6)]),
        ("CC(=O)OCC", [(3, 4)]),
        ("c1ccccc1-c2ccccc2", [(5, 6)]),
        ("C=CN(C)C", [(1, 2)]),
        ("CS(=O)(=O)NC", [(1, 4)]),
        ("CC(C)(C)CC", []),
        ("CC(C)(C)CCC", [(4, 5)]),
    ],
)
def test_rotatable_bonds_exclude_carbonyl_partial_double_bonds(
    smiles: str,
    expected: list[tuple[int, int]],
) -> None:
    assert find_rotatable_bonds(Chem.MolFromSmiles(smiles)) == expected


def test_amide_is_one_rigid_kinematic_frame() -> None:
    mol = Chem.MolFromSmiles("CC(=O)NC")
    topology = build_kinematic_topology(mol)
    assert topology["num_torsions"] == 0
    assert len(topology["frames"]) == 1


@pytest.mark.parametrize("ligand", ["CCO.[Cl-]", "[Na+].CC(=O)[O-]"])
def test_disconnected_ligands_are_not_silently_desalted(ligand: str) -> None:
    with pytest.raises(ValueError, match="exactly one connected component"):
        load_ligand(ligand)


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


def test_adamw_is_a_real_supported_optimizer() -> None:
    model = ScalarPose()
    _, stats = optimize_pose_module(
        model,
        lambda xyz: (xyz[:, 0, 0] - 1.0).square(),
        num_steps=10,
        learning_rate=0.1,
        optimizer="adamw",
        early_stopping=False,
    )
    assert stats.final_best_energy < stats.initial_best_energy


def test_pose_optimization_does_not_accumulate_scorer_gradients() -> None:
    model = ScalarPose()
    scorer_weight = nn.Parameter(torch.tensor(2.0))
    optimize_pose_module(
        model,
        lambda xyz: scorer_weight * (xyz[:, 0, 0] - 1.0).square(),
        num_steps=2,
        learning_rate=0.1,
        early_stopping=False,
    )
    assert scorer_weight.grad is None


def test_pose_optimization_rejects_energy_independent_of_pose() -> None:
    model = ScalarPose()
    scorer_bias = nn.Parameter(torch.tensor(2.0))
    with pytest.raises(RuntimeError, match="any pose parameter"):
        optimize_pose_module(
            model,
            lambda xyz: scorer_bias.expand(xyz.shape[0]),
            num_steps=2,
            learning_rate=0.1,
            early_stopping=False,
        )
    assert scorer_bias.grad is None


def test_impossible_coordinate_constraints_fail_loudly() -> None:
    molecule = Chem.MolFromSmiles("CC")
    with pytest.raises(RuntimeError, match="constraints|geometry"):
        generate_conformers_and_cluster(
            molecule,
            "cpu",
            num_confs=2,
            coord_map={0: Point3D(0.0, 0.0, 0.0), 1: Point3D(100.0, 0.0, 0.0)},
            random_seed=7,
        )


def test_stretched_bond_coordinate_constraints_fail_loudly() -> None:
    molecule = Chem.MolFromSmiles("CC")
    with pytest.raises(RuntimeError, match="constraints|geometry"):
        generate_conformers_and_cluster(
            molecule,
            "cpu",
            num_confs=2,
            coord_map={0: Point3D(0.0, 0.0, 0.0), 1: Point3D(2.99, 0.0, 0.0)},
            random_seed=7,
        )


@pytest.mark.parametrize(
    "coord_map",
    [
        {2: Point3D(0.0, 0.0, 0.0)},
        {0: Point3D(float("nan"), 0.0, 0.0)},
    ],
)
def test_invalid_coordinate_maps_fail_before_embedding(coord_map) -> None:
    with pytest.raises(ValueError, match="coordinate-map"):
        generate_conformers_and_cluster(
            Chem.MolFromSmiles("CC"),
            "cpu",
            num_confs=1,
            coord_map=coord_map,
        )


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
