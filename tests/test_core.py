from __future__ import annotations

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from anchor_dock.core.features import compute_vina_features
from anchor_dock.core.kinematics import (
    BatchedLigandKinematics,
    LigandKinematics,
    build_kinematic_topology,
    get_rotation_matrix,
)
from anchor_dock.core.masks import compute_intramolecular_mask
from anchor_dock.core.scoring import precompute_interaction_matrices, vina_scoring


def _embedded(smiles: str) -> tuple[Chem.Mol, torch.Tensor]:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    mol = Chem.RemoveHs(mol)
    coords = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
    return mol, coords


def test_scoring_shapes_gradients_and_masks() -> None:
    ligand, ligand_coords = _embedded("CCO")
    pocket, pocket_coords = _embedded("CCN")
    qf = compute_vina_features(ligand, "cpu")
    pf = compute_vina_features(pocket, "cpu")
    poses = torch.stack((ligand_coords, ligand_coords + 0.2)).requires_grad_(True)
    precomputed = precompute_interaction_matrices(qf, pf, "cpu")
    scores = vina_scoring(
        poses,
        pocket_coords,
        qf,
        pf,
        intramolecular_mask=compute_intramolecular_mask(ligand, "cpu"),
        precomputed_matrices=precomputed,
    )
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
    scores.sum().backward()
    assert poses.grad is not None and torch.isfinite(poses.grad).all()

    all_excluded = torch.ones(1, ligand.GetNumAtoms(), pocket.GetNumAtoms(), dtype=torch.bool)
    excluded_scores = vina_scoring(
        poses.detach(), pocket_coords, qf, pf,
        precomputed_matrices=precomputed,
        intermolecular_exclusion_mask=all_excluded,
    )
    assert torch.allclose(excluded_scores, torch.zeros_like(excluded_scores))


def test_kinematics_single_batch_and_legacy_topology() -> None:
    mol, coords = _embedded("CCCC")
    topology = build_kinematic_topology(mol, [0], freeze_anchor=True)
    assert topology["num_torsions"] >= 1
    assert len(topology["child_frames"]) == topology["num_torsions"]

    single = LigandKinematics(mol, [0], coords, "cpu")
    batched = LigandKinematics(mol, [0], torch.stack((coords, coords)), "cpu")
    legacy = BatchedLigandKinematics(mol, [0], torch.stack((coords, coords)), "cpu")
    assert single().shape == coords.shape
    assert batched().shape == (2, *coords.shape)
    assert legacy().shape == (2, *coords.shape)
    assert torch.allclose(single()[0], coords[0])


def test_rotation_matrix_handles_zero_axis() -> None:
    rotation = get_rotation_matrix(torch.zeros(3), torch.tensor(1.2))
    assert torch.isfinite(rotation).all()
