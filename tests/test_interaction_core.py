"""Focused tests for the generic SE(3) pivot-pose substrate used by interaction docking."""

from __future__ import annotations

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from anchor_dock.core.engine import DockingEngine, PreparedDockingProblem
from anchor_dock.core.features import compute_atom_features
from anchor_dock.core.geometry import sample_uniform_rotation_vectors
from anchor_dock.core.optimization import SE3PoseModel, optimize_pose_module
from anchor_dock.core.scoring import PairwiseScorer


def _embedded(smiles: str, seed: int = 5) -> tuple[Chem.Mol, torch.Tensor]:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    mol = Chem.RemoveHs(mol)
    return mol, torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)


def _prepared_scorer():
    mol, _ = _embedded("CCO")
    features = compute_atom_features(mol, "cpu")
    scorer = PairwiseScorer("softdock").prepare(
        features,
        torch.tensor([[10.0, 0.0, 0.0]]),
        compute_atom_features(Chem.MolFromSmiles("C"), "cpu"),
    )
    return mol, scorer


# --- SE3PoseModel: pivot stationarity ---------------------------------------------------------


def test_pivot_remains_exactly_at_center_through_nonzero_torsions_and_rotation() -> None:
    mol, coords = _embedded("CCCCC")
    base_coords = coords.unsqueeze(0).repeat(2, 1, 1)
    pivot_atom_index = 2
    centers = torch.tensor([[1.0, 2.0, 3.0], [-4.0, 0.5, 7.0]])
    rotation_vectors = torch.tensor([[0.3, 0.0, 0.0], [0.0, 0.6, -0.2]])
    model = SE3PoseModel(mol, base_coords, pivot_atom_index, centers, rotation_vectors, "cpu")
    with torch.no_grad():
        model.kinematics.thetas.fill_(0.7)

    output = model()

    assert torch.allclose(output[:, pivot_atom_index, :], centers, atol=1e-5)


def test_pivot_follows_only_the_trainable_translation_after_gradient_steps() -> None:
    mol, coords = _embedded("CCCCC")
    base_coords = coords.unsqueeze(0)
    pivot_atom_index = 1
    centers = torch.tensor([[2.0, -1.0, 0.5]])
    rotation_vectors = torch.tensor([[0.1, 0.2, 0.0]])
    model = SE3PoseModel(mol, base_coords, pivot_atom_index, centers, rotation_vectors, "cpu")

    optimized, _ = optimize_pose_module(
        model,
        lambda values: values[:, 0, :].square().sum(dim=1),
        num_steps=5,
        learning_rate=0.1,
        early_stopping=False,
    )

    assert not torch.allclose(model.translations.detach(), centers)
    assert torch.allclose(
        optimized[:, pivot_atom_index, :],
        model.translations.detach(),
        atol=1e-5,
    )


# --- SE3PoseModel: validation -------------------------------------------------------------------


def test_se3_pose_model_rejects_out_of_range_pivot() -> None:
    mol, coords = _embedded("CCO")
    base_coords = coords.unsqueeze(0)
    with pytest.raises(ValueError, match="pivot_atom_index"):
        SE3PoseModel(mol, base_coords, base_coords.shape[1], torch.zeros(1, 3), torch.zeros(1, 3), "cpu")


def test_se3_pose_model_rejects_bad_shapes() -> None:
    mol, coords = _embedded("CCO")
    base_coords = coords.unsqueeze(0)
    with pytest.raises(ValueError, match="base_coords"):
        SE3PoseModel(mol, coords, 0, torch.zeros(1, 3), torch.zeros(1, 3), "cpu")
    with pytest.raises(ValueError, match="initial_centers"):
        SE3PoseModel(mol, base_coords, 0, torch.zeros(1, 2), torch.zeros(1, 3), "cpu")
    with pytest.raises(ValueError, match="initial_rotation_vectors"):
        SE3PoseModel(mol, base_coords, 0, torch.zeros(1, 3), torch.zeros(2, 3), "cpu")


def test_se3_pose_model_rejects_non_finite_inputs() -> None:
    mol, coords = _embedded("CCO")
    base_coords = coords.unsqueeze(0)
    bad_centers = torch.tensor([[float("nan"), 0.0, 0.0]])
    with pytest.raises(ValueError, match="finite"):
        SE3PoseModel(mol, base_coords, 0, bad_centers, torch.zeros(1, 3), "cpu")
    bad_coords = base_coords.clone()
    bad_coords[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        SE3PoseModel(mol, bad_coords, 0, torch.zeros(1, 3), torch.zeros(1, 3), "cpu")


# --- sample_uniform_rotation_vectors --------------------------------------------------------------


def test_sample_uniform_rotation_vectors_rejects_non_positive_count() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    with pytest.raises(ValueError, match="count"):
        sample_uniform_rotation_vectors(0, generator)


def test_sample_uniform_rotation_vectors_is_seeded_and_deterministic() -> None:
    generator_a = torch.Generator(device="cpu")
    generator_a.manual_seed(123)
    first = sample_uniform_rotation_vectors(16, generator_a)

    generator_b = torch.Generator(device="cpu")
    generator_b.manual_seed(123)
    second = sample_uniform_rotation_vectors(16, generator_b)

    assert torch.allclose(first, second)


def test_sample_uniform_rotation_vectors_matches_haar_mean_angle_regression() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    vectors = sample_uniform_rotation_vectors(20000, generator)
    angles = torch.linalg.vector_norm(vectors, dim=1)
    assert float(angles.min()) >= 0.0
    assert float(angles.max()) <= torch.pi + 1e-5
    # Analytic mean angle for Haar-uniform SO(3) rotations: (pi**2 / 2 + 2) / pi.
    expected_mean = (torch.pi**2 / 2 + 2) / torch.pi
    assert float(angles.mean()) == pytest.approx(expected_mean, abs=0.05)


# --- injected differentiable energy has a live translation gradient ----------------------------


def test_flat_bottom_energy_has_a_live_translation_gradient() -> None:
    mol, coords = _embedded("CCO")
    base_coords = coords.unsqueeze(0)
    pivot_atom_index = 0
    centers = torch.tensor([[10.0, 0.0, 0.0]])
    rotation_vectors = torch.zeros(1, 3)
    model = SE3PoseModel(mol, base_coords, pivot_atom_index, centers, rotation_vectors, "cpu")

    def flat_bottom_energy(values: torch.Tensor) -> torch.Tensor:
        distance = torch.linalg.vector_norm(values[:, pivot_atom_index, :], dim=1)
        excess = torch.relu(distance - 3.5) + torch.relu(2.5 - distance)
        return excess.square()

    loss = flat_bottom_energy(model()).sum()
    loss.backward()

    assert model.translations.grad is not None
    assert torch.any(model.translations.grad.abs() > 0)


# --- guide then release over the same live model parameters ------------------------------------


def test_guide_then_release_reuse_same_model_and_rotation_stays_trainable() -> None:
    mol, coords = _embedded("CCO")
    base_coords = coords.unsqueeze(0)
    centers = torch.tensor([[10.0, 0.0, 0.0]])
    rotation_vectors = torch.tensor([[1e-3, 0.0, 0.0]])
    model = SE3PoseModel(mol, base_coords, 0, centers, rotation_vectors, "cpu")
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())

    def restraint_energy(values: torch.Tensor) -> torch.Tensor:
        distance = torch.linalg.vector_norm(values[:, 0, :], dim=1)
        return (distance - 5.0).square()

    guided, guide_stats = optimize_pose_module(
        model, restraint_energy, num_steps=5, learning_rate=0.05, early_stopping=False
    )
    assert guide_stats.num_poses == 1

    def release_energy(values: torch.Tensor) -> torch.Tensor:
        target = torch.tensor([10.0, 2.0, 1.0], dtype=values.dtype, device=values.device)
        return (values[:, 1, :] - target).square().sum(dim=1)

    released, release_stats = optimize_pose_module(
        model, release_energy, num_steps=5, learning_rate=0.05, early_stopping=False
    )
    assert release_stats.num_poses == 1
    assert released.shape == guided.shape
    assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids

    model.zero_grad(set_to_none=True)
    loss = release_energy(model()).sum()
    loss.backward()
    assert model.rotation_vectors.requires_grad
    assert model.rotation_vectors.grad is not None
    assert torch.any(model.rotation_vectors.grad.abs() > 0)


# --- DockingEngine.optimize_se3 --------------------------------------------------------------


def test_optimize_se3_returns_expected_shapes_and_stats() -> None:
    num_poses = 3
    mol, base = _embedded("CCO")
    _, scorer = _prepared_scorer()
    problem = PreparedDockingProblem(mol=mol, receptor=None, scorer=scorer, anchor_indices=(), num_rotatable_bonds=0)
    base_coords = base.unsqueeze(0).repeat(num_poses, 1, 1)
    centers = torch.tensor([[10.0, 0.0, 0.0], [10.0, 1.0, 0.0], [10.0, 0.0, 1.0]])
    rotation_vectors = torch.zeros(num_poses, 3)

    engine = DockingEngine("softdock", device="cpu", optimizer="adam", num_steps=4, learning_rate=0.05, batch_size=2)
    guided, final, guide_stats, release_stats = engine.optimize_se3(
        problem,
        base_coords,
        0,
        centers=centers,
        rotation_vectors=rotation_vectors,
        release_steps=3,
    )

    assert guided.shape == base_coords.shape
    assert final.shape == base_coords.shape
    assert guide_stats.num_poses == num_poses
    assert release_stats.num_poses == num_poses


def test_optimize_se3_without_release_leaves_final_equal_to_guided_and_zero_stats() -> None:
    mol, base = _embedded("CCO")
    _, scorer = _prepared_scorer()
    problem = PreparedDockingProblem(mol=mol, receptor=None, scorer=scorer, anchor_indices=(), num_rotatable_bonds=0)
    base_coords = base.unsqueeze(0)
    centers = torch.tensor([[10.0, 0.0, 0.0]])
    rotation_vectors = torch.zeros(1, 3)

    engine = DockingEngine("softdock", device="cpu", optimizer="adam", num_steps=4, learning_rate=0.05, batch_size=4)
    guided, final, _, release_stats = engine.optimize_se3(
        problem,
        base_coords,
        0,
        centers=centers,
        rotation_vectors=rotation_vectors,
        release_steps=0,
    )

    assert torch.allclose(guided, final)
    assert release_stats.maximum_steps == 0
    assert release_stats.average_steps == 0.0


def test_optimize_se3_rejects_negative_release_steps() -> None:
    mol, base = _embedded("CCO")
    _, scorer = _prepared_scorer()
    problem = PreparedDockingProblem(mol=mol, receptor=None, scorer=scorer, anchor_indices=(), num_rotatable_bonds=0)
    base_coords = base.unsqueeze(0)
    engine = DockingEngine("softdock", device="cpu", num_steps=1, batch_size=4)
    with pytest.raises(ValueError, match="release_steps"):
        engine.optimize_se3(
            problem,
            base_coords,
            0,
            centers=torch.zeros(1, 3),
            rotation_vectors=torch.zeros(1, 3),
            release_steps=-1,
        )
