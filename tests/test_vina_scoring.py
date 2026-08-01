"""Unit tests for Vina scoring functions."""

import pytest
import torch

from anchor_dock.core.scoring import precompute_interaction_matrices, vina_scoring


@pytest.fixture
def simple_features():
    """Two query atoms, three pocket atoms with known features."""
    query = {
        'vdw': torch.tensor([1.7, 1.5]),
        'hydro': torch.tensor([1.0, 0.0]),
        'hbd': torch.tensor([0.0, 1.0]),
        'hba': torch.tensor([1.0, 0.0]),
    }
    pocket = {
        'vdw': torch.tensor([1.8, 1.6, 1.5]),
        'hydro': torch.tensor([0.0, 1.0, 0.0]),
        'hbd': torch.tensor([1.0, 0.0, 0.0]),
        'hba': torch.tensor([0.0, 0.0, 1.0]),
    }
    return query, pocket


class TestPrecomputeMatrices:
    def test_shapes(self, simple_features):
        q, p = simple_features
        m = precompute_interaction_matrices(q, p, torch.device('cpu'))
        assert m['R_ij'].shape == (2, 3)
        assert m['is_hydrophobic'].shape == (2, 3)
        assert m['is_hbond'].shape == (2, 3)

    def test_hydrophobic_match(self, simple_features):
        q, p = simple_features
        m = precompute_interaction_matrices(q, p, torch.device('cpu'))
        # query[0] hydro=1, pocket[1] hydro=1 -> match
        assert m['is_hydrophobic'][0, 1].item() == 1.0
        # query[1] hydro=0 -> no match with any pocket atom
        assert m['is_hydrophobic'][1].sum().item() == 0.0

    def test_hbond_match(self, simple_features):
        q, p = simple_features
        m = precompute_interaction_matrices(q, p, torch.device('cpu'))
        # query[0] hba=1, pocket[0] hbd=1 -> match (acceptor-donor)
        assert m['is_hbond'][0, 0].item() == 1.0
        # query[1] hbd=1, pocket[2] hba=1 -> match (donor-acceptor)
        assert m['is_hbond'][1, 2].item() == 1.0

    def test_vdw_sum(self, simple_features):
        q, p = simple_features
        m = precompute_interaction_matrices(q, p, torch.device('cpu'))
        assert m['R_ij'][0, 0].item() == pytest.approx(1.7 + 1.8)
        assert m['R_ij'][1, 2].item() == pytest.approx(1.5 + 1.5)


class TestVinaScoring:
    def test_output_shape(self, simple_features):
        q, p = simple_features
        coords = torch.randn(5, 2, 3)
        pocket_coords = torch.randn(3, 3)
        scores = vina_scoring(coords, pocket_coords, q, p)
        assert scores.shape == (5,)

    def test_precomputed_matches_on_the_fly(self, simple_features):
        q, p = simple_features
        coords = torch.randn(8, 2, 3)
        pocket_coords = torch.randn(3, 3)
        matrices = precompute_interaction_matrices(q, p, torch.device('cpu'))
        s1 = vina_scoring(coords, pocket_coords, q, p)
        s2 = vina_scoring(coords, pocket_coords, q, p, precomputed_matrices=matrices)
        assert torch.allclose(s1, s2, atol=1e-6)

    def test_repulsion_at_close_range(self, simple_features):
        """Atoms very close should produce high (positive) repulsion energy."""
        q, p = simple_features
        # Place query atoms at exact same position as pocket atoms
        pocket_coords = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        close_coords = torch.tensor([[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]]) # on top of pocket
        far_coords = torch.tensor([[[100.0, 0.0, 0.0], [100.0, 5.0, 0.0]]]) # very far
        score_close = vina_scoring(close_coords, pocket_coords, q, p)
        score_far = vina_scoring(far_coords, pocket_coords, q, p)
        # Close should be higher energy (more repulsion)
        assert score_close.item() > score_far.item()

    def test_differentiable(self, simple_features):
        q, p = simple_features
        coords = torch.randn(2, 2, 3, requires_grad=True)
        pocket_coords = torch.randn(3, 3)
        scores = vina_scoring(coords, pocket_coords, q, p)
        loss = scores.sum()
        loss.backward()
        assert coords.grad is not None
        assert not torch.isnan(coords.grad).any()

    @pytest.mark.parametrize("preset", ["vina", "vina_lp", "vinardo"])
    def test_weight_presets(self, simple_features, preset):
        q, p = simple_features
        coords = torch.randn(3, 2, 3)
        pocket_coords = torch.randn(3, 3)
        scores = vina_scoring(coords, pocket_coords, q, p, weight_preset=preset)
        assert scores.shape == (3,)
        assert not torch.isnan(scores).any()

    def test_torsion_penalty_reduces_score(self, simple_features):
        q, p = simple_features
        coords = torch.randn(2, 2, 3)
        pocket_coords = torch.randn(3, 3)
        score_no_penalty = vina_scoring(coords, pocket_coords, q, p)
        score_with_penalty = vina_scoring(coords, pocket_coords, q, p, num_rotatable_bonds=5)
        # Torsion penalty divides by (1 + w*n_rot), so absolute values should be smaller
        assert torch.abs(score_with_penalty).sum() < torch.abs(score_no_penalty).sum()
