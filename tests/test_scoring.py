from __future__ import annotations

import torch
from rdkit import Chem

from anchor_dock.core.features import compute_atom_features
from anchor_dock.core.scoring import (
    VINA_CONFIG,
    VINARDO_CONFIG,
    PairwiseScorer,
    RawScoreComponents,
    pair_terms,
)


def test_official_default_weights_and_cutoffs() -> None:
    assert VINA_CONFIG.cutoff == 8.0
    assert VINA_CONFIG.weights == {
        "gauss1": -0.035579,
        "gauss2": -0.005156,
        "repulsion": 0.840245,
        "hydrophobic": -0.035069,
        "hbond": -0.587439,
    }
    assert VINARDO_CONFIG.cutoff == 8.0
    assert VINARDO_CONFIG.weights == {
        "gauss1": -0.045,
        "gauss2": 0.0,
        "repulsion": 0.8,
        "hydrophobic": -0.035,
        "hbond": -0.600,
    }


def test_vina_piecewise_terms_at_boundaries() -> None:
    surface = torch.tensor([-0.7, -0.35, 0.0, 0.5, 1.0, 1.5])
    terms = pair_terms(surface, torch.zeros_like(surface), torch.ones_like(surface), torch.ones_like(surface), VINA_CONFIG)
    assert torch.allclose(terms["hbond"], torch.tensor([1.0, 0.5, 0.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(terms["hydrophobic"], torch.tensor([1.0, 1.0, 1.0, 1.0, 0.5, 0.0]))


def test_actual_distance_cutoff_is_hard_zero() -> None:
    features = {
        "radius_vina": torch.tensor([1.9]),
        "radius_vinardo": torch.tensor([2.0]),
        "hydrophobic": torch.tensor([1.0]),
        "donor": torch.tensor([0.0]),
        "acceptor": torch.tensor([0.0]),
        "active": torch.tensor([True]),
    }
    scorer = PairwiseScorer(VINA_CONFIG).prepare(features, torch.tensor([[0.0, 0.0, 0.0]]), features)
    near = scorer.search_energy(torch.tensor([[7.999, 0.0, 0.0]]))
    far = scorer.search_energy(torch.tensor([[8.0, 0.0, 0.0]]))
    assert near.abs().item() > 0
    assert far.item() == 0.0


def test_inferred_xs_radii_distinguish_vina_and_vinardo() -> None:
    mol = Chem.MolFromSmiles("CO")
    features = compute_atom_features(mol, "cpu")
    assert features["xs_types"] == ("C_P", "O_DA") or features["xs_types"] == ("C_P", "O_A")
    assert torch.allclose(features["radius_vina"], torch.tensor([1.9, 1.7]))
    assert torch.allclose(features["radius_vinardo"], torch.tensor([2.0, 1.6]))


def test_report_uses_common_intramolecular_baseline_and_torsion_denominator() -> None:
    features = {
        "radius_vina": torch.tensor([1.9]),
        "radius_vinardo": torch.tensor([2.0]),
        "hydrophobic": torch.tensor([0.0]),
        "donor": torch.tensor([0.0]),
        "acceptor": torch.tensor([0.0]),
        "active": torch.tensor([True]),
    }
    scorer = PairwiseScorer(VINA_CONFIG).prepare(
        features,
        torch.tensor([[0.0, 0.0, 0.0]]),
        features,
        num_rotatable_bonds=5,
    )
    raw = RawScoreComponents(
        intermolecular=torch.tensor([-4.0, -3.0]),
        intramolecular=torch.tensor([1.0, 2.0]),
        search_energy=torch.tensor([-3.0, -1.0]),
    )
    reported = scorer.report(raw, intramolecular_reference=torch.tensor(1.0))
    expected = (raw.search_energy - 1.0) / (1.0 + 0.05846 * 5)
    assert torch.allclose(reported.score, expected)


def test_pairwise_score_is_differentiable() -> None:
    mol = Chem.MolFromSmiles("CCO")
    features = compute_atom_features(mol, "cpu")
    scorer = PairwiseScorer(VINA_CONFIG).prepare(
        features,
        torch.tensor([[3.0, 0.0, 0.0]]),
        compute_atom_features(Chem.MolFromSmiles("C"), "cpu"),
    )
    coords = torch.randn(2, mol.GetNumAtoms(), 3, requires_grad=True)
    scorer.search_energy(coords).sum().backward()
    assert coords.grad is not None
    assert torch.isfinite(coords.grad).all()


def test_custom_neural_scorer_adapter() -> None:
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, ligand_coords, receptor_coords, ligand_features, receptor_features):
            del receptor_coords, ligand_features, receptor_features
            return ligand_coords.square().sum(dim=(1, 2))

    from anchor_dock.core.scoring import resolve_scorer

    ligand = Chem.MolFromSmiles("CC")
    features = compute_atom_features(ligand, "cpu")
    prepared = resolve_scorer(Model()).prepare(features, torch.zeros(1, 3), features)
    coords = torch.randn(3, 2, 3, requires_grad=True)
    values = prepared.search_energy(coords)
    assert values.shape == (3,)
    values.sum().backward()
    assert coords.grad is not None


def test_pdb_water_oxygen_is_donor_and_acceptor() -> None:
    block = (
        "HETATM    1  O   HOH A 101       0.000   0.000   0.000  1.00 20.00           O  \n"
        "END\n"
    )
    mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=True)
    features = compute_atom_features(mol, "cpu")
    assert features["donor"].tolist() == [1.0]
    assert features["acceptor"].tolist() == [1.0]
    assert features["xs_types"] == ("O_DA",)
