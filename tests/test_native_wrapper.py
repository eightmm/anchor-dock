"""Prepared/native reporting choices must not silently alter default semantics."""

import math

import pytest
import torch

from anchor_dock.core.scoring import PairwiseScorer, RawScoreComponents, positive_energy_curl


def _features(atoms):
    return {
        "radius_vina": torch.full((atoms,), 1.9),
        "radius_vinardo": torch.full((atoms,), 2.0),
        "hydrophobic": torch.zeros(atoms), "donor": torch.zeros(atoms), "acceptor": torch.zeros(atoms),
        "active": torch.ones(atoms, dtype=torch.bool),
    }


def test_wrapper_forward_and_derivative_include_negative_singularity():
    x = torch.tensor([-1000., -2., 0., 2., 1000.], dtype=torch.float64, requires_grad=True)
    values = positive_energy_curl(x, 1000.)
    assert torch.allclose(values, torch.tensor([-1000., -2., 0., 2 / 1.002, 500.], dtype=x.dtype))
    grad, = torch.autograd.grad(values.sum(), x)
    assert torch.allclose(grad, torch.tensor([1., 1., 1., 1 / 1.002**2, .25], dtype=x.dtype))
    assert torch.autograd.gradcheck(lambda v: positive_energy_curl(v, 1000.), (x,))


def test_inter_wrapper_applies_after_receptor_atom_sum():
    coords = torch.tensor([[[.2, .3, .4]]])
    receptor = torch.tensor([[0., 0., 0.], [1., 0., 0.]])
    direct = PairwiseScorer("vina").prepare(_features(1), receptor, _features(2))
    wrapped = PairwiseScorer("vina", positive_energy_cap=1000.).prepare(_features(1), receptor, _features(2))
    raw = direct.raw_components(coords).intermolecular
    assert raw.item() > 0
    assert torch.allclose(wrapped.raw_components(coords).intermolecular, positive_energy_curl(raw, 1000.))
    assert direct.fingerprint != wrapped.fingerprint


def test_native_half_torsion_reporting_is_preserved():
    prepared = PairwiseScorer("vina").prepare(_features(1), torch.zeros(1, 3), _features(1), num_rotatable_bonds=.5)
    zero, e = torch.zeros(1), torch.tensor([-2.])
    assert prepared.effective_rotatable_bonds == .5
    assert torch.allclose(prepared.report(RawScoreComponents(e, zero, e)).score, e / (1 + .05846 * .5))


@pytest.mark.parametrize("value", [-1, math.nan, math.inf, True])
def test_invalid_torsion_count_fails(value):
    with pytest.raises(ValueError, match="num_rotatable_bonds"):
        PairwiseScorer("vina").prepare(_features(1), torch.zeros(1, 3), _features(1), num_rotatable_bonds=value)


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, True])
def test_invalid_curl_fails(value):
    with pytest.raises(ValueError, match="positive_energy_cap"):
        PairwiseScorer("vina", positive_energy_cap=value)


def test_removed_softdock_is_not_silently_aliased_to_vina():
    from anchor_dock.core.scoring import SCORING_CONFIGS, resolve_scorer

    assert "softdock" not in SCORING_CONFIGS
    with pytest.raises(ValueError, match="softdock was removed"):
        PairwiseScorer("softdock")
    with pytest.raises(ValueError, match="softdock was removed"):
        resolve_scorer("softdock")


@pytest.mark.parametrize("name", ["vina", "vinardo"])
def test_factory_configuration_and_prepared_weights_have_independent_lifetimes(name):
    from dataclasses import replace

    from anchor_dock.core.scoring import SCORING_CONFIGS

    source = replace(SCORING_CONFIGS[name], weights=dict(SCORING_CONFIGS[name].weights))
    factory = PairwiseScorer(source)
    builtin_weight = SCORING_CONFIGS[name].weights["repulsion"]
    receptor = torch.zeros(1, 3)
    coordinates = torch.tensor([[[.2, .3, .4]]])
    prepared = factory.prepare(_features(1), receptor, _features(1))
    energy, identity = prepared.search_energy(coordinates).clone(), prepared.fingerprint
    source.weights["repulsion"] *= 7
    assert factory.config.weights["repulsion"] == builtin_weight
    assert factory.fingerprint == identity
    torch.testing.assert_close(prepared.search_energy(coordinates), energy, rtol=0, atol=0)

    factory.config.weights["repulsion"] *= 2
    assert factory.fingerprint != identity
    assert prepared.fingerprint == identity
    torch.testing.assert_close(prepared.search_energy(coordinates), energy, rtol=0, atol=0)
    subsequent = factory.prepare(_features(1), receptor, _features(1))
    assert subsequent.fingerprint == factory.fingerprint
    assert subsequent.search_energy(coordinates).item() > energy.item()
    assert SCORING_CONFIGS[name].weights["repulsion"] == builtin_weight
    assert PairwiseScorer(name).config.weights["repulsion"] == builtin_weight


@pytest.mark.parametrize("mode", ["dense", "sparse"])
@pytest.mark.parametrize("name", ["vina", "vinardo"])
def test_preparation_snapshots_caller_coordinates_features_and_masks(name, mode):
    ligand, receptor = _features(2), _features(2)
    ligand["xs_types"] = ["C_H", "C_H"]
    ligand["extra"] = {"source": ["original"], "tensor": torch.tensor([1.])}
    receptor_coordinates = torch.tensor([[0., 0., 0.], [1., 0., 0.]])
    coordinates = torch.tensor([[[.2, .3, .4], [2.3, .4, -.1]]])
    intra = ~torch.eye(2, dtype=torch.bool)
    exclusions = torch.zeros(2, 2, dtype=torch.bool)
    prepared = PairwiseScorer(name, intra_mode=mode).prepare(
        ligand, receptor_coordinates, receptor, intramolecular_mask=intra,
        intermolecular_exclusion_mask=exclusions,
    )
    energy, identity = prepared.search_energy(coordinates).clone(), prepared.fingerprint
    receptor_coordinates.add_(100)
    intra.fill_(False)
    exclusions.fill_(True)
    ligand["radius_vina"].fill_(100)
    ligand["xs_types"][0] = "X"
    ligand["extra"]["source"][0] = "changed"
    ligand["extra"]["tensor"].fill_(5)
    receptor["active"].fill_(False)
    assert prepared.fingerprint == identity
    assert prepared.ligand_features["xs_types"] == ["C_H", "C_H"]
    assert prepared.ligand_features["extra"]["source"] == ["original"]
    assert prepared.ligand_features["extra"]["tensor"].item() == 1
    assert prepared.receptor_features["active"].all()
    assert prepared.intramolecular_mask.sum() == 2
    assert not prepared.intermolecular_exclusion_mask.any()
    torch.testing.assert_close(prepared.search_energy(coordinates), energy, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["vina", "vinardo"])
def test_preparation_snapshot_keeps_coordinate_and_feature_autograd(name):
    receptor = torch.tensor([[0., 0., 0.]], dtype=torch.float64, requires_grad=True)
    coordinates = torch.tensor([[[.2, .3, .4]]], dtype=torch.float64, requires_grad=True)
    ligand_features, receptor_features = _features(1), _features(1)
    radius_key = "radius_vina" if name == "vina" else "radius_vinardo"
    radius = ligand_features[radius_key].double().requires_grad_()
    ligand_features[radius_key] = radius
    prepared = PairwiseScorer(name).prepare(ligand_features, receptor, receptor_features)
    energy = prepared.search_energy(coordinates).sum()
    ligand_gradient, receptor_gradient, radius_gradient = torch.autograd.grad(energy, (coordinates, receptor, radius))
    assert torch.isfinite(ligand_gradient).all() and torch.isfinite(receptor_gradient).all()
    assert radius_gradient.abs().item() > 0
    torch.testing.assert_close(ligand_gradient.sum((0, 1)), -receptor_gradient.sum(0), atol=1e-12, rtol=1e-12)


def test_weight_reduction_order_is_part_of_factory_and_config_identity():
    from dataclasses import replace

    from anchor_dock.core.scoring import VINA_CONFIG, pair_terms, scoring_config_fingerprint

    features = {key: value.double() if value.is_floating_point() else value for key, value in _features(1).items()}
    features["radius_vina"].fill_(1.9)
    coordinates = torch.tensor([[[3.55, 0., 0.]]], dtype=torch.float64)
    terms = pair_terms(torch.tensor(3.55, dtype=torch.float64), torch.tensor(3.8, dtype=torch.float64),
                       torch.tensor(0.), torch.tensor(0.), VINA_CONFIG)
    # Equal mappings can have different insertion orders. Strong cancellation
    # makes their distinct floating-point reductions visible (0 versus 1).
    weights = {"gauss1": 1e16 / terms["gauss1"].item(), "gauss2": 1 / terms["gauss2"].item(),
               "repulsion": -1e16 / terms["repulsion"].item()}
    first = replace(VINA_CONFIG, weights=weights)
    second = replace(VINA_CONFIG, weights={key: weights[key] for key in ("gauss1", "repulsion", "gauss2")})
    assert first == second
    models = [PairwiseScorer(config) for config in (first, second)]
    assert models[0].fingerprint != models[1].fingerprint
    assert scoring_config_fingerprint(first) != scoring_config_fingerprint(second)
    energies = [model.prepare(features, torch.zeros(1, 3, dtype=torch.float64), features).search_energy(coordinates).item()
                for model in models]
    assert energies == [0., 1.]
