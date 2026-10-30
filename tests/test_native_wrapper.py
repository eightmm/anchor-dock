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


def test_softdock_cannot_claim_vina_wrapper():
    with pytest.raises(ValueError, match="only for Vina"):
        PairwiseScorer("softdock", positive_energy_cap=1000.)
