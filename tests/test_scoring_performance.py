"""Numerical/gradient regression against independent pre-optimization equations.

The reference equations deliberately do not call pair_terms/_pair_term, and do
not require Git history at test time (installed/source archives work as well).
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from anchor_dock.core.scoring import (
    SCORING_CONFIGS,
    VINA_CONFIG,
    PairwiseScorer,
    ScoringConfig,
    _weighted_pair_energy,
    pair_terms,
)


def _legacy_slope(bad: float, good: float, value: torch.Tensor) -> torch.Tensor:
    if bad == good:
        return (value <= good).to(value.dtype)
    low, high = min(bad, good), max(bad, good)
    return torch.where(
        value <= low,
        torch.tensor(float(good == low), device=value.device, dtype=value.dtype),
        torch.where(
            value >= high,
            torch.tensor(float(good == high), device=value.device, dtype=value.dtype),
            (value - bad) / (good - bad),
        ),
    )


def _legacy_terms(distances, radius_sum, hydrophobic, hbond, config):
    surface = distances - radius_sum
    if config.formula == "softdock":
        overlap = torch.nn.functional.softplus(-surface / 0.20) * 0.20
        return {
            "repulsion": overlap.square(),
            "contact": torch.exp(-((surface - 0.6) / 1.6).square()),
            "hydrophobic": hydrophobic * torch.sigmoid((1.5 - surface) / 0.35),
            "hbond": hbond * torch.exp(-((surface + 0.25) / 0.65).square()),
        }
    return {
        "gauss1": torch.exp(-((surface / config.gauss1_width) ** 2)),
        "gauss2": (
            torch.zeros_like(surface)
            if config.gauss2_offset is None
            else torch.exp(-(((surface - config.gauss2_offset) / config.gauss2_width) ** 2))
        ),
        "repulsion": torch.where(surface < 0.0, surface.square(), torch.zeros_like(surface)),
        "hydrophobic": hydrophobic * _legacy_slope(config.hydrophobic_bad, config.hydrophobic_good, surface),
        "hbond": hbond * _legacy_slope(config.hbond_bad, config.hbond_good, surface),
    }


def _legacy_weighted(distances, matrices, config):
    terms = _legacy_terms(
        distances,
        matrices["radius_sum"].to(device=distances.device, dtype=distances.dtype),
        matrices["hydrophobic"].to(device=distances.device, dtype=distances.dtype),
        matrices["hbond"].to(device=distances.device, dtype=distances.dtype),
        config,
    )
    energy = torch.zeros_like(distances)
    for name, weight in config.weights.items():
        energy = energy + weight * terms[name]
    return energy.masked_fill(~(matrices["active"].to(distances.device) & (distances < config.cutoff)), 0.0)


def _broadcast_mask(mask, batch):
    if mask is None:
        return None
    mask = mask.bool()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    return mask.expand(batch, -1, -1)


def _legacy_raw(prepared, xyz):
    if xyz.ndim == 2:
        xyz = xyz.unsqueeze(0)
    receptor = prepared.receptor_coords.to(device=xyz.device, dtype=xyz.dtype)
    distances = torch.cdist(xyz, receptor.unsqueeze(0).expand(xyz.shape[0], -1, -1))
    pair_energy = _legacy_weighted(distances, prepared.intermolecular_matrices, prepared.config)
    exclusion = _broadcast_mask(prepared.intermolecular_exclusion_mask, xyz.shape[0])
    if exclusion is not None:
        pair_energy = pair_energy.masked_fill(exclusion, 0.0)
    inter = pair_energy.sum(dim=(1, 2))
    intra = torch.zeros_like(inter)
    if prepared.intramolecular_mask is not None:
        distances = torch.cdist(xyz, xyz)
        energy = _legacy_weighted(distances, prepared.intramolecular_matrices, prepared.config)
        intra = (energy * _broadcast_mask(prepared.intramolecular_mask, xyz.shape[0])).sum(dim=(1, 2)) * 0.5
    return inter, intra, inter + intra


def _features(n, generator):
    active = torch.rand(n, generator=generator) > 0.2
    active[0] = True
    return {
        "radius_vina": torch.rand(n, generator=generator) * 0.5 + 1.5,
        "radius_vinardo": torch.rand(n, generator=generator) * 0.5 + 1.5,
        "hydrophobic": (torch.rand(n, generator=generator) > 0.4).float(),
        "donor": (torch.rand(n, generator=generator) > 0.6).float(),
        "acceptor": (torch.rand(n, generator=generator) > 0.5).float(),
        "active": active,
    }


def _case(dtype, mask_kind, config, *, single=False):
    rng = torch.Generator().manual_seed(74013)
    batch, atoms, receptor_atoms = (1 if single else 3), 5, 7
    ligand, receptor = _features(atoms, rng), _features(receptor_atoms, rng)
    receptor_xyz = (torch.randn(receptor_atoms, 3, generator=rng) * 3).to(dtype).requires_grad_()
    old_xyz = torch.randn(batch, atoms, 3, generator=rng).to(dtype).requires_grad_()
    if single:
        old_xyz = old_xyz[0].detach().requires_grad_()
    new_xyz = old_xyz.detach().clone().requires_grad_()
    intra = torch.rand(batch, atoms, atoms, generator=rng) > 0.4
    diagonal = torch.arange(atoms)
    intra[:, diagonal, diagonal] = False
    exclusion = torch.rand(batch, atoms, receptor_atoms, generator=rng) > 0.8
    if mask_kind == "none":
        intra = exclusion = None
    elif mask_kind == "2d":
        intra, exclusion = intra[0], exclusion[0]
    elif mask_kind == "broadcast3d":
        intra, exclusion = intra[:1], exclusion[:1]
    prepared = PairwiseScorer(config).prepare(
        ligand, receptor_xyz, receptor, intramolecular_mask=intra,
        intermolecular_exclusion_mask=exclusion,
    )
    expected = _legacy_raw(prepared, old_xyz)
    actual_raw = prepared.raw_components(new_xyz)
    actual = (actual_raw.intermolecular, actual_raw.intramolecular, actual_raw.search_energy)
    for before, after in zip(expected, actual, strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    weighting = torch.linspace(0.5, 1.5, batch, dtype=dtype)
    old_gradient = torch.autograd.grad((expected[-1] * weighting).sum(), (old_xyz, receptor_xyz))
    new_gradient = torch.autograd.grad((actual[-1] * weighting).sum(), (new_xyz, receptor_xyz))
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    for before, after in zip(old_gradient, new_gradient, strict=True):
        torch.testing.assert_close(before, after, rtol=tolerance, atol=tolerance)
        assert torch.isfinite(after).all()


@pytest.mark.parametrize("config", list(SCORING_CONFIGS.values()), ids=list(SCORING_CONFIGS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mask_kind", ["none", "2d", "broadcast3d", "batched3d"])
@pytest.mark.parametrize("threads", [1, 2])
def test_raw_energy_and_coordinate_gradients_match_dense_equations(config, dtype, mask_kind, threads):
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        _case(dtype, mask_kind, config)
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("config", list(SCORING_CONFIGS.values()), ids=list(SCORING_CONFIGS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_single_pose_rank2_coordinates_preserve_energy_and_gradients(config, dtype):
    _case(dtype, "2d", config, single=True)


@pytest.mark.parametrize("config", list(SCORING_CONFIGS.values()), ids=list(SCORING_CONFIGS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_public_pair_terms_preserve_all_keys_values_and_endpoint_gradients(config, dtype):
    surfaces = [-0.7, -0.6, -0.35, 0.0, 0.5, 1.0, 1.5, 2.5, 3.0]
    x_old = torch.tensor(surfaces, dtype=dtype, requires_grad=True)
    x_new = x_old.detach().clone().requires_grad_()
    zeros, ones = torch.zeros_like(x_old), torch.ones_like(x_old)
    expected = _legacy_terms(x_old, zeros, ones, ones, config)
    actual = pair_terms(x_new, zeros, ones, ones, config)
    assert tuple(actual) == tuple(expected)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    old_loss, new_loss = sum(expected.values()).sum(), sum(actual.values()).sum()
    old_grad = torch.autograd.grad(old_loss, x_old)[0]
    new_grad = torch.autograd.grad(new_loss, x_new)[0]
    torch.testing.assert_close(old_grad, new_grad, rtol=0, atol=0)


@pytest.mark.parametrize("config", list(SCORING_CONFIGS.values()), ids=list(SCORING_CONFIGS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_strict_cutoff_and_inactive_pairs_have_exact_zero_value_and_gradient(config, dtype):
    delta = torch.finfo(dtype).eps * config.cutoff * 8
    distances = torch.tensor([config.cutoff - delta, config.cutoff, config.cutoff + delta, 2.0], dtype=dtype, requires_grad=True)
    matrices = {"radius_sum": torch.zeros(4, dtype=dtype), "hydrophobic": torch.ones(4, dtype=dtype),
                "hbond": torch.ones(4, dtype=dtype), "active": torch.tensor([True, True, True, False])}
    energy = _weighted_pair_energy(distances, matrices, config)
    gradient = torch.autograd.grad(energy.sum(), distances)[0]
    assert energy[0] != 0
    assert torch.equal(energy[1:], torch.zeros(3, dtype=dtype))
    assert torch.equal(gradient[1:], torch.zeros(3, dtype=dtype))
    torch.testing.assert_close(energy, _legacy_weighted(distances, matrices, config), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("weight_mode", ["reverse", "zero", "subset"])
def test_custom_weight_accumulation_order_and_zero_paths_match_legacy(dtype, weight_mode):
    weights = dict(reversed(tuple(VINA_CONFIG.weights.items())))
    if weight_mode == "zero":
        weights["gauss1"] = 0.0
        weights["hbond"] = 0.0
    elif weight_mode == "subset":
        weights = {"hydrophobic": -0.3, "gauss1": 0.0}
    _case(dtype, "batched3d", replace(VINA_CONFIG, name=f"custom-{weight_mode}", weights=weights))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_piecewise_endpoint_derivatives_remain_zero(dtype):
    for term, bad, good in (("hydrophobic", 1.5, 0.5), ("hbond", 0.0, -0.7)):
        surface = torch.tensor([bad, good], dtype=dtype, requires_grad=True)
        terms = pair_terms(surface, torch.zeros_like(surface), torch.ones_like(surface), torch.ones_like(surface), VINA_CONFIG)
        gradient = torch.autograd.grad(terms[term].sum(), surface)[0]
        assert torch.equal(gradient, torch.zeros_like(surface))


def test_unknown_weight_still_fails_closed():
    config: ScoringConfig = replace(VINA_CONFIG, weights={"unknown": 0.0})
    matrices = {"radius_sum": torch.zeros(1), "hydrophobic": torch.ones(1),
                "hbond": torch.ones(1), "active": torch.ones(1, dtype=torch.bool)}
    with pytest.raises(KeyError, match="unknown"):
        _weighted_pair_energy(torch.ones(1), matrices, config)
