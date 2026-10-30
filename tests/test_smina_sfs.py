"""DKOES preset facts, independent scalar equations and coordinate derivatives."""

from __future__ import annotations

import math

import pytest
import torch

from anchor_dock.core.features import _features_from_xs
from anchor_dock.core.scoring import RawScoreComponents
from anchor_dock.core.smina import SMINA_PRESETS, SMINA_SOURCE, SminaScorer, smina_pair_terms


def features(xs=("C_H", "N_A", "O_A"), types=("C", "NA", "OA"), charges=(.3, -.4, -.2)):
    values = _features_from_xs(xs, "cpu", version="prepared-ad4-xs-v1", source="test graph")
    values.update(ad4_types=types, ad4_charges=torch.tensor(charges))
    return values


def test_source_and_preset_facts_include_unusual_native_signs():
    assert SMINA_SOURCE["commit"] == "01cae6aa6e014e0bf11e511f1fe7a61b17975855"
    assert SMINA_SOURCE["native_execution_verified"] is False
    assert SMINA_PRESETS["dkoes_fast"].hbond_weight == .387739
    assert SMINA_PRESETS["dkoes_scoring_old"].hbond_weight == .197201
    assert SMINA_PRESETS["dkoes_scoring"].hbond_weight == -.153055


def test_four_eight_well_and_finite_coincident_atom_derivative():
    r = torch.tensor([0., 2., 4., 8.], dtype=torch.float64, requires_grad=True)
    terms = smina_pair_terms(r, torch.full_like(r, 4.), torch.zeros_like(r))
    torch.testing.assert_close(terms["vdw"], r.new_tensor([100., 100., -1., 1 / 256 - 2 / 16]))
    gradient, = torch.autograd.grad(terms["vdw"].sum(), r)
    assert torch.isfinite(gradient).all()
    assert gradient[0] == gradient[1] == gradient[2] == 0


def test_dkoes_gaussian_is_not_standard_ad4_gaussian():
    r = torch.tensor([0., 3.6, 7.2], dtype=torch.float64)
    terms = smina_pair_terms(r, torch.ones_like(r), torch.zeros_like(r), solvation_amplitude=torch.full_like(r, 2.))
    torch.testing.assert_close(terms["solvation"], r.new_tensor([2., 2 * math.exp(-.25), 2 * math.exp(-1.)]))


def scalar_pair(distance, left, right, preset):
    """Separate scalar reference from the published preset equations/facts."""
    if distance >= 8. or left[0] == "H" or right[0] == "H":
        return 0.
    # (radius, donor, acceptor, solvation parameter, volume, charge)
    a, b = left[1:], right[1:]
    optimum = a[0] + b[0]
    lj = 100. if distance == 0 else min(100., (optimum / distance)**8 - 2 * (optimum / distance)**4)
    surface = distance - optimum
    hb = 0.
    if a[1] * b[2] + a[2] * b[1] > 0:
        hb = 1. if surface <= -.7 else 0. if surface >= 0 else -surface / .7
    weights = {
        "dkoes_fast": (.008962, .387739, 0.),
        "dkoes_scoring_old": (.010607, .197201, 0.),
        "dkoes_scoring": (.009900, -.153055, .048934),
    }[preset]
    desolv = ((a[3] + .01097 * abs(a[5])) * b[4] + (b[3] + .01097 * abs(b[5])) * a[4])
    desolv *= math.exp(-(distance / 7.2)**2)
    return weights[0] * lj + weights[1] * hb + weights[2] * desolv


@pytest.mark.parametrize("name", list(SMINA_PRESETS))
def test_full_raw_components_match_independent_scalar_reference(name):
    ligand = features(("C_H", "N_D", "O_A"), ("C", "N", "OA"))
    receptor = features(("O_A", "C_H", "N_A"), ("OA", "C", "NA"), (-.5, .2, -.1))
    receptor_xyz = torch.tensor([[3., 2., 0.], [-2., 4., 0.], [0., 0., 9.]], dtype=torch.float64)
    xyz = torch.tensor([[[0., 0., 0.], [1., .2, 0.], [5., 2., 0.]],
                        [[.2, -.1, .4], [1.1, .1, .3], [4.7, 2.1, .2]]], dtype=torch.float64)
    intra_mask = torch.tensor([[False, False, True], [False, False, True], [True, True, False]])
    excluded = torch.zeros(3, 3, dtype=torch.bool)
    excluded[0, 0] = True
    prepared = SminaScorer(name).prepare(ligand, receptor_xyz, receptor, intramolecular_mask=intra_mask,
                                         intermolecular_exclusion_mask=excluded)
    raw = prepared.raw_components(xyz)
    # Charges are copied from supplied tensors to include their exact input precision.
    ql, qr = ligand["ad4_charges"].tolist(), receptor["ad4_charges"].tolist()
    lig = [("C", 1.9, 0, 0, -.00143, 33.5103, ql[0]), ("N", 1.8, 1, 0, -.00162, 22.4493, ql[1]),
           ("O", 1.7, 0, 1, -.00251, 17.1573, ql[2])]
    rec = [("O", 1.7, 0, 1, -.00251, 17.1573, qr[0]), ("C", 1.9, 0, 0, -.00143, 33.5103, qr[1]),
           ("N", 1.8, 0, 1, -.00162, 22.4493, qr[2])]
    expected_inter, expected_intra = [], []
    # Input radii deliberately follow the existing FP32 feature contract.
    for index, item in enumerate(lig):
        lig[index] = (item[0], float(ligand["radius_vina"][index]), *item[2:])
    for index, item in enumerate(rec):
        rec[index] = (item[0], float(receptor["radius_vina"][index]), *item[2:])
    for pose in xyz:
        inter, intra = 0., 0.
        for i in range(3):
            for j in range(3):
                if not excluded[i, j]:
                    inter += scalar_pair(float(torch.linalg.vector_norm(pose[i] - receptor_xyz[j])), lig[i], rec[j], name)
                if i < j and intra_mask[i, j]:
                    intra += scalar_pair(float(torch.linalg.vector_norm(pose[i] - pose[j])), lig[i], lig[j], name)
        expected_inter.append(inter)
        expected_intra.append(intra)
    torch.testing.assert_close(raw.intermolecular, xyz.new_tensor(expected_inter), rtol=2e-7, atol=2e-8)
    torch.testing.assert_close(raw.intramolecular, xyz.new_tensor(expected_intra), rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize("name", list(SMINA_PRESETS))
def test_smina_coordinate_gradcheck_and_rigid_motion_invariance(name):
    ligand, receptor = features(), features(("N_D",), ("N",), (.2,))
    rec = torch.tensor([[4.2, .1, .3]], dtype=torch.float64)
    xyz = torch.tensor([[[0., .3, .1], [1.1, 1.4, .2], [2.1, .4, 1.8]]], dtype=torch.float64, requires_grad=True)
    prepared = SminaScorer(name, positive_energy_cap=1000.).prepare(ligand, rec, receptor)
    assert torch.autograd.gradcheck(prepared.search_energy, (xyz,), eps=1e-6, atol=2e-6, rtol=2e-5)
    offset = xyz.new_tensor([12., -8., 5.])
    transformed = SminaScorer(name, positive_energy_cap=1000.).prepare(ligand, rec + offset, receptor)
    torch.testing.assert_close(prepared.search_energy(xyz), transformed.search_energy(xyz + offset), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", list(SMINA_PRESETS))
def test_reporting_square_torsion_and_constant_with_common_reference(name):
    prepared = SminaScorer(name).prepare(features(), torch.zeros(3, 3), features(), num_rotatable_bonds=1.5)
    inter, intra = torch.tensor([-3., -2.]), torch.tensor([1., 4.])
    raw = RawScoreComponents(inter, intra, inter + intra)
    result = prepared.report(raw, 2.)
    weights = {"dkoes_fast": (.285035, -2.467357), "dkoes_scoring_old": (.285035, -2.585651),
               "dkoes_scoring": (.317267, -2.469020)}[name]
    expected = raw.search_energy - 2 + .1 * weights[0] * 1.5**2 / 5 + weights[1]
    torch.testing.assert_close(result.score, expected)
    assert prepared.effective_rotatable_bonds == 1.5 and prepared.torsion_penalty_applied


def test_solvation_requires_prepared_charges_and_reacts_to_receptor_charge():
    left, right = features(), features(("O_A",), ("OA",), (0.,))
    right.pop("ad4_charges")
    with pytest.raises(ValueError, match="prepared receptor AD4 charges"):
        SminaScorer("dkoes_scoring").prepare(left, torch.zeros(1, 3), right)
    xyz = torch.tensor([[[3., 1., 0.], [4., 1., 0.], [3., 2., 1.]]])
    scores = []
    for q in (0., .5, -.5):
        scorer = SminaScorer("dkoes_scoring").prepare(left, torch.zeros(1, 3), features(("O_A",), ("OA",), (q,)))
        scores.append(scorer.search_energy(xyz))
    assert scores[1] > scores[0]
    torch.testing.assert_close(scores[1], scores[2])


def test_inactive_hydrogen_cutoff_and_prepared_snapshot():
    left = features(("C_H", "H"), ("C", "HD"), (.2, .2))
    right = features(("O_A",), ("OA",), (-.3,))
    rec = torch.zeros(1, 3)
    prepared = SminaScorer("dkoes_scoring").prepare(left, rec, right)
    xyz = torch.tensor([[[8., 0., 0.], [1., 0., 0.]]], requires_grad=True)
    assert prepared.search_energy(xyz) == 0.
    prepared.search_energy(xyz).sum().backward()
    assert torch.isfinite(xyz.grad).all() and xyz.grad.abs().max() == 0
    before = prepared.search_energy(xyz.detach().sub(1.)).clone()
    left["ad4_charges"].fill_(99.)
    rec.fill_(100.)
    torch.testing.assert_close(prepared.search_energy(xyz.detach().sub(1.)), before)


def test_presets_fingerprint_cap_and_unsupported_roles_fail_explicitly():
    assert len({SminaScorer(name).fingerprint for name in SMINA_PRESETS}) == 3
    assert SminaScorer().fingerprint != SminaScorer(positive_energy_cap=1000.).fingerprint
    with pytest.raises(ValueError, match="unknown"):
        SminaScorer("chemplp")
    bad = features(("C_H",), ("C",), (0.,))
    bad["xs_types"] = ("Si",)
    with pytest.raises(ValueError, match="supported ordinary"):
        SminaScorer().prepare(bad, torch.zeros(1, 3), features(("C_H",), ("C",), (0.,)))


@pytest.mark.parametrize("value", [-1, True, math.inf, math.nan])
def test_invalid_rotor_count_fails(value):
    with pytest.raises(ValueError, match="num_rotatable_bonds"):
        SminaScorer().prepare(features(), torch.zeros(3, 3), features(), num_rotatable_bonds=value)


@pytest.mark.parametrize("name", list(SMINA_PRESETS))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_smina_cpu_cuda_energy_gradient_parity(name):
    left, right = features(), features()
    rec = torch.tensor([[3., 1., 0.], [-2., 1., 0.], [1., -2., .3]])
    cpu = SminaScorer(name).prepare(left, rec, right)
    gpu = SminaScorer(name).prepare(left, rec.cuda(), right)
    x = torch.tensor([[[0., 0., 0.], [1., 1., 0.], [2., .2, .4]]], requires_grad=True)
    y = x.detach().cuda().requires_grad_()
    torch.testing.assert_close(cpu.search_energy(x), gpu.search_energy(y).cpu(), rtol=3e-5, atol=3e-6)
    a, = torch.autograd.grad(cpu.search_energy(x).sum(), x)
    b, = torch.autograd.grad(gpu.search_energy(y).sum(), y)
    torch.testing.assert_close(a, b.cpu(), rtol=5e-5, atol=5e-6)
