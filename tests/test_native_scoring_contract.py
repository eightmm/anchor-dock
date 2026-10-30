"""Pinned native executable energies; no Meeko/Vina installation needed here."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from rdkit import Chem

from anchor_dock.core.native_vina import NATIVE_VINA_SCORE_CONTRACT, native_vina_components, native_vina_score
from anchor_dock.core.prepared import PreparedLigand
from anchor_dock.core.scoring import PairwiseScorer

ROOT = Path(__file__).resolve().parents[1]
ORACLE = json.loads((ROOT / "tests/fixtures/native_scoring.json").read_text())


def prepared(payload):
    molecule = Chem.MolFromMolBlock(payload["molblock"], removeHs=False)
    assert molecule is not None
    return PreparedLigand.from_pdbqt(
        molecule, payload["pdbqt"],
        serial_to_atom_index={int(key): value for key, value in payload["serial_to_atom_index"].items()},
    )


@pytest.mark.parametrize("ligand_name", ["secondary_amine", "branched", "10gs_ligand"])
@pytest.mark.parametrize("scorer_name", ["vina", "vinardo"])
def test_prepared_scores_against_pinned_native_oracle(ligand_name, scorer_name):
    source = ORACLE["cases"][ligand_name]
    ligand, receptor = prepared(source), prepared(ORACLE["receptor"])
    assert ORACLE["versions"]["vina"] == "1.2.7"
    assert ligand.vina_num_tors == source["vina_num_tors"]
    assert [list(pair) for pair in ligand.intramolecular_pairs] == source["intramolecular_pairs"]
    xyz = ligand.coordinates(dtype=torch.float64).unsqueeze(0)
    scorer = PairwiseScorer(scorer_name, positive_energy_cap=1000.0).prepare(
        ligand.features(), receptor.coordinates(dtype=torch.float64), receptor.features(),
        num_rotatable_bonds=ligand.vina_num_tors, intramolecular_mask=ligand.intramolecular_mask(),
    )
    expected = source["expected"][scorer_name]
    analytic = scorer.raw_components(xyz)
    # These deliberately clashing poses exercise the native cap. Analytic
    # equations do not reproduce the native r^2 lookup bins (factor=32).
    assert abs(float(analytic.intermolecular[0]) - expected["intermolecular"]) < 0.08
    assert abs(float(analytic.intramolecular[0]) - expected["intramolecular"]) < 0.003
    analytic_report = scorer.report(analytic, intramolecular_reference=0.0)
    assert abs(float(analytic_report.score[0]) - expected["total_zero_reference"]) < 0.06
    # Once the native final-energy table convention is matched, errors are
    # near input-feature precision. This independently checks pair roles,
    # native mobility exclusions, half-integral rotors and reporting together.
    tabulated = native_vina_components(scorer, xyz)
    tabulated_report = native_vina_score(scorer, xyz, intramolecular_reference=0.0)
    assert NATIVE_VINA_SCORE_CONTRACT == "vina-1.2.7-fast-r2-factor32-curl1000-prepared-v1"
    for actual, key in ((tabulated.intermolecular, "intermolecular"),
                        (tabulated.intramolecular, "intramolecular"),
                        (tabulated_report.score, "total_zero_reference")):
        assert abs(float(actual[0]) - expected[key]) < 2e-5


def _prepared_scorer(*, cap=None):
    ligand, receptor = prepared(ORACLE["cases"]["secondary_amine"]), prepared(ORACLE["receptor"])
    scorer = PairwiseScorer("vina", positive_energy_cap=cap).prepare(
        ligand.features(), receptor.coordinates(dtype=torch.float64), receptor.features(),
        num_rotatable_bonds=ligand.vina_num_tors, intramolecular_mask=ligand.intramolecular_mask())
    return scorer, ligand.coordinates(dtype=torch.float64)


def test_score_only_api_requires_explicit_detach_and_preserves_continuous_objective():
    scorer, xyz = _prepared_scorer()
    differentiable = xyz.clone().requires_grad_()
    before = scorer.search_energy(differentiable).detach()
    with pytest.raises(ValueError, match="nondifferentiable.*detach"):
        native_vina_components(scorer, differentiable)
    native = native_vina_score(scorer, differentiable.detach())
    assert not native.score.requires_grad
    torch.testing.assert_close(scorer.search_energy(differentiable).detach(), before, rtol=0, atol=0)
    assert torch.isfinite(torch.autograd.grad(scorer.search_energy(differentiable).sum(), differentiable)[0]).all()


def test_native_final_cap_is_explicitly_independent_of_search_cap():
    direct, xyz = _prepared_scorer()
    search_capped, _ = _prepared_scorer(cap=5.0)
    assert not torch.allclose(direct.search_energy(xyz), search_capped.search_energy(xyz))
    torch.testing.assert_close(native_vina_score(direct, xyz).score, native_vina_score(search_capped, xyz).score)


def test_native_score_uses_validated_features_instead_of_mutated_pair_cache():
    scorer, xyz = _prepared_scorer()
    expected = native_vina_components(scorer, xyz)
    scorer.intermolecular_matrices["radius_sum"].fill_(0.1)
    scorer.intramolecular_matrices["hydrophobic"].fill_(3.0)
    actual = native_vina_components(scorer, xyz)
    torch.testing.assert_close(actual.search_energy, expected.search_energy, rtol=0, atol=0)
    assert (scorer.intermolecular_matrices["radius_sum"] == 0.1).all()


@pytest.mark.parametrize("change, message", [
    ("inferred", "authoritative prepared"), ("weights", "official ordinary"),
    ("radius", "modified prepared"), ("roles", "modified prepared"),
    ("mask", "mask disagrees"), ("rotors", "rotor count disagrees"),
    ("exclusions", "exclusions"), ("metal", "metals"),
    ("receptor_grad", "fixed detached receptor"),
])
def test_native_score_rejects_inputs_outside_prepared_contract(change, message):
    scorer, xyz = _prepared_scorer()
    if change == "inferred":
        scorer.receptor_features["typing_version"] = "inferred-xs-v3"
    elif change == "weights":
        scorer.config = replace(scorer.config, weights={**scorer.config.weights, "gauss1": -0.04})
    elif change == "radius":
        scorer.ligand_features["radius_vina"][0] += 0.1
    elif change == "roles":
        scorer.ligand_features["donor"][0] = 1.0 - scorer.ligand_features["donor"][0]
    elif change == "mask":
        scorer.intramolecular_mask = ~scorer.intramolecular_mask
    elif change == "rotors":
        scorer.num_rotatable_bonds = 3
    elif change == "exclusions":
        scorer.intermolecular_exclusion_mask = torch.zeros(xyz.shape[0], scorer.receptor_coords.shape[0], dtype=torch.bool)
    elif change == "metal":
        scorer.receptor_features["ad4_types"] = ("Zn", *scorer.receptor_features["ad4_types"][1:])
    elif change == "receptor_grad":
        scorer.receptor_coords = scorer.receptor_coords.requires_grad_()
    with pytest.raises(ValueError, match=message):
        native_vina_components(scorer, xyz)


@pytest.mark.parametrize("reference", [float("nan"), float("inf"), True, torch.ones(2), torch.tensor(0., requires_grad=True)])
def test_native_report_rejects_invalid_reference(reference):
    scorer, xyz = _prepared_scorer()
    with pytest.raises(ValueError, match="finite detached scalar"):
        native_vina_score(scorer, xyz, intramolecular_reference=reference)


def test_native_score_batch_uses_explicit_common_reference():
    scorer, xyz = _prepared_scorer()
    batch = torch.stack((xyz, xyz + xyz.new_tensor([0.2, -0.1, 0.3])))
    together = native_vina_components(scorer, batch)
    separate = [native_vina_components(scorer, pose) for pose in batch]
    torch.testing.assert_close(together.search_energy, torch.cat([value.search_energy for value in separate]))
    expected_reference = together.intramolecular[together.search_energy.argmin()]
    report = native_vina_score(scorer, batch)
    assert report.intramolecular_reference == expected_reference
    explicit = native_vina_score(scorer, batch, intramolecular_reference=expected_reference)
    torch.testing.assert_close(report.score, explicit.score)
    # Components remain detached; full results do not accidentally imply a
    # differentiable native-search implementation.
    assert not together.search_energy.requires_grad
