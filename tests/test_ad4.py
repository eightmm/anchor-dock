import copy

import pytest
import torch

from anchor_dock.core.ad4 import AD4GridScorer, AD4PairEnergy, AD4Scorer, AD4Weights
from anchor_dock.core.autogrid import load_autogrid_maps
from anchor_dock.core.grid import AD4IntermolecularGrid, GridBoundsError


def _pair(distance, kinds=("C", "C"), *, charges=(0., 0.), formulation="standard", smoothing=.5):
    scorer = AD4PairEnergy(kinds, charges, [(0, 1)], formulation=formulation, smoothing=smoothing, dtype=torch.float64)
    xyz = torch.tensor([[[0., 0., 0.], [distance, 0., 0.]]], dtype=torch.float64, requires_grad=True)
    return scorer, xyz


def _grid(*, curl=None, upper_exclusive=False, dtype=torch.float64):
    values = torch.stack((torch.full((5, 5, 5), 2., dtype=dtype), torch.full((5, 5, 5), 3., dtype=dtype)))
    return AD4IntermolecularGrid(values, torch.zeros_like(values[0]), torch.zeros_like(values[0]),
                                 atom_types=["C", "OA"], origin=(-5., -5., -5.), spacing=2.5,
                                 map_curl=curl, upper_exclusive=upper_exclusive)


def test_vdw_well_depth_and_smoothing_plateau():
    for distance in (3.75, 4., 4.25):
        scorer, xyz = _pair(distance)
        components = scorer.components(xyz)
        assert components.vdw.item() == pytest.approx(-.150 * .1662)
        assert components.hbond.item() == 0
        gradient, = torch.autograd.grad(components.vdw.sum(), xyz)
        assert gradient.abs().max().item() < 1e-12


@pytest.mark.parametrize("acceptor,minimum,depth", [("OA", 1.9, 5.), ("NA", 1.9, 5.), ("SA", 2.5, 1.)])
def test_polar_hydrogen_acceptor_hbond_replaces_vdw(acceptor, minimum, depth):
    scorer, xyz = _pair(minimum, ("HD", acceptor))
    result = scorer.components(xyz)
    assert result.hbond.item() == pytest.approx(-depth * .1209)
    assert result.vdw.item() == 0


def test_acceptor_acceptor_and_nonpolar_hydrogen_use_vdw():
    for kinds in (("OA", "NA"), ("H", "OA")):
        scorer, xyz = _pair(3., kinds)
        value = scorer.components(xyz)
        assert value.hbond.item() == 0
        assert value.vdw.item() != 0


@pytest.mark.parametrize("formulation", ["standard", "adgpu"])
def test_strict_component_cutoffs_and_charge_sign(formulation):
    scorer, xyz = _pair(8., charges=(.3, -.3), formulation=formulation)
    result = scorer.components(xyz)
    assert result.vdw.item() == 0
    assert result.electrostatic.item() < 0
    xyz = xyz.detach()
    xyz[0, 1, 0] = 20.48
    assert scorer(xyz).item() == 0


@pytest.mark.parametrize("formulation", ["standard", "adgpu"])
def test_pair_coordinate_gradcheck_and_rigid_motion_invariance(formulation):
    scorer, _ = _pair(3., ("C", "OA"), charges=(.2, -.4), formulation=formulation)
    xyz = torch.tensor([[[.1, .2, .4], [3.2, 1.3, -.7]]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(scorer, (xyz,), eps=1e-6, atol=2e-5)
    torch.testing.assert_close(scorer(xyz), scorer(xyz + 100))
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=xyz.dtype)
    torch.testing.assert_close(scorer(xyz), scorer(xyz @ rotation))


def test_repulsion_cap_is_standard_only_and_zero_gradient_is_finite():
    standard, xyz = _pair(.01)
    gpu, _ = _pair(.01, formulation="adgpu")
    assert standard.components(xyz).vdw.item() == pytest.approx(100000 * .1662)
    assert gpu.components(xyz).vdw.item() > 100000 * .1662
    standard, xyz = _pair(0., smoothing=0.)
    gradient, = torch.autograd.grad(standard(xyz).sum(), xyz)
    assert torch.isfinite(gradient).all()


def test_standard_electrostatic_coincidence_branch_caps_before_dielectric():
    scorer = AD4PairEnergy(["C", "C"], [.2, -.3], [(0, 1)], dtype=torch.float64,
                           weights=AD4Weights(vdw=0., hbond=0., desolvation=0.))
    xyz = torch.zeros(1, 2, 3, dtype=torch.float64, requires_grad=True)
    dielectric_zero = -8.5525 + (78.4 + 8.5525) / (1 + 7.7839)
    expected = 332 * .2 * -.3 * 100 / dielectric_zero * .1406
    assert scorer(xyz).item() == pytest.approx(expected)
    gradient, = torch.autograd.grad(scorer(xyz).sum(), xyz)
    assert torch.isfinite(gradient).all()


def test_empty_contributors_have_exact_zero_energy_and_gradient():
    scorer = AD4PairEnergy(["C", "OA"], [.1, -.1], [], dtype=torch.float64)
    xyz = torch.zeros(2, 2, 3, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(scorer(xyz), torch.zeros(2, dtype=torch.float64))
    gradient, = torch.autograd.grad(scorer(xyz).sum(), xyz)
    assert gradient.count_nonzero() == 0


@pytest.mark.parametrize("pairs", [[(1, 0)], [(0, 0)], [(0, 2)], [(0, 1), (0, 1)], [(0., 1)]])
def test_reject_invalid_contributors(pairs):
    with pytest.raises(ValueError, match="contributor"):
        AD4PairEnergy(["C", "C"], [0., 0.], pairs)


@pytest.mark.parametrize("kind", ["G0", "CG0", "W", "Cu", "B"])
def test_unsupported_extended_types_fail_closed(kind):
    with pytest.raises(ValueError, match="unsupported"):
        AD4PairEnergy([kind], [0.], [])


def test_explicit_unbound_models_and_search_report_separation():
    xyz = torch.tensor([[[0., 0., 0.], [.1, 0., 0.]]], dtype=torch.float64)
    common = dict(maps=_grid(), atom_types=["C", "OA"], charges=[0., 0.], intra_pairs=[(0, 1)], torsdof=2)
    same = AD4GridScorer(**common, unbound_model="same_as_bound").components(xyz)
    fixed = AD4GridScorer(**common, unbound_model="fixed", unbound_energy=-1.).components(xyz)
    assert same.intermolecular.item() == 5
    assert same.intramolecular.item() > 1000
    assert same.score.item() == pytest.approx(5 + .2983 * 2)
    assert fixed.score.item() == pytest.approx(fixed.search_energy.item() + .2983 * 2 + 1)
    assert same.unbound.item() == same.intramolecular.item()
    torch.testing.assert_close(same.search_energy, fixed.search_energy)


def test_vina_map_curl_applies_before_signed_charge_multiplication():
    aff = torch.ones(1, 3, 3, 3, dtype=torch.float64)
    grid = AD4IntermolecularGrid(aff, aff[0] * 2, aff[0] * .5, atom_types=["C"], origin=(0., 0., 0.),
                                 spacing=1., map_curl=1000.)
    value = grid(torch.tensor([[[.2, .3, .4]]], dtype=torch.float64), [0], torch.tensor([-.3], dtype=torch.float64))
    assert value.item() == pytest.approx(1 / 1.001 - .3 * 2 / 1.002 + .3 * .5 / 1.0005)


def test_raw_upper_exclusive_and_vina_closed_domain():
    xyz = torch.tensor([[[5., 0., 0.]]], dtype=torch.float64)
    with pytest.raises(GridBoundsError):
        _grid(upper_exclusive=True)(xyz, [0], torch.zeros(1, dtype=torch.float64))
    assert _grid()(xyz, [0], torch.zeros(1, dtype=torch.float64)).item() == 2


def _maps(tmp_path, *, value=2.):
    prefix = tmp_path / "receptor"
    header = ("GRID_PARAMETER_FILE r.gpf\nGRID_DATA_FILE r.maps.fld\nMACROMOLECULE r.pdbqt\n"
              "SPACING 2.5\nNELEMENTS 4 4 4\nCENTER 0 0 0\n")
    for kind, constant in (("C", value), ("OA", 3.), ("e", 0.), ("d", 0.)):
        (tmp_path / f"receptor.{kind}.map").write_text(header + "\n".join([str(constant)] * 125) + "\n")
    return load_autogrid_maps(prefix, ["C", "OA"], dtype=torch.float64)


def _features():
    return dict(ad4_types=("C", "OA"), ad4_charges=torch.tensor([.1, -.1]), ad4_torsdof=2,
                prepared_intramolecular_pairs=((0, 1),))


def test_engine_scorer_protocol_snapshot_fingerprint_and_report(tmp_path):
    maps = _maps(tmp_path)
    scorer = AD4Scorer(maps)
    fingerprint = scorer.fingerprint
    prepared = scorer.prepare(_features(), torch.zeros(1, 3), {})
    assert prepared.effective_rotatable_bonds == 2
    xyz = torch.tensor([[[0., 0., 0.], [3.4, .1, .2]]], dtype=torch.float64)
    raw = prepared.raw_components(xyz)
    report = prepared.report(raw, 999.)
    assert report.score.item() == pytest.approx(5 + 2 * .2983)
    assert report.intramolecular_reference.item() == pytest.approx(raw.intramolecular.item())
    before = prepared.search_energy(xyz).clone()
    maps.affinity[0].values.fill_(100.)
    assert scorer.fingerprint == fingerprint
    scorer.grid.field.values[0].add_(10.)
    assert scorer.fingerprint != fingerprint
    torch.testing.assert_close(prepared.search_energy(xyz), before)
    changed = _features()
    changed["ad4_charges"] *= 2
    assert scorer.prepare(changed, torch.zeros(1, 3), {}).fingerprint != prepared.fingerprint


def test_factory_rejects_inference_mismatched_masks_and_receptor_exclusions(tmp_path):
    scorer = AD4Scorer(_maps(tmp_path))
    with pytest.raises(ValueError, match="authoritative"):
        scorer.prepare({}, torch.zeros(1, 3), {})
    with pytest.raises(ValueError, match="contributor"):
        scorer.prepare(_features(), torch.zeros(1, 3), {}, intramolecular_mask=torch.zeros(2, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="exclusions"):
        scorer.prepare(_features(), torch.zeros(1, 3), {}, intermolecular_exclusion_mask=torch.ones(2, 1))
    mask = torch.tensor([[False, True], [True, False]])
    assert scorer.prepare(_features(), torch.zeros(1, 3), {}, intramolecular_mask=mask).name == "ad4-grid"


@pytest.mark.parametrize("kwargs", [dict(unbound_model="guess"), dict(unbound_model="fixed"),
                                    dict(unbound_model="same_as_bound", unbound_energy=0.),
                                    dict(unbound_model="fixed", unbound_energy=float("nan"))])
def test_unbound_reference_is_never_guessed(kwargs):
    with pytest.raises(ValueError, match="unbound"):
        AD4GridScorer(_grid(), ["C"], [0.], [], torsdof=0, **kwargs)


def test_fingerprint_covers_formulation_report_and_weights(tmp_path):
    maps = _maps(tmp_path)
    variants = [AD4Scorer(maps), AD4Scorer(maps, map_mode="vina"), AD4Scorer(maps, formulation="adgpu"),
                AD4Scorer(maps, unbound_model="fixed", unbound_energy=0.),
                AD4Scorer(maps, weights=AD4Weights(torsion=.4))]
    assert len({v.fingerprint for v in variants}) == len(variants)
    prepared_vina = variants[1].prepare(_features(), torch.zeros(1, 3), {})
    assert prepared_vina.module.pair_energy.pair_curl == 1000.


def test_pair_curl_is_applied_after_component_sum_with_consistent_gradients():
    raw, xyz = _pair(2.7, charges=(.2, .3))
    wrapped = AD4PairEnergy(["C", "C"], [.2, .3], [(0, 1)], dtype=torch.float64, pair_curl=1000.)
    energy = raw(xyz)
    expected = energy / (1 + energy.clamp_min(0) / 1000.)
    torch.testing.assert_close(wrapped.components(xyz).total, expected)
    assert wrapped.components(xyz).curl_correction.item() < 0
    assert torch.autograd.gradcheck(wrapped, (xyz,), eps=1e-6, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("formulation", ["standard", "adgpu"])
def test_ad4_cuda_cpu_forward_and_coordinate_gradient_parity(formulation):
    cpu = AD4GridScorer(_grid(dtype=torch.float32), ["C", "OA"], [.2, -.4], [(0, 1)], torsdof=2,
                        unbound_model="same_as_bound", formulation=formulation)
    gpu = copy.deepcopy(cpu).cuda()
    xcpu = torch.tensor([[[.1, .2, .4], [3.2, 1.3, -.7]]], requires_grad=True)
    xgpu = xcpu.detach().cuda().requires_grad_(True)
    ecpu, egpu = cpu(xcpu), gpu(xgpu)
    gcpu, = torch.autograd.grad(ecpu.sum(), xcpu)
    ggpu, = torch.autograd.grad(egpu.sum(), xgpu)
    torch.testing.assert_close(ecpu, egpu.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(gcpu, ggpu.cpu(), rtol=2e-5, atol=2e-5)


def test_native_vina_ad4_prepared_flexible_ligand_intra_and_report(tmp_path):
    """Optional independent native oracle; real prepared polar-H tree, six intra pairs.

    Native uses a radial lookup table, so analytic intra is compared at 2e-5
    tolerance. The same-as-bound affinity report should agree much tighter.
    """
    vina = pytest.importorskip("vina")
    from rdkit import Chem

    from anchor_dock.core.prepared import PreparedLigand

    text = """ROOT
ATOM      1  C   UNL     1       0.918  -0.591  -0.615  1.00  0.00    +0.028 C
ENDROOT
BRANCH   1   2
ATOM      2  C   UNL     1      -0.087   0.034   0.350  1.00  0.00    +0.002 C
BRANCH   2   3
ATOM      3  C   UNL     1      -1.471  -0.091  -0.252  1.00  0.00    -0.003 C
ATOM      4  C   UNL     1      -2.525   0.515   0.664  1.00  0.00    +0.004 C
ENDBRANCH   2   3
ENDBRANCH   1   2
BRANCH   1   5
ATOM      5  C   UNL     1       2.277  -0.438   0.036  1.00  0.00    +0.156 C
BRANCH   5   6
ATOM      6  O   UNL     1       2.485   0.936   0.208  1.00  0.00    -0.396 OA
ATOM      7  H   UNL     1       3.164   1.309  -0.406  1.00  0.00    +0.210 HD
ENDBRANCH   5   6
ENDBRANCH   1   5
TORSDOF 4
"""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCCCO"), onlyOnAtoms=[5])
    mapping = {1: 3, 2: 2, 3: 1, 4: 0, 5: 4, 6: 5, 7: 6}
    conf = Chem.Conformer(mol.GetNumAtoms())
    for line in text.splitlines():
        if line.startswith("ATOM"):
            index = mapping[int(line[6:11])]
            conf.SetAtomPosition(index, tuple(float(line[start:start + 8]) for start in (30, 38, 46)))
    mol.AddConformer(conf)
    data = PreparedLigand.from_pdbqt(mol, text, serial_to_atom_index=mapping)
    assert len(data.intramolecular_pairs) == 6
    prefix = tmp_path / "native"
    header = ("GRID_PARAMETER_FILE r.gpf\nGRID_DATA_FILE r.maps.fld\nMACROMOLECULE r.pdbqt\n"
              "SPACING 1\nNELEMENTS 20 20 20\nCENTER 0 0 0\n")
    # Nonzero constant maps check map weighting/report while retaining a
    # nonempty independent native intramolecular calculation.
    for kind, constant in (("C", 1.), ("OA", 2.), ("HD", .3), ("e", 2.), ("d", .5)):
        (tmp_path / f"native.{kind}.map").write_text(header + f"{constant}\n" * 21 ** 3)
    native = vina.Vina(sf_name="ad4", cpu=1, verbosity=0)
    native.set_ligand_from_string(text)
    native.load_maps(str(prefix))
    exact = native._vina.score()
    scorer = AD4Scorer.from_autogrid(prefix, ["C", "OA", "HD"], dtype=torch.float64, map_mode="vina")
    features = data.features()
    # The prepared object deliberately stores float32 engine features; the
    # oracle uses exact decimal input charges for the FP64 comparison.
    features["ad4_charges"] = torch.tensor(data.charges, dtype=torch.float64)
    own = scorer.prepare(features, torch.zeros(1, 3), {}, intramolecular_mask=data.intramolecular_mask())
    result = own.module.components(data.coordinates(dtype=torch.float64))
    assert result.intermolecular.item() == pytest.approx(exact[1], abs=1e-10)
    assert result.intramolecular.item() == pytest.approx(exact[5], abs=2e-5)
    assert result.score.item() == pytest.approx(exact[0], abs=1e-10)
