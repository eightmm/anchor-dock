"""Optional OpenMM executable oracle: force units, ordering and fixed context."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from rdkit import Chem

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

from anchor_dock.core.amber import AMBER_SCORING_CONTRACT, OpenMMScorer  # noqa: E402


def system_fixture():
    system = openmm.System()
    for mass in (12., 16., 12.):
        system.addParticle(mass)
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    for charge, sigma, epsilon in ((.3, .31, .2), (.2, .30, .4), (-.5, .32, .3)):
        nonbonded.addParticle(charge, sigma, epsilon)
    nonbonded.addException(1, 2, 0., .3, 0.)
    system.addForce(nonbonded)
    bonds = openmm.HarmonicBondForce()
    bonds.addBond(1, 2, .15, 500.)
    system.addForce(bonds)
    constant = openmm.CustomExternalForce("5*(x*x+y*y+z*z)")
    constant.addParticle(0)
    system.addForce(constant)
    xyz = np.array([[.2, -.1, .1], [5.5, .4, .2], [4., .4, .2]])
    return system, xyz


@pytest.fixture
def scorer():
    system, xyz = system_fixture()
    with OpenMMScorer.from_system(system, xyz, [2, 1], atomic_numbers=[6, 8, 6]) as value:
        yield value


def native_value_and_gradient(system, coordinates, indices):
    integrator = openmm.VerletIntegrator(.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(coordinates * unit.angstrom)
    state = context.getState(getEnergy=True, getForces=True)
    value = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) / 4.184
    # Explicit conversion independently tests the bridge's unit-based path.
    gradient = -state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)[indices] / 41.84
    del context, integrator
    return value, gradient


def test_amber_energy_gradient_units_and_nontrivial_atom_order(scorer):
    xyz = scorer.ligand_coordinates(dtype=torch.float64).requires_grad_()
    energy = scorer.energy(xyz)
    gradient, = torch.autograd.grad(energy.sum(), xyz)
    system, positions = system_fixture()
    expected_energy, expected_gradient = native_value_and_gradient(system, positions, [2, 1])
    assert energy.shape == (1,)
    assert float(energy.detach()[0]) == pytest.approx(expected_energy, abs=1e-12)
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-12, rtol=1e-12)
    assert AMBER_SCORING_CONTRACT == "openmm-fixed-receptor-kcal-ang-relative-search-v1"


def test_amber_force_bridge_passes_full_coordinate_gradcheck(scorer):
    xyz = scorer.ligand_coordinates(dtype=torch.float64)[None].requires_grad_()
    assert torch.autograd.gradcheck(scorer.energy, (xyz,), eps=1e-5, atol=2e-7, rtol=2e-6)


def test_coulomb_sign_and_angstrom_factor():
    system = openmm.System()
    for _ in range(2):
        system.addParticle(12.)
    force = openmm.NonbondedForce()
    force.addParticle(1., .3, 0.)
    force.addParticle(-1., .3, 0.)
    system.addForce(force)
    with OpenMMScorer.from_system(system, [[0., 0., 0.], [2., 0., 0.]], [1]) as scorer:
        xyz = scorer.ligand_coordinates(dtype=torch.float64).requires_grad_()
        energy = scorer.energy(xyz)
        gradient, = torch.autograd.grad(energy.sum(), xyz)
        # At 2 A opposite charges have attractive physical force, so dE/dx>0.
        assert float(energy.detach()[0]) == pytest.approx(-166.03185665, rel=2e-7)
        assert float(gradient[0, 0]) == pytest.approx(-float(energy.detach()[0]) / 2, rel=1e-12)
        torch.testing.assert_close(gradient[0, 1:], torch.zeros(2, dtype=xyz.dtype))


def test_amber_batch_gradients_are_independent_and_weighted_vjp(scorer):
    xyz = scorer.ligand_coordinates(dtype=torch.float64)
    batch = torch.stack((xyz, xyz + torch.tensor([.2, -.1, .3]))).requires_grad_()
    weights = torch.tensor([2., -3.], dtype=torch.float64)
    energies = scorer.energy(batch)
    batch_gradient, = torch.autograd.grad((energies * weights).sum(), batch)
    for row in range(2):
        single = batch[row].detach().clone().requires_grad_()
        value = scorer.energy(single)
        gradient, = torch.autograd.grad(value.sum(), single)
        torch.testing.assert_close(energies[row:row+1], value, rtol=0, atol=0)
        torch.testing.assert_close(batch_gradient[row], weights[row] * gradient, rtol=0, atol=0)


def test_amber_receptor_never_moves_and_threaded_calls_do_not_share_pose(scorer):
    initial = scorer.receptor_coordinates(dtype=torch.float64)
    xyz = scorer.ligand_coordinates(dtype=torch.float64)
    candidates = [xyz + .1 * i for i in range(4)]
    expected = [scorer.energy(x).item() for x in candidates]
    with ThreadPoolExecutor(max_workers=3) as pool:
        actual = list(pool.map(lambda x: scorer.energy(x).item(), candidates))
    assert actual == expected
    torch.testing.assert_close(scorer.receptor_coordinates(dtype=torch.float64), initial, rtol=0, atol=0)
    restored = scorer._context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    _, original = system_fixture()
    np.testing.assert_allclose(restored, original, rtol=0, atol=1e-15)


def test_caller_mutation_cannot_change_system_positions_or_identity():
    system, positions = system_fixture()
    with OpenMMScorer.from_system(system, positions, [2, 1], source_metadata={"force_field": "explicit-test"}) as scorer:
        xyz = scorer.ligand_coordinates(dtype=torch.float64)
        value, identity = scorer.energy(xyz).clone(), scorer.fingerprint
        positions[:] += 100
        system.getForce(0).setParticleParameters(0, 100., .8, 12.)
        data = scorer.metadata
        data["source"]["force_field"] = "changed"
        assert scorer.fingerprint == identity
        assert scorer.metadata["source"]["force_field"] == "explicit-test"
        torch.testing.assert_close(scorer.energy(xyz), value, rtol=0, atol=0)


def test_scorer_protocol_declares_undecomposed_potential(scorer):
    prepared = scorer.prepare({"active": torch.ones(2, dtype=torch.bool)}, scorer.receptor_coordinates(), {})
    xyz = scorer.ligand_coordinates(dtype=torch.float64)[None].requires_grad_()
    raw = prepared.raw_components(xyz)
    reported = prepared.report(raw, 999.)
    assert prepared.component_semantics == "undecomposed-full-complex-potential"
    assert set(prepared.components(xyz)) == {"complex_potential"}
    assert prepared.effective_rotatable_bonds == 0 and not prepared.torsion_penalty_applied
    torch.testing.assert_close(reported.score, scorer.energy(xyz))
    torch.testing.assert_close(raw.search_energy, scorer.energy(xyz) - scorer.reference_energy)
    assert torch.count_nonzero(raw.intermolecular) == torch.count_nonzero(raw.intramolecular) == 0
    assert scorer.metadata["objective"] == "full_complex_potential"
    assert reported.score.requires_grad


def test_amber_checks_allatom_graph_and_fixed_receptor_contract(scorer):
    scorer.validate_ligand(Chem.MolFromSmiles("CO"))
    with pytest.raises(ValueError, match="elements/order"):
        scorer.validate_ligand(Chem.MolFromSmiles("OC"))
    with pytest.raises(ValueError, match="every mapped atom"):
        scorer.validate_ligand(Chem.MolFromSmiles("C"))
    with pytest.raises(ValueError, match="all mapped ligand atoms"):
        scorer.prepare({"active": torch.ones(1)}, scorer.receptor_coordinates(), {})
    with pytest.raises(ValueError, match="fixed nonligand"):
        scorer.prepare({"active": torch.ones(2)}, scorer.receptor_coordinates() + 1., {})
    with pytest.raises(ValueError, match="exclusions"):
        scorer.prepare({"active": torch.ones(2)}, scorer.receptor_coordinates(), {},
                       intermolecular_exclusion_mask=torch.ones(2, 1))


@pytest.mark.parametrize("coordinates", [torch.ones(2, 3, dtype=torch.long), torch.ones(0, 2, 3),
                                        torch.ones(1, 3, 3), torch.full((1, 2, 3), float("nan"))])
def test_amber_rejects_invalid_coordinate_inputs(scorer, coordinates):
    with pytest.raises(ValueError, match="coordinates"):
        scorer.energy(coordinates)


@pytest.mark.parametrize("indices", [[], [1, 1], [-1], [3], [True], [1.5], "1", None, 1])
def test_amber_rejects_ambiguous_particle_mapping(indices):
    system, xyz = system_fixture()
    with pytest.raises(ValueError, match="ligand_atom_indices"):
        OpenMMScorer.from_system(system, xyz, indices)


def test_close_fails_explicitly_and_second_derivatives_are_unsupported(scorer):
    xyz = scorer.ligand_coordinates(dtype=torch.float64).requires_grad_()
    gradient, = torch.autograd.grad(scorer.energy(xyz).sum(), xyz, create_graph=True)
    with pytest.raises(RuntimeError):
        torch.autograd.grad(gradient.sum(), xyz)
    scorer.close()
    with pytest.raises(RuntimeError, match="closed"):
        scorer.energy(xyz)


def test_force_bridge_can_optimize_without_accumulating_scorer_parameters(scorer):
    parameter = torch.nn.Parameter(scorer.ligand_coordinates(dtype=torch.float64)[None])
    initial = scorer.energy(parameter).item()
    optimizer = torch.optim.LBFGS([parameter], lr=.2, max_iter=15, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = scorer.energy(parameter).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    assert scorer.energy(parameter).item() < initial


def test_relative_search_subtracts_large_constant_before_fp32_rounding():
    system = openmm.System()
    for _ in range(2):
        system.addParticle(12.)
    bond = openmm.HarmonicBondForce()
    bond.addBond(0, 1, .2, 100.)
    system.addForce(bond)
    constant = openmm.CustomExternalForce("offset")
    constant.addGlobalParameter("offset", 1e9)
    constant.addParticle(0)
    system.addForce(constant)
    with OpenMMScorer.from_system(system, [[0., 0., 0.], [3., 0., 0.]], [1]) as scorer:
        initial = scorer.ligand_coordinates()[None]
        moved = (initial + torch.tensor([[[.01, 0., 0.]]])).requires_grad_()
        assert scorer.energy(initial).item() == scorer.energy(moved).item()
        delta = scorer.relative_energy(moved)
        assert abs(delta.item()) > .001
        exact_delta = scorer.energy(moved.double()) - scorer.reference_energy
        torch.testing.assert_close(delta.double(), exact_delta, atol=1e-9, rtol=1e-6)
        full_gradient, = torch.autograd.grad(scorer.energy(moved).sum(), moved)
        relative_gradient, = torch.autograd.grad(delta.sum(), moved)
        torch.testing.assert_close(full_gradient, relative_gradient, atol=0, rtol=0)
        prepared = scorer.prepare({"active": torch.ones(1)}, scorer.receptor_coordinates(), {})
        report = prepared.score_components(moved)
        assert report.score.dtype == torch.float64
        torch.testing.assert_close(report.score, scorer.energy(moved.double()), atol=1e-7, rtol=0)


def test_engine_refinement_uses_reference_shift_but_reports_full_potential(scorer):
    from anchor_dock.core.engine import DockingEngine
    from anchor_dock.core.io import receptor_context_from_mol

    ligand = Chem.MolFromSmiles("CO")
    receptor_mol = Chem.MolFromSmiles("C")
    conformer = Chem.Conformer(1)
    conformer.SetAtomPosition(0, scorer.receptor_coordinates().tolist()[0])
    receptor_mol.AddConformer(conformer)
    receptor = receptor_context_from_mol(receptor_mol, "cpu")
    scorer.validate_ligand(ligand)
    engine = DockingEngine(scorer, optimizer="adam", learning_rate=.01, num_steps=4, device="cpu")
    problem = engine.prepare(ligand, receptor)
    xyz = scorer.ligand_coordinates()[None]
    _, final, _, _ = engine.optimize_se3(problem, xyz, 0, centers=xyz[:, 0], rotation_vectors=torch.zeros(1, 3))
    initial_report, final_report = engine.report_scores(problem, xyz, final)
    assert final_report.search_energy < initial_report.search_energy
    assert final_report.score < initial_report.score
    assert initial_report.score.dtype == final_report.score.dtype == torch.float64
    assert problem.scorer.metadata["objective"] == "full_complex_potential"


def test_float32_overflow_is_explicit_but_reference_shift_is_stable():
    system = openmm.System()
    system.addParticle(12.)
    force = openmm.CustomExternalForce("offset")
    force.addGlobalParameter("offset", 1e40)
    force.addParticle(0)
    system.addForce(force)
    with OpenMMScorer.from_system(system, [[0., 0., 0.]], [0]) as scorer:
        with pytest.raises(FloatingPointError, match="requested Torch dtype"):
            scorer.energy(torch.zeros(1, 1, 3))
        assert scorer.relative_energy(torch.zeros(1, 1, 3)).item() == 0.


@pytest.mark.parametrize("options", [{"implicit_solvent": "made-up"}, {"nonbonded_method": "unknown"},
                                    {"cutoff_angstrom": float("nan")}, {"solute_dielectric": True}])
def test_amber_file_options_fail_before_loading_nonexistent_files(options):
    with pytest.raises(ValueError):
        OpenMMScorer.from_amber("missing.prmtop", "missing.inpcrd", [0], **options)
