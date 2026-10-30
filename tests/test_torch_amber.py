"""Native Reference oracles for the pure Torch physical potential."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from rdkit import Chem

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

from anchor_dock.core.torch_amber import COULOMB_KCAL_ANGSTROM, TorchAmberGeometryError, TorchAmberScorer  # noqa: E402


def system_fixture():
    system = openmm.System()
    for _ in range(7):
        system.addParticle(12.)
    force = openmm.NonbondedForce()
    for q, sigma, eps in ((.3, .31, .2), (.2, .30, .4), (-.5, .32, .3),
                          (.1, .29, .25), (-.2, .31, .1), (.4, .27, .12), (-.1, .28, .13)):
        force.addParticle(q, sigma, eps)
    force.addException(0, 1, 0., .3, 0.)
    force.addException(1, 2, -.031, .25, .07)
    force.addException(3, 6, .017, .29, .15)
    system.addForce(force)
    bonds = openmm.HarmonicBondForce()
    bonds.addBond(0, 1, .15, 500.)  # fixed-only
    bonds.addBond(2, 3, .15, 700.)
    bonds.addBond(3, 4, .14, 800.)
    system.addForce(bonds)
    angles = openmm.HarmonicAngleForce()
    angles.addAngle(0, 1, 5, 1.5, 8.)
    angles.addAngle(2, 3, 4, 1.3, 12.)
    system.addForce(angles)
    torsions = openmm.PeriodicTorsionForce()
    torsions.addTorsion(0, 1, 5, 6, 2, .23, 1.2)
    torsions.addTorsion(1, 2, 3, 4, 3, .4, 2.3)
    torsions.addTorsion(1, 2, 3, 4, 1, -.7, -.2)
    system.addForce(torsions)
    xyz = np.array([[0., 0., 0.], [1.5, .1, .2], [4., .3, -.4], [5.2, 1.2, .3],
                    [6.3, .5, 1.1], [-1., 2., 1.], [1., 4., -1.]])
    return system, xyz


def native(system, coordinates, indices):
    integrator = openmm.VerletIntegrator(.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(coordinates * unit.angstrom)
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) / 4.184
    gradient = -state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)[indices] / 41.84
    del context, integrator
    return energy, gradient


@pytest.fixture
def scorer():
    system, xyz = system_fixture()
    return TorchAmberScorer.from_system(system, xyz, [4, 2, 3], atomic_numbers=[6]*7, pair_chunk_size=1)


def test_native_energy_force_parity_with_reordered_map_and_pair_tiles(scorer):
    xyz = scorer.ligand_coordinates(dtype=torch.float64).requires_grad_()
    energy = scorer.energy(xyz)
    gradient, = torch.autograd.grad(energy.sum(), xyz)
    system, reference = system_fixture()
    expected_energy, expected_gradient = native(system, reference, [4, 2, 3])
    assert energy.dtype == torch.float64
    assert energy.item() == pytest.approx(expected_energy, abs=2e-11)
    np.testing.assert_allclose(gradient, expected_gradient, atol=2e-11, rtol=2e-11)
    for chunk in (2, 3, 100):
        alternate = TorchAmberScorer.from_system(system, reference, [4, 2, 3], pair_chunk_size=chunk)
        torch.testing.assert_close(alternate.energy(xyz), energy, atol=2e-11, rtol=2e-13)


def test_full_coordinate_gradcheck_and_gradgradcheck(scorer):
    xyz = scorer.ligand_coordinates(dtype=torch.float64)[None].requires_grad_()
    assert torch.autograd.gradcheck(scorer.energy, (xyz,), eps=1e-5, atol=1e-6, rtol=1e-5)
    assert torch.autograd.gradgradcheck(scorer.relative_energy, (xyz,), eps=1e-5, atol=2e-5, rtol=2e-5)


def test_batch_independent_and_fixed_receptor(scorer):
    xyz = scorer.ligand_coordinates(dtype=torch.float64)
    receptor = scorer.receptor_coordinates(dtype=torch.float64)
    batch = torch.stack((xyz, xyz + .1, xyz - .2)).requires_grad_()
    values = scorer.energy(batch)
    weights = torch.tensor([1., -2., .5], dtype=torch.float64)
    gradients, = torch.autograd.grad((weights * values).sum(), batch)
    for i in range(3):
        single = batch[i].detach().clone().requires_grad_()
        energy = scorer.energy(single)
        gradient, = torch.autograd.grad(energy.sum(), single)
        torch.testing.assert_close(values[i:i+1], energy, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(gradients[i], weights[i] * gradient, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(scorer.receptor_coordinates(dtype=torch.float64), receptor, atol=0, rtol=0)
    assert not any(value.requires_grad for value in scorer._buffers.values())


def test_runtime_never_constructs_context_or_imports_openmm(scorer, monkeypatch):
    import anchor_dock.core.torch_amber as module
    def forbidden(*args, **kwargs):
        raise AssertionError("OpenMM is forbidden during runtime Torch evaluation")
    monkeypatch.setattr(openmm, "Context", forbidden)
    monkeypatch.setattr(module, "_openmm", forbidden)
    xyz = scorer.ligand_coordinates(dtype=torch.float64).requires_grad_()
    energy = scorer.energy(xyz)
    gradient, = torch.autograd.grad(energy.sum(), xyz)
    assert torch.isfinite(energy).all() and torch.isfinite(gradient).all()


@pytest.mark.parametrize("chunk", [1, 2, 128])
def test_cached_and_fallback_pairs_preserve_batched_energy_gradient_and_exceptions(chunk):
    system, reference = system_fixture()
    system.getForce(0).addException(2, 4, 0., .3, 0.)
    cached = TorchAmberScorer.from_system(system, reference, [4, 2, 3], pair_chunk_size=chunk)
    fallback = TorchAmberScorer.from_system(system, reference, [4, 2, 3], pair_chunk_size=chunk, pair_cache_max_pairs=0)
    coordinates = cached.ligand_coordinates(dtype=torch.float64)
    batch = torch.stack((coordinates, coordinates + .1, coordinates - .2)).requires_grad_()
    expected = fallback.energy(batch)
    actual = cached.energy(batch)
    expected_gradient, = torch.autograd.grad(expected.sum(), batch, retain_graph=True)
    actual_gradient, = torch.autograd.grad(actual.sum(), batch)
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-13)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-11, rtol=2e-13)
    torch.testing.assert_close(cached.relative_energy(batch), fallback.relative_energy(batch), atol=2e-11, rtol=2e-13)
    assert cached.metadata["pair_cache"]["moving_candidate_pairs"] == 15
    assert cached.metadata["pair_cache"]["cached_active_pairs"] == 14
    assert not fallback.metadata["pair_cache"]["enabled"]
    assert "moving_pair_indices" not in fallback._buffers


def test_cached_runtime_does_not_rebuild_static_pairs_or_coefficients(scorer, monkeypatch):
    assert scorer.metadata["pair_cache"]["enabled"]
    snapshots = {name: value.data_ptr() for name, value in scorer._buffers.items()}
    def forbidden(*args, **kwargs):
        raise AssertionError("static pair construction must only run during preparation")
    monkeypatch.setattr(scorer, "_pair_tiles", forbidden)
    monkeypatch.setattr(scorer, "_pair_parameters", forbidden)
    monkeypatch.setattr(scorer, "_nonbonded", forbidden)
    coordinates = scorer.ligand_coordinates(dtype=torch.float64).requires_grad_()
    result = scorer.energy(coordinates)
    gradient, = torch.autograd.grad(result.sum(), coordinates)
    assert torch.isfinite(gradient).all()
    assert snapshots == {name: value.data_ptr() for name, value in scorer._buffers.items()}
    assert not any(value.requires_grad for value in scorer._buffers.values())


@pytest.mark.parametrize(("limit", "enabled"), [(0, False), (14, False), (15, True), (16, True)])
def test_pair_cache_threshold_is_inclusive_and_bounded_before_exclusions(limit, enabled):
    system, positions = system_fixture()
    movable = {4, 2, 3}
    force = system.getForce(0)
    for a in range(7):
        for b in range(a + 1, 7):
            if a in movable or b in movable:
                force.addException(a, b, 0., .3, 0., True)
    scorer = TorchAmberScorer.from_system(system, positions, [4, 2, 3], pair_cache_max_pairs=limit)
    cache = scorer.metadata["pair_cache"]
    assert cache["enabled"] is enabled
    assert cache["max_candidate_pairs"] == limit
    assert cache["moving_candidate_pairs"] == 15
    assert cache["cached_active_pairs"] == 0
    assert ("moving_pair_indices" in scorer._buffers) is enabled
    if enabled:
        assert scorer._buffers["moving_pair_indices"].shape == (0, 2)
        assert scorer._buffers["moving_pair_parameters"].shape == (0, 3)


@pytest.mark.parametrize("limit", [-1, True, 1.5, None])
def test_pair_cache_limit_validation(limit):
    system, positions = system_fixture()
    with pytest.raises(ValueError, match="pair_cache_max_pairs"):
        TorchAmberScorer.from_system(system, positions, [4, 2, 3], pair_cache_max_pairs=limit)


@pytest.mark.parametrize("nonbonded", [False, True])
@pytest.mark.parametrize("limit", [0, 10])
def test_no_nonbonded_force_or_no_pairs_has_zero_energy_and_gradient(nonbonded, limit):
    system = openmm.System()
    system.addParticle(12.)
    if nonbonded:
        force = openmm.NonbondedForce()
        force.addParticle(1., .3, 1.)
        system.addForce(force)
    scorer = TorchAmberScorer.from_system(system, [[0., 0., 0.]], [0], pair_cache_max_pairs=limit)
    coordinates = torch.tensor([[[1., 2., 3.]], [[4., 5., 6.]]], dtype=torch.float64, requires_grad=True)
    value = scorer.energy(coordinates)
    gradient, = torch.autograd.grad(value.sum(), coordinates)
    torch.testing.assert_close(value, torch.zeros(2, dtype=torch.float64), atol=0, rtol=0)
    torch.testing.assert_close(gradient, torch.zeros_like(coordinates), atol=0, rtol=0)
    assert scorer.metadata["pair_cache"]["enabled"] is (nonbonded and limit > 0)
    assert scorer.metadata["pair_cache"]["moving_candidate_pairs"] == 0


@pytest.mark.parametrize("limit", [0, 10])
def test_all_excluded_coincident_pairs_do_not_evaluate_singular_distances(limit):
    system = openmm.System()
    force = openmm.NonbondedForce()
    for _ in range(3):
        system.addParticle(12.)
        force.addParticle(1., .3, 1.)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        force.addException(a, b, 0., .3, 0.)
    system.addForce(force)
    scorer = TorchAmberScorer.from_system(system, [[0., 0., 0.]]*3, [2, 0], pair_cache_max_pairs=limit)
    coordinates = scorer.ligand_coordinates(dtype=torch.float64).requires_grad_()
    value = scorer.energy(coordinates)
    gradient, = torch.autograd.grad(value.sum(), coordinates)
    assert value.item() == 0.
    torch.testing.assert_close(gradient, torch.zeros_like(coordinates), atol=0, rtol=0)


def test_coulomb_units_exceptions_and_coincident_exclusion():
    system = openmm.System()
    for _ in range(3):
        system.addParticle(12.)
    force = openmm.NonbondedForce()
    for q in (1., -1., .3):
        force.addParticle(q, .3, 0.)
    force.addException(0, 2, 0., .3, 0.)
    force.addException(1, 2, 0., .3, 0.)
    system.addForce(force)
    scorer = TorchAmberScorer.from_system(system, [[0., 0., 0.], [2., 0., 0.], [0., 0., 0.]], [1])
    xyz = scorer.ligand_coordinates(dtype=torch.float64).requires_grad_()
    energy = scorer.energy(xyz)
    gradient, = torch.autograd.grad(energy.sum(), xyz)
    assert energy.item() == pytest.approx(-COULOMB_KCAL_ANGSTROM / 2, abs=1e-12)
    assert gradient[0, 0] == pytest.approx(COULOMB_KCAL_ANGSTROM / 4, abs=1e-12)
    with pytest.raises(ValueError, match="zero separation"):
        scorer.energy(torch.zeros_like(xyz))


@pytest.mark.parametrize("limit", [0, 10])
@pytest.mark.parametrize("exception", [False, True])
def test_coulomb_only_pairs_ignore_unused_sigma_before_lj_powers(limit, exception):
    system, force = openmm.System(), openmm.NonbondedForce()
    for charge in (1., -1.):
        system.addParticle(12.)
        # Finite imported parameters whose unused LJ power would overflow.
        force.addParticle(charge, 1e100, 0.)
    product = -1.
    if exception:
        product = .25
        force.addException(0, 1, product, 1e100, 0.)
    system.addForce(force)
    scorer = TorchAmberScorer.from_system(system, [[0., 0., 0.], [2., .1, .3]], [1],
                                         pair_cache_max_pairs=limit)
    coordinates = scorer.ligand_coordinates(dtype=torch.float64).requires_grad_()
    expected = COULOMB_KCAL_ANGSTROM * product / torch.linalg.vector_norm(coordinates, dim=-1)
    energy = scorer.energy(coordinates)
    gradient, = torch.autograd.grad(energy.sum(), coordinates)
    expected_gradient = -COULOMB_KCAL_ANGSTROM * product * coordinates / torch.linalg.vector_norm(
        coordinates, dim=-1, keepdim=True).pow(3)
    torch.testing.assert_close(energy, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(gradient, expected_gradient, atol=1e-12, rtol=1e-12)
    assert torch.autograd.gradcheck(scorer.energy, (coordinates,), eps=1e-5, atol=1e-7, rtol=1e-6)
    assert torch.autograd.gradgradcheck(scorer.relative_energy, (coordinates,), eps=1e-5, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("limit", [0, 10])
def test_fixed_coulomb_only_constant_ignores_unused_sigma(limit):
    system, force = openmm.System(), openmm.NonbondedForce()
    for charge in (1., -1., 0.):
        system.addParticle(12.)
        force.addParticle(charge, 1e100, 0.)
    system.addForce(force)
    scorer = TorchAmberScorer.from_system(system, [[0., 0., 0.], [2., 0., 0.], [5., 0., 0.]], [2],
                                         pair_cache_max_pairs=limit)
    coordinates = scorer.ligand_coordinates(dtype=torch.float64).requires_grad_()
    energy = scorer.energy(coordinates)
    gradient, = torch.autograd.grad(energy.sum(), coordinates)
    assert energy.item() == pytest.approx(-COULOMB_KCAL_ANGSTROM / 2, abs=1e-12)
    torch.testing.assert_close(scorer.relative_energy(coordinates), torch.zeros(1, dtype=torch.float64), atol=0, rtol=0)
    torch.testing.assert_close(gradient, torch.zeros_like(coordinates), atol=0, rtol=0)


def test_fixed_constant_removed_before_search_and_fp32_gradient_preserved():
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12.)
    force = openmm.HarmonicBondForce()
    force.addBond(0, 1, .1, 1e18)
    force.addBond(2, 3, .1, 10.)
    system.addForce(force)
    scorer = TorchAmberScorer.from_system(system, [[0., 0., 0.], [2., 0., 0.], [4., 0., 0.], [5.2, 0., 0.]], [3])
    xyz = (scorer.ligand_coordinates() + .01).requires_grad_()
    relative = scorer.relative_energy(xyz)
    assert relative.dtype == torch.float64 and abs(relative.item()) > 1e-6
    g_relative, = torch.autograd.grad(relative.sum(), xyz)
    g_full, = torch.autograd.grad(scorer.energy(xyz).sum(), xyz)
    assert g_full.dtype == torch.float32
    torch.testing.assert_close(g_relative, g_full, rtol=0, atol=0)


def test_protocol_mapping_preparation_and_caller_mutation(scorer):
    scorer.validate_ligand(Chem.MolFromSmiles("CCC"))
    with pytest.raises(ValueError, match="elements/order"):
        scorer.validate_ligand(Chem.MolFromSmiles("CCO"))
    with pytest.raises(ValueError, match="every mapped atom"):
        scorer.validate_ligand(Chem.MolFromSmiles("C"))
    prepared = scorer.prepare({"active": torch.ones(3)}, scorer.receptor_coordinates(), {})
    xyz = scorer.ligand_coordinates(dtype=torch.float64)
    assert prepared.component_semantics == "undecomposed-full-complex-potential"
    assert set(prepared.components(xyz)) == {"complex_potential"}
    torch.testing.assert_close(prepared.search_energy(xyz), torch.zeros(1, dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(prepared.score_components(xyz).score, scorer.energy(xyz), atol=1e-12, rtol=0)
    with pytest.raises(ValueError, match="fixed nonligand"):
        scorer.prepare({"active": torch.ones(3)}, scorer.receptor_coordinates() + 1, {})
    with pytest.raises(ValueError, match="exclusions"):
        scorer.prepare({"active": torch.ones(3)}, scorer.receptor_coordinates(), {}, intermolecular_exclusion_mask=torch.ones(3, 4))
    system, positions = system_fixture()
    other = TorchAmberScorer.from_system(system, positions, [4, 2, 3], source_metadata={"test": [1]})
    before, identity = other.energy(xyz).clone(), other.fingerprint
    system.getForce(0).setParticleParameters(0, 99, .8, 80)
    positions[:] = 1e6
    metadata = other.metadata
    metadata["source"]["test"][0] = 9
    assert other.fingerprint == identity and other.metadata["source"]["test"] == [1]
    torch.testing.assert_close(other.energy(xyz), before, atol=0, rtol=0)


def test_topology_graph_edges_and_missing_hydrogen_are_validated():
    system, positions = system_fixture()
    scorer = TorchAmberScorer.from_system(system, positions, [4, 2, 3], atomic_numbers=[6]*7,
                                         topology_bonds=[(2, 3), (3, 4)])
    # Explicit map makes 2 the central ligand atom; identical element counts
    # alone cannot detect an incorrect local graph edge set.
    graph = Chem.RWMol()
    for _ in range(3):
        graph.AddAtom(Chem.Atom(6))
    graph.AddBond(0, 2, Chem.BondType.SINGLE)
    graph.AddBond(1, 2, Chem.BondType.SINGLE)
    scorer.validate_ligand(graph)
    with pytest.raises(ValueError, match="connectivity"):
        scorer.validate_ligand(Chem.MolFromSmiles("CCC"))
    with pytest.raises(ValueError, match="bonded hydrogens"):
        TorchAmberScorer.from_system(system, positions, [4, 2, 3], atomic_numbers=[1, 6, 6, 6, 6, 6, 6],
                                     topology_bonds=[(0, 2)])
    assert "induced topology edges" in scorer.metadata["ligand_graph_validation"]


@pytest.mark.parametrize("coordinates", [torch.ones(3, 3, dtype=torch.int64), torch.ones(0, 3, 3),
                                        torch.ones(1, 2, 3), torch.full((1, 3, 3), float("nan"))])
def test_invalid_coordinate_inputs_rejected(scorer, coordinates):
    with pytest.raises(ValueError, match="coordinates"):
        scorer.energy(coordinates)


def test_parameter_transfer_and_precision_are_explicit(scorer):
    assert scorer.to("cpu") is scorer
    with pytest.raises(ValueError, match="float64"):
        scorer.to(dtype=torch.float32)
    assert scorer.device == scorer._buffers["positions"].device


@pytest.mark.parametrize("indices", [[], [1, 1], [-1], [7], [True], [1.5], "1", None, 1])
def test_explicit_particle_map_validation(indices):
    system, positions = system_fixture()
    with pytest.raises(ValueError, match="ligand_atom_indices"):
        TorchAmberScorer.from_system(system, positions, indices)


@pytest.mark.parametrize("mode", ["PME", "GB", "custom", "virtual", "offset", "periodic-bond", "switch", "direct"])
def test_unsupported_physics_fails_closed(mode):
    system, positions = system_fixture()
    if mode == "PME":
        system.getForce(0).setNonbondedMethod(openmm.NonbondedForce.PME)
    elif mode == "GB":
        system.addForce(openmm.GBSAOBCForce())
    elif mode == "custom":
        system.addForce(openmm.CustomExternalForce("0"))
    elif mode == "virtual":
        system.setVirtualSite(6, openmm.TwoParticleAverageSite(0, 1, .5, .5))
    elif mode == "offset":
        force = system.getForce(0)
        force.addGlobalParameter("lambda", 1.)
        force.addParticleParameterOffset("lambda", 0, .1, 0., 0.)
    elif mode == "periodic-bond":
        system.getForce(1).setUsesPeriodicBoundaryConditions(True)
    elif mode == "switch":
        system.getForce(0).setUseSwitchingFunction(True)
    elif mode == "direct":
        system.getForce(0).setIncludeDirectSpace(False)
    with pytest.raises(ValueError, match="unsupported|virtual|NoCutoff"):
        TorchAmberScorer.from_system(system, positions, [4, 2, 3])


def test_singular_active_geometry_rejected_and_zero_terms_ignored():
    assert issubclass(TorchAmberGeometryError, ValueError)
    assert issubclass(TorchAmberGeometryError, FloatingPointError)
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12.)
    angle = openmm.HarmonicAngleForce()
    angle.addAngle(0, 1, 2, 1., 2.)
    system.addForce(angle)
    positions = [[0., 0., 0.], [1., 0., 0.], [2., .3, 0.], [3., .5, 1.]]
    scorer = TorchAmberScorer.from_system(system, positions, [2])
    with pytest.raises(ValueError, match="collinear"):
        scorer.energy(torch.tensor([[2., 0., 0.]], dtype=torch.float64))
    angle.setAngleParameters(0, 0, 1, 2, 1., 0.)
    torsion = openmm.PeriodicTorsionForce()
    torsion.addTorsion(0, 1, 2, 3, 2, .3, 1.)
    system.addForce(torsion)
    scorer = TorchAmberScorer.from_system(system, positions, [2])
    with pytest.raises(ValueError, match="degenerate"):
        scorer.energy(torch.tensor([[2., 0., 0.]], dtype=torch.float64))
    torsion.setTorsionParameters(0, 0, 1, 2, 3, 2, .3, 0.)
    scorer = TorchAmberScorer.from_system(system, positions, [2])
    assert scorer.energy(torch.tensor([[2., 0., 0.]], dtype=torch.float64)).item() == 0.


def test_invalid_force_particle_indices_do_not_wrap_as_torch_indices():
    system, positions = system_fixture()
    system.getForce(1).addBond(-1, 3, .2, 1.)
    with pytest.raises(ValueError, match="invalid System particle index"):
        TorchAmberScorer.from_system(system, positions, [4, 2, 3])


def test_constraints_retain_force_terms_and_optimizer_decreases_energy(scorer):
    system, reference = system_fixture()
    system.addConstraint(2, 3, .15)
    constrained = TorchAmberScorer.from_system(system, reference, [4, 2, 3])
    xyz = scorer.ligand_coordinates(dtype=torch.float64)
    torch.testing.assert_close(constrained.energy(xyz), scorer.energy(xyz), atol=1e-12, rtol=1e-12)
    movable = (xyz + torch.tensor([[.1, -.1, .2], [.1, .3, -.1], [0., -.2, .3]])).requires_grad_()
    before = scorer.relative_energy(movable).item()
    optimizer = torch.optim.LBFGS([movable], max_iter=12, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = scorer.relative_energy(movable).sum()
        loss.backward()
        return loss
    optimizer.step(closure)
    assert scorer.relative_energy(movable).item() < before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_cpu_energy_force_parity(scorer):
    cpu = scorer.ligand_coordinates(dtype=torch.float64).requires_grad_()
    energy = scorer.energy(cpu)
    gradient, = torch.autograd.grad(energy.sum(), cpu)
    scorer.to("cuda")
    gpu = cpu.detach().cuda().requires_grad_()
    value = scorer.energy(gpu)
    derivative, = torch.autograd.grad(value.sum(), gpu)
    torch.testing.assert_close(value.cpu(), energy, atol=1e-10, rtol=1e-11)
    torch.testing.assert_close(derivative.cpu(), gradient, atol=1e-10, rtol=1e-11)


def test_standard_amber_vacuum_fixture_matches_reference():
    root = Path(__file__).parent / "fixtures" / "amber"
    scorer = TorchAmberScorer.from_amber(root / "alanine-dipeptide-implicit.prmtop", root / "alanine-dipeptide-implicit.inpcrd", list(range(6, 16)))
    fallback = TorchAmberScorer.from_amber(root / "alanine-dipeptide-implicit.prmtop", root / "alanine-dipeptide-implicit.inpcrd", list(range(6, 16)), pair_cache_max_pairs=0)
    assert scorer.metadata["pair_cache"]["enabled"] and not fallback.metadata["pair_cache"]["enabled"]
    from openmm import app
    topology = app.AmberPrmtopFile(str(root / "alanine-dipeptide-implicit.prmtop"))
    system = topology.createSystem(nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False, removeCMMotion=False)
    coordinates = app.AmberInpcrdFile(str(root / "alanine-dipeptide-implicit.inpcrd")).positions.value_in_unit(unit.angstrom)
    xyz = scorer.ligand_coordinates(dtype=torch.float64).requires_grad_()
    expected, derivative = native(system, coordinates, list(range(6, 16)))
    value = scorer.energy(xyz)
    gradient, = torch.autograd.grad(value.sum(), xyz)
    fallback_value = fallback.energy(xyz)
    fallback_gradient, = torch.autograd.grad(fallback_value.sum(), xyz)
    torch.testing.assert_close(value, fallback_value, atol=1e-11, rtol=1e-12)
    torch.testing.assert_close(gradient, fallback_gradient, atol=1e-11, rtol=1e-12)
    assert value.item() == pytest.approx(expected, abs=1e-9)
    np.testing.assert_allclose(gradient, derivative, atol=1e-9, rtol=1e-9)
    assert torch.autograd.gradcheck(scorer.relative_energy, (xyz,), eps=1e-5, atol=2e-6, rtol=2e-5)
    graph = Chem.RWMol()
    for z in scorer.atomic_numbers[6:16]:
        graph.AddAtom(Chem.Atom(z))
    for a, b in topology.topology.bonds():
        if 6 <= a.index < 16 and 6 <= b.index < 16:
            graph.AddBond(a.index-6, b.index-6, Chem.BondType.SINGLE)
    scorer.validate_ligand(graph)
    perturbed = xyz.detach() + .01 * torch.arange(xyz.numel(), dtype=torch.float64).reshape(xyz.shape) / xyz.numel()
    combined = np.array(coordinates, copy=True)
    combined[6:16] = perturbed.numpy()
    expected_perturbed, derivative_perturbed = native(system, combined, list(range(6, 16)))
    perturbed.requires_grad_()
    value = scorer.energy(perturbed)
    gradient, = torch.autograd.grad(value.sum(), perturbed)
    assert value.item() == pytest.approx(expected_perturbed, abs=1e-9)
    np.testing.assert_allclose(gradient, derivative_perturbed, atol=1e-9, rtol=1e-9)
    with pytest.raises(ValueError, match="vacuum NoCutoff"):
        TorchAmberScorer.from_amber("missing", "missing", [0], implicit_solvent="GBn2")
