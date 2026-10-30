"""Public SF selection, independent starts, direction and provenance contracts."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from rdkit import Chem

from anchor_dock import (
    DockingEngine,
    ScoringFunctionInfo,
    create_scoring_function,
    list_scoring_functions,
    refine_pose,
    refine_with_scorers,
    register_scoring_function,
)
from anchor_dock.core import registry
from anchor_dock.core.scoring import (
    PairwiseScorer,
    RawScoreComponents,
    ScoreComponents,
    resolve_scorer,
    scorer_fingerprint,
)


@pytest.fixture
def inputs(tmp_path):
    mol = Chem.MolFromSmiles("CCO")
    conformer = Chem.Conformer(3)
    for i, point in enumerate(((4., 0., .2), (5.4, .2, .3), (6.5, .6, .2))):
        conformer.SetAtomPosition(i, point)
    mol.AddConformer(conformer)
    ligand = tmp_path / "pose.sdf"
    writer = Chem.SDWriter(str(ligand))
    writer.write(mol)
    writer.close()
    receptor = Chem.MolFromSmiles("C")
    receptor.AddConformer(Chem.Conformer(1))
    protein = tmp_path / "protein.pdb"
    Chem.MolToPDBFile(receptor, str(protein))
    return protein, ligand, mol


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", registry._REGISTRY.copy())


def test_catalog_has_torch_physical_routes_and_no_native_wrappers():
    entries = {v["name"]: v for v in list_scoring_functions()}
    assert set(entries) == {"vina", "vinardo", "ad4", "adgpu", "amber", "vina-final", "vinardo-final"}
    assert all(v["backend"] == "torch" for v in entries.values())
    assert entries["vina-final"]["optimization"] == ("pattern",)
    assert "maps_prefix" in entries["ad4"]["required_options"]
    assert entries["amber"]["units"] == "kcal/mol"


@pytest.mark.parametrize("name", ["softdock", "SOFTDOCK", "unknown", "dkoes_fast", "dkoes_scoring", "dkoes_scoring_old", "smina:vinardo", "gnina:cnn", "vina-native:ad4"])
def test_removed_or_unknown_sf_fails_closed(name):
    with pytest.raises(ValueError):
        create_scoring_function(name)


@pytest.mark.parametrize("name", ["vina", "vinardo"])
def test_existing_pose_noop_preserves_coordinates_order_and_records_options(inputs, name):
    protein, ligand, source = inputs
    input_bytes = ligand.read_bytes()
    before = source.GetConformer().GetPositions().copy()
    result = refine_pose(protein, ligand, scorer=name, num_steps=0)
    torch.testing.assert_close(result.coordinates, torch.tensor(before, dtype=result.coordinates.dtype))
    assert tuple(a.GetAtomicNum() for a in result.mol.GetAtoms()) == (6, 6, 8)
    assert result.metadata["output_atom_to_input_atom"] == [0, 1, 2]
    assert result.metadata["optimization_parameters"]["num_steps"] == 0
    assert result.metadata["optimization_parameters"]["gradient_clip"] == 1.
    assert result.scores["search_energy"]["value"] == result.initial_scores["search_energy"]["value"]
    assert ligand.read_bytes() == input_bytes


def test_multiple_sfs_start_independently_and_are_not_raw_ranked(inputs, tmp_path):
    protein, ligand, _ = inputs
    options = dict(num_steps=3, optimizer="adam")
    first = refine_pose(protein, ligand, scorer="vina", **options)
    combined = refine_with_scorers(protein, ligand, ["vinardo", "vina"], **options)
    assert combined["success"] and combined["raw_scores_comparable"] is False
    assert combined["results"]["vina"]["scores"] == first.scores
    assert combined["results"]["vina"]["initial_scores"] == first.initial_scores
    assert first.scores["search_energy"]["value"] <= first.initial_scores["search_energy"]["value"]


def test_output_contains_atom_identity_and_exact_optimization_options(inputs, tmp_path):
    protein, ligand, _ = inputs
    destination = tmp_path / "out"
    result = refine_pose(protein, ligand, num_steps=1, optimizer_options={"gradient_clip": .3, "patience": 7}, output_dir=destination)
    saved = json.loads((destination / "result.json").read_text())
    assert saved["metadata"]["output_atomic_numbers"] == [6, 6, 8]
    assert saved["metadata"]["output_atom_to_input_atom"] == [0, 1, 2]
    assert saved["metadata"]["optimization_parameters"]["patience"] == 7
    assert saved["metadata"]["optimization_parameters"]["gradient_clip"] == .3
    assert saved["scores"] == result.scores
    with pytest.raises(FileExistsError):
        refine_pose(protein, ligand, num_steps=0, output_dir=destination)


def test_unavailable_or_unconfigured_sf_records_error_without_substitution(inputs):
    protein, ligand, _ = inputs
    result = refine_with_scorers(protein, ligand, ["vina", "ad4", "unknown"], num_steps=0)
    assert not result["success"]
    assert set(result["results"]) == {"vina"}
    assert set(result["errors"]) == {"ad4", "unknown"}
    assert "maps_prefix" in result["errors"]["ad4"]["message"]


def test_refinement_rejects_unpositioned_smiles_and_two_dimensional_pose(inputs):
    protein, _, _ = inputs
    with pytest.raises(ValueError):
        refine_pose(protein, "CCO", num_steps=0)
    with pytest.raises(ValueError, match="positioned 3D"):
        refine_pose(protein, "CCO", num_steps=0, preparation_options={"ligand": {"generate_coordinates_if_missing": True}})


class HigherReporter:
    name = "higher-reporter"
    units = "arbitrary"
    fingerprint = "test:higher-v1"
    effective_rotatable_bonds = 0
    torsion_penalty_applied = False

    def prepare(self, *args, **kwargs):
        return self

    def raw_components(self, x):
        value = x.detach().square().sum((1, 2))
        zero = torch.zeros_like(value)
        return RawScoreComponents(zero, zero, value)

    def search_energy(self, x):
        return self.raw_components(x).search_energy

    def report(self, raw, intramolecular_reference=None):
        return ScoreComponents(raw.intermolecular, raw.intramolecular, raw.search_energy, -raw.search_energy, raw.search_energy.new_zeros(()))


def test_plugin_capability_and_report_direction_do_not_need_duplicate_factory_flags(inputs):
    protein, ligand, _ = inputs
    info = ScoringFunctionInfo("test-higher", "test", "torch", ("pattern",), "inferred", "arbitrary", direction="higher")
    register_scoring_function(info, HigherReporter)
    result = refine_pose(protein, ligand, scorer="test-higher", num_steps=1)
    assert result.metadata["optimization_route"] == "pattern"
    assert result.scores["score"]["direction"] == "higher"
    assert result.scores["score"]["value"] >= result.initial_scores["score"]["value"]
    with pytest.raises(ValueError, match="autograd"):
        DockingEngine("test-higher")
    with pytest.raises(ValueError, match="autograd"):
        refine_pose(protein, ligand, scorer="test-higher", optimizer="adam", num_steps=0)


def test_registry_replacement_is_honored_by_resolution_and_fingerprint():
    info = ScoringFunctionInfo("vina", "test", "torch", ("autograd", "pattern"), "inferred", "arbitrary")
    register_scoring_function(info, HigherReporter, replace=True)
    assert isinstance(resolve_scorer("vina"), HigherReporter)
    assert scorer_fingerprint("vina") == HigherReporter.fingerprint


def test_duplicate_and_invalid_registration_is_explicit():
    info = ScoringFunctionInfo("vina", "vina", "torch", ("autograd",), "inferred", "kcal/mol-like")
    with pytest.raises(ValueError, match="already registered"):
        register_scoring_function(info, lambda: PairwiseScorer("vina"))
    for broken in (replace(info, name="../bad"), replace(info, direction="maybe"), replace(info, optimization=("fake",))):
        with pytest.raises(ValueError):
            register_scoring_function(broken, HigherReporter)


def test_cli_catalog_and_refine_command(inputs, tmp_path, capsys):
    from anchor_dock.cli import main
    protein, ligand, _ = inputs
    assert main(["scorers"]) == 0
    catalog = json.loads(capsys.readouterr().out)
    assert not any(v["name"] == "softdock" for v in catalog)
    assert main(["refine", "-p", str(protein), "-l", str(ligand), "--scorers", "vina", "vinardo", "--steps", "0", "-o", str(tmp_path / "cli")]) == 0
    assert json.loads(capsys.readouterr().out)["success"]


def _oracle_prepared_inputs():
    from anchor_dock.core.prepared import PreparedLigand
    from anchor_dock.preparation import LigandPreparation, ReceptorPreparation
    data = json.loads((Path(__file__).parent / "fixtures/native_scoring.json").read_text())
    def read(payload):
        mol = Chem.MolFromMolBlock(payload["molblock"], removeHs=False)
        mapping = {int(k): v for k, v in payload["serial_to_atom_index"].items()}
        prepared = PreparedLigand.from_pdbqt(mol, payload["pdbqt"], serial_to_atom_index=mapping)
        return mol, mapping, prepared
    source = data["cases"]["secondary_amine"]
    mol, _, prepared = read(source)
    ligand = LigandPreparation(prepared, source["pdbqt"], mol, {"input_atom_count": mol.GetNumAtoms()})
    rec_mol, mapping, _ = read(data["receptor"])
    rigid = "\n".join(line for line in data["receptor"]["pdbqt"].splitlines() if line.startswith(("ATOM", "HETATM"))) + "\n"
    receptor = ReceptorPreparation(rigid, rec_mol, mapping, {"source": "pinned native fixture"})
    return receptor, ligand


@pytest.mark.parametrize("name", ["vina-final", "vinardo-final"])
def test_native_final_public_pattern_refinement_and_atom_mapping(name):
    receptor, ligand = _oracle_prepared_inputs()
    result = refine_pose(receptor, ligand, scorer=name, num_steps=1, optimizer_options={"pattern_max_evaluations": 200})
    assert result.metadata["optimization_route"] == "pattern"
    assert result.scores["search_energy"]["value"] <= result.initial_scores["search_energy"]["value"]
    assert result.metadata["output_atom_to_input_atom"] == list(ligand.prepared.original_atom_indices)
    assert result.metadata["pdbqt_serials"] == list(ligand.prepared.pdbqt_serials)


def test_amber_public_path_reports_full_potential_without_fake_decomposition(inputs):
    openmm = pytest.importorskip("openmm")
    from anchor_dock import OpenMMScorer
    _, _, mol = inputs
    system, force = openmm.System(), openmm.NonbondedForce()
    for charge in (.1, -.1, -.2, .2):
        system.addParticle(12.)
        force.addParticle(charge, .3, .1)
    system.addForce(force)
    positions = torch.cat((torch.tensor(mol.GetConformer().GetPositions()), torch.tensor([[0., 0., 0.]]))).numpy()
    with OpenMMScorer.from_system(system, positions, [0, 1, 2], atomic_numbers=[6, 6, 8, 6]) as scorer:
        result = refine_pose(None, mol, scorer=scorer, num_steps=2)
    assert result.scores["score"]["direction"] == "lower"
    assert "complex_potential" in result.scores
    assert "intermolecular" not in result.scores and "intramolecular" not in result.scores
    assert result.scores["search_energy"]["value"] <= result.initial_scores["search_energy"]["value"]
    assert result.metadata["component_semantics"] == "undecomposed-full-complex-potential"


def test_prepared_objects_do_not_silently_ignore_preparation_options():
    receptor, ligand = _oracle_prepared_inputs()
    with pytest.raises(ValueError, match="already prepared ligand"):
        refine_pose(receptor, ligand, num_steps=0, preparation_options={"ligand": {"charge_model": "gasteiger"}})
    with pytest.raises(ValueError, match="already prepared receptor"):
        refine_pose(receptor, ligand, num_steps=0, preparation_options={"receptor": {"charge_model": "read"}})


@pytest.mark.parametrize("smiles", ["CC", "CCO", "CCCO", "CCOc1ccccc1", "CC(=O)N"])
def test_inferred_rotors_are_independent_of_explicit_hydrogen_and_map_to_original_order(smiles):
    from anchor_dock.core.topology import find_rotatable_bonds
    heavy = Chem.MolFromSmiles(smiles)
    all_atoms = Chem.AddHs(heavy)
    assert find_rotatable_bonds(all_atoms) == find_rotatable_bonds(heavy)
    order = list(reversed(range(all_atoms.GetNumAtoms())))
    shuffled = Chem.RenumberAtoms(all_atoms, order)
    expected = sorted(tuple(sorted(order.index(index) for index in pair)) for pair in find_rotatable_bonds(heavy))
    assert find_rotatable_bonds(shuffled) == expected


def test_installed_plugin_entry_point_load_is_explicit_and_uses_declared_units(monkeypatch):
    info = ScoringFunctionInfo("entry-test", "custom", "torch", ("pattern",), "inferred", "arbitrary")
    registration = registry.ScoringFunctionRegistration(info, HigherReporter)
    calls = []
    class Entry:
        name = "custom-package"
        def load(self):
            calls.append("loaded")
            return lambda: registration
    monkeypatch.setattr(registry, "entry_points", lambda **kw: [Entry()])
    assert calls == [] and "entry-test" not in {e["name"] for e in list_scoring_functions()}
    assert registry.load_scoring_plugins() == ["entry-test"]
    assert calls == ["loaded"]
    assert isinstance(create_scoring_function("entry-test"), HigherReporter)
    wrong = replace(info, name="wrong-units", units="kcal/mol")
    register_scoring_function(wrong, HigherReporter)
    with pytest.raises(ValueError, match="units disagree"):
        create_scoring_function("wrong-units")


@pytest.mark.parametrize("name", ["ad4", "adgpu"])
def test_map_factory_uses_refinement_device_without_duplicate_configuration(inputs, monkeypatch, name):
    protein, ligand, _ = inputs
    received = {}
    def factory(**options):
        received.update(options)
        return PairwiseScorer("vina")
    original = registry._REGISTRY[name]
    monkeypatch.setitem(registry._REGISTRY, name, registry.ScoringFunctionRegistration(
        replace(original.info, preparation="inferred"), factory,
    ))
    refine_pose(protein, ligand, scorer=name, scorer_options={"maps_prefix": "maps/receptor", "atom_types": ["C"]}, num_steps=0)
    assert received["device"] == "cpu"


@pytest.mark.parametrize("options", [{"map_mode": "vina"}, {"formulation": "standard"}])
def test_adgpu_name_cannot_silently_select_a_different_energy_convention(options):
    with pytest.raises(ValueError, match="raw maps.*adgpu formulation"):
        create_scoring_function("adgpu", maps_prefix="absent", atom_types=["C"], **options)


def test_removed_native_flags_fail_at_cli_parse_time():
    from anchor_dock.cli import build_parser
    parser = build_parser()
    for flag in (["--optimizer", "native"], ["--native-options", "{}"]):
        with pytest.raises(SystemExit):
            parser.parse_args(["refine", "-l", "pose.sdf", *flag])


@pytest.mark.parametrize("optimizer", ["adam", "lbfgs", "cartesian"])
def test_torch_amber_public_refinement_and_json_use_no_openmm_context(inputs, monkeypatch, tmp_path, optimizer):
    openmm = pytest.importorskip("openmm")
    from anchor_dock.core.torch_amber import TorchAmberScorer
    _, _, mol = inputs
    system, force = openmm.System(), openmm.NonbondedForce()
    for charge in (.1, -.1, -.2, .2):
        system.addParticle(12.)
        force.addParticle(charge, .3, .1)
    system.addForce(force)
    positions = torch.cat((torch.tensor(mol.GetConformer().GetPositions()), torch.tensor([[0., 0., 0.]]))).numpy()
    scorer = TorchAmberScorer.from_system(system, positions, [0, 1, 2], atomic_numbers=[6, 6, 8, 6])
    def forbidden(*args, **kwargs):
        raise AssertionError("Torch scoring must not execute an OpenMM Context")
    monkeypatch.setattr(openmm, "Context", forbidden)
    result = refine_pose(None, mol, scorer=scorer, optimizer=optimizer, num_steps=3, output_dir=tmp_path / "torch-amber")
    assert result.metadata["amber"]["runtime"] == "pure-torch"
    assert result.scores["score"]["direction"] == "lower"
    assert "complex_potential" in result.scores
    assert "intermolecular" not in result.scores and "intramolecular" not in result.scores
    assert result.scores["search_energy"]["value"] <= result.initial_scores["search_energy"]["value"]
    saved = json.loads((tmp_path / "torch-amber/result.json").read_text())
    assert saved["scores"] == result.scores


def test_uninstalled_dotted_plugin_dependency_does_not_break_catalog():
    info = ScoringFunctionInfo("missing-plugin", "test", "torch", ("autograd",), "inferred", "arbitrary",
                               optional_modules=("anchor_missing_parent_xyz.submodule",))
    register_scoring_function(info, HigherReporter)
    entry = next(v for v in list_scoring_functions() if v["name"] == info.name)
    assert entry["dependencies_available"] is False
    assert entry["missing_dependencies"] == ["Python module anchor_missing_parent_xyz.submodule"]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_public_adgpu_reuses_supplied_maps_and_prepared_atom_identity(tmp_path, device):
    receptor, ligand = _oracle_prepared_inputs()
    prefix = tmp_path / "receptor"
    atom_types = sorted(set(ligand.prepared.ad4_types))
    header = "GRID_PARAMETER_FILE receptor.gpf\nGRID_DATA_FILE receptor.maps.fld\nMACROMOLECULE receptor.pdbqt\nSPACING 10\nNELEMENTS 4 4 4\nCENTER 0 0 0\n"
    for kind in (*atom_types, "e", "d"):
        Path(f"{prefix}.{kind}.map").write_text(header + "\n".join(["0.01"] * 125) + "\n")
    options = {"maps_prefix": str(prefix), "atom_types": atom_types}
    result = refine_pose(receptor, ligand, scorer="adgpu", scorer_options=options,
                         optimizer="cartesian", num_steps=2, device=device)
    assert result.scorer == "adgpu"
    assert result.metadata["effective_tensor_device"] == device
    assert result.metadata["output_atom_to_input_atom"] == list(ligand.prepared.original_atom_indices)
    assert result.scores["search_energy"]["value"] <= result.initial_scores["search_energy"]["value"]
    saved = result.as_dict()
    assert saved["scores"]["score"]["units"] == "kcal/mol-like"
