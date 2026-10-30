"""Automatic preparation preserves caller poses and explicit chemical state."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from anchor_dock import preparation
from anchor_dock.core.prepared import parse_pdbqt


def pose(smiles="O[C@H](C)N", seed=19):
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    assert AllChem.EmbedMolecule(molecule, parameters) == 0
    return molecule


def gly_file(tmp_path):
    molecule = Chem.AddHs(Chem.MolFromFASTA("GG"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=7) == 0
    path = tmp_path / "gly.pdb"
    path.write_text(Chem.MolToPDBBlock(Chem.RemoveHs(molecule)))
    return path


def test_core_loader_preserves_order_pose_stereo_and_input_copy(tmp_path):
    source = pose()
    original = np.array(source.GetConformer().GetPositions())
    path = tmp_path / "ligand.sdf"
    writer = Chem.SDWriter(str(path))
    writer.write(source)
    writer.close()
    loaded = preparation.load_ligand_for_refinement(path, generate_coordinates_if_missing=False)
    assert loaded.GetAtomWithIdx(0).GetAtomicNum() == 8
    assert Chem.MolToSmiles(loaded) == Chem.MolToSmiles(source)
    np.testing.assert_allclose(loaded.GetConformer().GetPositions(), original, atol=5.1e-5)
    loaded.GetConformer().SetAtomPosition(0, (999., 0., 0.))
    np.testing.assert_array_equal(source.GetConformer().GetPositions(), original)


def test_coordinate_generation_is_seeded_and_explicitly_disableable():
    first = preparation.load_ligand_for_refinement("CCCO", seed=9)
    second = preparation.load_ligand_for_refinement("CCCO", seed=9)
    np.testing.assert_array_equal(first.GetConformer().GetPositions(), second.GetConformer().GetPositions())
    assert first.GetBoolProp("_AnchorDockCoordinatesGenerated")
    with pytest.raises(ValueError, match="no 3D pose"):
        preparation.load_ligand_for_refinement("CCCO", generate_coordinates_if_missing=False)


def test_2d_embedding_preserves_caller_explicit_hydrogens_and_atom_order():
    molecule = Chem.AddHs(Chem.MolFromSmiles("O[C@H](C)N"))
    AllChem.Compute2DCoords(molecule)
    result = preparation.load_ligand_for_refinement(molecule, add_hydrogens=False)
    assert result.GetNumAtoms() == molecule.GetNumAtoms()
    assert [a.GetAtomicNum() for a in result.GetAtoms()] == [a.GetAtomicNum() for a in molecule.GetAtoms()]
    assert Chem.MolToSmiles(result) == Chem.MolToSmiles(molecule)
    assert result.GetConformer().Is3D()


@pytest.mark.parametrize("seed", [True, -1, 2**31, 1.5])
def test_seed_validation(seed):
    with pytest.raises(ValueError, match="seed"):
        preparation.load_ligand_for_refinement("CC", seed=seed)


def test_missing_file_multiple_molecules_and_conformers_fail(tmp_path):
    with pytest.raises(FileNotFoundError):
        preparation.load_ligand_for_refinement(tmp_path / "missing.sdf")
    with pytest.raises(ValueError, match="connected"):
        preparation.load_ligand_for_refinement("CC.[Na+]")
    molecule = pose()
    molecule.AddConformer(Chem.Conformer(molecule.GetConformer()), assignId=True)
    with pytest.raises(ValueError, match="selected conformer"):
        preparation.load_ligand_for_refinement(molecule)
    path = tmp_path / "multiple.sdf"
    writer = Chem.SDWriter(str(path))
    writer.write(pose("CC"))
    writer.write(pose("CO"))
    writer.close()
    with pytest.raises(ValueError, match="exactly one SDF"):
        preparation.load_ligand_for_refinement(path)


def test_optional_dependency_error_is_actionable(monkeypatch):
    actual = preparation.importlib.import_module

    def unavailable(name):
        if name == "meeko":
            raise ImportError("not installed")
        return actual(name)

    monkeypatch.setattr(preparation.importlib, "import_module", unavailable)
    with pytest.raises(preparation.PreparationDependencyError, match=r"anchor-dock\[prepare\]"):
        preparation.prepare_ligand("CC")


def test_actual_meeko_ligand_retains_coordinates_mapping_and_fingerprint():
    pytest.importorskip("meeko")
    source = pose()
    original = np.array(source.GetConformer().GetPositions())
    result = preparation.prepare_ligand(source, generate_coordinates_if_missing=False)
    np.testing.assert_array_equal(result.source_mol.GetConformer().GetPositions(), original)
    np.testing.assert_allclose(result.prepared.coordinates().numpy(), original[list(result.prepared.original_atom_indices)], atol=.00051)
    assert result.prepared.ad4_types[0] == "OA"
    assert "HD" in result.prepared.ad4_types
    assert Chem.MolToSmiles(result.source_mol) == Chem.MolToSmiles(source)
    assert result.metadata["charge_model"] == "gasteiger"
    assert result.metadata["coordinates_generated"] is False
    assert result.metadata["input_atom_count"] == source.GetNumAtoms()
    assert result.metadata["output_atom_to_input_atom"] == list(result.prepared.original_atom_indices)
    fingerprint = result.fingerprint
    result.metadata["charge_model"] = "changed"
    result.source_mol.GetConformer().SetAtomPosition(0, (999., 0., 0.))
    assert result.fingerprint == fingerprint
    np.testing.assert_array_equal(source.GetConformer().GetPositions(), original)


def test_preparation_maps_added_polar_hydrogens_to_none():
    pytest.importorskip("meeko")
    source = Chem.RemoveHs(pose())
    result = preparation.prepare_ligand(source, generate_coordinates_if_missing=False)
    mapping = result.metadata["output_atom_to_input_atom"]
    assert result.metadata["input_atom_count"] == source.GetNumAtoms()
    for atom, original in zip(result.mol.GetAtoms(), mapping, strict=True):
        if atom.GetAtomicNum() == 1:
            assert original is None
        else:
            assert original == atom.GetIdx()
            assert source.GetAtomWithIdx(original).GetAtomicNum() == atom.GetAtomicNum()


def test_undeclared_terminal_chemical_state_fails_with_atom_identity(tmp_path):
    pytest.importorskip("meeko")
    with pytest.raises(ValueError, match="unresolved template links.*A:1:N"):
        preparation.prepare_receptor(gly_file(tmp_path))


def test_actual_meeko_receptor_stitches_graph_and_preserves_heavy_pose(tmp_path):
    pytest.importorskip("meeko")
    path = gly_file(tmp_path)
    result = preparation.prepare_receptor(path, set_template={"A:1": "NGLY", "A:2": "CGLY"})
    context = result.context()
    assert context.features["typing_ambiguities"] == ()
    assert context.features["typing_version"] == "prepared-ad4-xs-v1"
    assert len(Chem.GetMolFrags(result.mol)) == 1
    assert result.metadata["selected_templates"] == {"A:1": "NGLY", "A:2": "CGLY"}
    assert result.metadata["scope"] == "entire supplied receptor before pocket selection"
    raw = Chem.MolFromPDBFile(str(path), sanitize=False, removeHs=False)
    raw_heavy = sorted(map(tuple, raw.GetConformer().GetPositions()))
    prepared_heavy = sorted(tuple(result.mol.GetConformer().GetAtomPosition(a.GetIdx()))
                            for a in result.mol.GetAtoms() if a.GetAtomicNum() != 1)
    np.testing.assert_allclose(raw_heavy, prepared_heavy, atol=1e-12)
    for atom in result.mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        info = atom.GetPDBResidueInfo()
        matches = [a for a in raw.GetAtoms() if a.GetPDBResidueInfo().GetResidueNumber() == info.GetResidueNumber()
                   and a.GetPDBResidueInfo().GetName().strip() == info.GetName().strip()]
        assert len(matches) == 1
        np.testing.assert_allclose(result.mol.GetConformer().GetAtomPosition(atom.GetIdx()),
                                   raw.GetConformer().GetAtomPosition(matches[0].GetIdx()), atol=1e-12)
    fingerprint = result.fingerprint
    result.serial_to_atom_index.clear()
    result.metadata["selected_templates"].clear()
    assert result.fingerprint == fingerprint


def test_pqr_supplied_charges_survive_meeko_residue_rectification(tmp_path):
    pytest.importorskip("meeko")
    receptor = preparation.prepare_receptor(gly_file(tmp_path), set_template={"A:1": "NGLY", "A:2": "CGLY"})
    graph = receptor.mol
    lines = []
    nonpolar_hydrogens = 0
    for i, atom in enumerate(graph.GetAtoms()):
        info = atom.GetPDBResidueInfo()
        xyz = graph.GetConformer().GetAtomPosition(i)
        charge = float(atom.GetFormalCharge())
        if atom.GetAtomicNum() == 1 and atom.GetNeighbors()[0].GetAtomicNum() == 6:
            charge = .01
            nonpolar_hydrogens += 1
        if info.GetResidueNumber() == 1 and info.GetName().strip() == "N":
            charge += .017
        lines.append(f"ATOM {i+1} {info.GetName().strip()} {info.GetResidueName()} {info.GetChainId()} "
                     f"{info.GetResidueNumber()} {xyz.x:.3f} {xyz.y:.3f} {xyz.z:.3f} {charge:.6f} 1.5")
    path = tmp_path / "gly.pqr"
    path.write_text("\n".join(lines) + "\n")
    prepared = preparation.prepare_receptor(path, set_template={"A:1": "NGLY", "A:2": "CGLY"})
    records = parse_pdbqt(prepared.pdbqt, rigid=True).atoms
    assert sum(a.charge for a in records) == pytest.approx(.017 + .01 * nonpolar_hydrogens, abs=1e-12)
    assert max(a.charge for a in records) == 1.017
    assert prepared.metadata["charge_model"] == "read"
    assert "not substituted" in prepared.metadata["source_radius_role"]
    with pytest.raises(ValueError, match="PQR preserves"):
        preparation.prepare_receptor(path, charge_model="gasteiger")


@pytest.mark.parametrize("case", ["altloc", "unknown", "his", "model"])
def test_receptor_unknown_states_fail_before_external_template_fetch(tmp_path, monkeypatch, case):
    meeko = pytest.importorskip("meeko")
    path = gly_file(tmp_path)
    text = path.read_text()
    if case == "altloc":
        text = "\n".join(line[:16] + "A" + line[17:] if line.startswith("ATOM") else line for line in text.splitlines())
    elif case == "unknown":
        text = text.replace("GLY", "UNK")
    elif case == "his":
        text = text.replace("GLY", "HIS")
    else:
        text = "MODEL        1\n" + text + "ENDMDL\nMODEL        2\n" + text + "ENDMDL\n"
    path.write_text(text)

    def unexpected(*args, **kwargs):
        raise AssertionError("unsafe input reached Meeko template construction")

    monkeypatch.setattr(meeko.Polymer, "from_pdb_string", unexpected)
    with pytest.raises(ValueError, match={"altloc": "alternate location", "unknown": "unknown residues", "his": "HIS protonation", "model": "one receptor model"}[case]):
        preparation.prepare_receptor(path)


def test_autogrid_input_encloses_requested_box_without_claiming_execution(tmp_path):
    pytest.importorskip("meeko")
    receptor = preparation.prepare_receptor(gly_file(tmp_path), set_template={"A:1": "NGLY", "A:2": "CGLY"})
    text = receptor.autogrid_gpf(center=[1., 2., 3.], size=[10.1, 12.2, 14.3], ligand_types=["C", "OA"])
    intervals = [int(v) for v in text.splitlines()[0].split()[1:]]
    assert all(n % 2 == 0 for n in intervals)
    assert all(n * .375 >= size for n, size in zip(intervals, [10.1, 12.2, 14.3], strict=True))
    assert "ligand_types C OA" in text
    assert "map receptor.OA.map" in text
    assert "receptor receptor.pdbqt" in text
    assert not (tmp_path / "receptor.C.map").exists()
    text = receptor.autogrid_gpf(center=[.00049, 0., 0.], size=[.75] * 3, ligand_types=["C"])
    lines = {line.split()[0]: line.split()[1:] for line in text.splitlines()}
    for requested, actual, count in zip([.00049, 0., 0.], map(float, lines["gridcenter"]), map(int, lines["npts"]), strict=True):
        assert actual - count * .375 / 2 <= requested - .75 / 2
        assert actual + count * .375 / 2 >= requested + .75 / 2
    with pytest.raises(ValueError, match="filenames"):
        receptor.autogrid_gpf(center=[0, 0, 0], size=[10]*3, ligand_types=["C"], map_prefix="bad\nreceptor")
