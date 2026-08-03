from __future__ import annotations

import math

import pytest
import torch
from rdkit import Chem

from anchor_dock.interaction import (
    InvalidSmartsError,
    ReceptorAtomNotFoundError,
    flat_bottom_distance_restraint,
    interaction_distances,
    select_ligand_anchors,
    select_receptor_atom,
)

# --- Receptor Atom Selector Tests -------------------------------------------------------------

def test_select_receptor_atom_success(tmp_path) -> None:
    pdb_content = (
        "ATOM      1  N   ASP A 189      10.000  10.000  10.000  1.00 20.00           N  \n"
        "ATOM      2  CA  ASP A 189      11.000  11.000  11.000  1.00 20.00           C  \n"
        "ATOM      3  C   ASP A 189      12.000  12.000  12.000  1.00 20.00           C  \n"
        "ATOM      4  O   ASP A 189      13.000  13.000  13.000  1.00 20.00           O  \n"
        "ATOM      5  CB  ASP A 189      14.000  14.000  14.000  1.00 20.00           C  \n"
        "ATOM      6  CG  ASP A 189      15.000  15.000  15.000  1.00 20.00           C  \n"
        "ATOM      7  OD1 ASP A 189      16.000  16.000  16.000  1.00 20.00           O  \n"
        "ATOM      8  OD2 ASP A 189      17.000  17.000  17.000  1.00 20.00           O  \n"
        "TER\n"
    )
    pdb_file = tmp_path / "receptor.pdb"
    pdb_file.write_text(pdb_content)

    rdkit_receptor = Chem.MolFromPDBFile(str(pdb_file), sanitize=False, removeHs=False)
    assert rdkit_receptor is not None

    sel = select_receptor_atom(pdb_file, rdkit_receptor, "ASP189:A", "OD1")
    assert sel.residue_id == "ASP189:A"
    assert sel.residue_name == "ASP"
    assert sel.residue_number == 189
    assert sel.atom_name == "OD1"
    assert sel.element == "O"
    assert sel.chain == "A"
    assert sel.insertion_code == ""
    assert sel.occupancy == 1.0
    assert sel.coordinate == (16.0, 16.0, 16.0)

    atom = rdkit_receptor.GetAtomWithIdx(sel.rdkit_index)
    info = atom.GetMonomerInfo()
    assert info.GetName().strip() == "OD1"
    assert info.GetSerialNumber() == sel.pdb_serial


def test_select_receptor_atom_insertion_code(tmp_path) -> None:
    pdb_content = (
        "ATOM      1  CA  ASP A 189A     10.000  10.000  10.000  1.00 20.00           C  \n"
        "TER\n"
    )
    pdb_file = tmp_path / "receptor.pdb"
    pdb_file.write_text(pdb_content)
    rdkit_receptor = Chem.MolFromPDBFile(str(pdb_file), sanitize=False, removeHs=False)

    sel = select_receptor_atom(pdb_file, rdkit_receptor, "ASP189A:A", "CA")
    assert sel.residue_id == "ASP189A:A"
    assert sel.chain == "A"
    assert sel.insertion_code == "A"
    assert sel.coordinate == (10.0, 10.0, 10.0)


def test_select_receptor_atom_ambiguous_chain(tmp_path) -> None:
    pdb_content = (
        "ATOM      1  CA  ASP A 189      10.000  10.000  10.000  1.00 20.00           C  \n"
        "ATOM      2  CA  ASP B 189      20.000  20.000  20.000  1.00 20.00           C  \n"
        "TER\n"
    )
    pdb_file = tmp_path / "receptor.pdb"
    pdb_file.write_text(pdb_content)
    rdkit_receptor = Chem.MolFromPDBFile(str(pdb_file), sanitize=False, removeHs=False)

    sel = select_receptor_atom(pdb_file, rdkit_receptor, "ASP189:A", "CA")
    assert sel.chain == "A"

    sel2 = select_receptor_atom(pdb_file, rdkit_receptor, "ASP189:B", "CA")
    assert sel2.chain == "B"

    with pytest.raises(ValueError, match="ambiguous"):
        select_receptor_atom(pdb_file, rdkit_receptor, "ASP189", "CA")


def test_select_receptor_atom_omitted_chain_unambiguous(tmp_path) -> None:
    pdb_content = (
        "ATOM      1  CA  ASP A 189      10.000  10.000  10.000  1.00 20.00           C  \n"
        "TER\n"
    )
    pdb_file = tmp_path / "receptor.pdb"
    pdb_file.write_text(pdb_content)
    rdkit_receptor = Chem.MolFromPDBFile(str(pdb_file), sanitize=False, removeHs=False)

    sel = select_receptor_atom(pdb_file, rdkit_receptor, "ASP189", "CA")
    assert sel.chain == "A"
    assert sel.residue_id == "ASP189:A"


def test_select_receptor_atom_blank_chain(tmp_path) -> None:
    pdb_content = (
        "ATOM      1  CA  ASP   189      10.000  10.000  10.000  1.00 20.00           C  \n"
        "TER\n"
    )
    pdb_file = tmp_path / "receptor.pdb"
    pdb_file.write_text(pdb_content)
    rdkit_receptor = Chem.MolFromPDBFile(str(pdb_file), sanitize=False, removeHs=False)

    sel = select_receptor_atom(pdb_file, rdkit_receptor, "ASP189:", "CA")
    assert sel.chain == ""
    assert sel.residue_id == "ASP189:"

    sel2 = select_receptor_atom(pdb_file, rdkit_receptor, "ASP189", "CA")
    assert sel2.chain == ""
    assert sel2.residue_id == "ASP189:"


def test_select_receptor_atom_altloc_rejection(tmp_path) -> None:
    pdb_content = (
        "ATOM      1  CA AASP A 189      10.000  10.000  10.000  0.50 20.00           C  \n"
        "ATOM      2  CA BASP A 189      10.100  10.100  10.100  0.50 20.00           C  \n"
        "TER\n"
    )
    pdb_file = tmp_path / "receptor.pdb"
    pdb_file.write_text(pdb_content)
    rdkit_receptor = Chem.MolFromPDBFile(str(pdb_file), sanitize=False, removeHs=False)

    with pytest.raises(ValueError, match="alternate location"):
        select_receptor_atom(pdb_file, rdkit_receptor, "ASP189:A", "CA")


def test_select_receptor_atom_hydrogen_rejection(tmp_path) -> None:
    pdb_content = (
        "ATOM      1  H   ASP A 189      10.000  10.000  10.000  1.00 20.00           H  \n"
        "TER\n"
    )
    pdb_file = tmp_path / "receptor.pdb"
    pdb_file.write_text(pdb_content)
    rdkit_receptor = Chem.MolFromPDBFile(str(pdb_file), sanitize=False, removeHs=False)

    with pytest.raises(ValueError, match="heavy atom"):
        select_receptor_atom(pdb_file, rdkit_receptor, "ASP189:A", "H")


def test_select_receptor_atom_duplicate_rejection(tmp_path) -> None:
    pdb_content = (
        "ATOM      1  CA  ASP A 189      10.000  10.000  10.000  1.00 20.00           C  \n"
        "ATOM      2  CA  ASP A 189      10.000  10.000  10.000  1.00 20.00           C  \n"
        "TER\n"
    )
    pdb_file = tmp_path / "receptor.pdb"
    pdb_file.write_text(pdb_content)
    rdkit_receptor = Chem.MolFromPDBFile(str(pdb_file), sanitize=False, removeHs=False)

    with pytest.raises(ValueError, match="ambiguous or duplicate"):
        select_receptor_atom(pdb_file, rdkit_receptor, "ASP189:A", "CA")


def test_select_receptor_atom_nonfinite_rejection(tmp_path) -> None:
    pdb_content = (
        "ATOM      1  CA  ASP A 189         nan  10.000  10.000  1.00 20.00           C  \n"
        "TER\n"
    )
    pdb_file = tmp_path / "receptor.pdb"
    pdb_file.write_text(pdb_content)
    rdkit_receptor = Chem.MolFromPDBFile(str(pdb_file), sanitize=False, removeHs=False)

    with pytest.raises(ValueError, match="non-finite"):
        select_receptor_atom(pdb_file, rdkit_receptor, "ASP189:A", "CA")


def test_select_receptor_atom_missing_or_invalid_syntax(tmp_path) -> None:
    pdb_content = (
        "ATOM      1  CA  ASP A 189      10.000  10.000  10.000  1.00 20.00           C  \n"
        "TER\n"
    )
    pdb_file = tmp_path / "receptor.pdb"
    pdb_file.write_text(pdb_content)
    rdkit_receptor = Chem.MolFromPDBFile(str(pdb_file), sanitize=False, removeHs=False)

    # Invalid residue syntax
    with pytest.raises(ValueError, match="invalid receptor residue"):
        select_receptor_atom(pdb_file, rdkit_receptor, "ASP:A", "CA")

    # Missing residue
    with pytest.raises(ReceptorAtomNotFoundError, match="no ATOM record matches"):
        select_receptor_atom(pdb_file, rdkit_receptor, "ALA200:A", "CA")

    # Missing atom
    with pytest.raises(ReceptorAtomNotFoundError, match="no ATOM record matches"):
        select_receptor_atom(pdb_file, rdkit_receptor, "ASP189:A", "O")


# --- Ligand Anchor Selector Tests -------------------------------------------------------------

def test_select_ligand_anchors_success() -> None:
    mol = Chem.MolFromSmiles("CCO")
    matches = select_ligand_anchors(mol, "[O:1]")
    assert len(matches) == 1
    m = matches[0]
    assert m.match_index == 0
    assert m.element == "O"
    assert m.formal_charge == 0
    assert m.ligand_atom_index == 2
    assert m.representative_match == (2,)


def test_select_ligand_anchors_symmetric_collapse() -> None:
    mol = Chem.MolFromSmiles("c1ccccc1")
    matches = select_ligand_anchors(mol, "[c:1]~[c]")
    assert len(matches) == 6
    indices = [m.ligand_atom_index for m in matches]
    assert indices == [0, 1, 2, 3, 4, 5]
    for i, m in enumerate(matches):
        assert m.match_index == i


def test_select_ligand_anchors_invalid_patterns() -> None:
    mol = Chem.MolFromSmiles("CCO")

    with pytest.raises(InvalidSmartsError, match="invalid ligand SMARTS"):
        select_ligand_anchors(mol, "[invalid")

    with pytest.raises(InvalidSmartsError, match="exactly one mapped atom"):
        select_ligand_anchors(mol, "[C]CO")

    with pytest.raises(InvalidSmartsError, match="exactly one mapped atom"):
        select_ligand_anchors(mol, "[C:1]~[O:2]")

    with pytest.raises(InvalidSmartsError, match="must be :1"):
        select_ligand_anchors(mol, "[C:2]")

    with pytest.raises(InvalidSmartsError, match="heavy atom"):
        select_ligand_anchors(mol, "[#1:1]")

    with pytest.raises(ValueError, match="has no match"):
        select_ligand_anchors(mol, "[F:1]")


def test_select_ligand_anchors_max_matches() -> None:
    mol = Chem.MolFromSmiles("c1ccccc1")

    with pytest.raises(ValueError, match="exceeding max_matches"):
        select_ligand_anchors(mol, "[c:1]", max_matches=3)

    with pytest.raises(ValueError, match="positive"):
        select_ligand_anchors(mol, "[c:1]", max_matches=-1)


# --- Restraint Primitive Tests ---------------------------------------------------------------

def test_interaction_distances() -> None:
    coords = torch.tensor([
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
    ], dtype=torch.float32)

    receptor_coord = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

    distances = interaction_distances(coords, 2, receptor_coord)
    assert distances.shape == (2,)
    assert torch.allclose(distances[0], torch.tensor(math.sqrt(3.0)))
    assert torch.allclose(distances[1], torch.tensor(math.sqrt(27.0)))

    # Bad shape
    with pytest.raises(ValueError, match="shape"):
        interaction_distances(coords[0], 2, receptor_coord)

    # Bad index
    with pytest.raises(ValueError, match="out of bounds"):
        interaction_distances(coords, 10, receptor_coord)

    # Non-finite coordinates
    coords_nan = coords.clone()
    coords_nan[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        interaction_distances(coords_nan, 2, receptor_coord)

    receptor_nan = torch.tensor([1.0, float("inf"), 1.0])
    with pytest.raises(ValueError, match="finite"):
        interaction_distances(coords, 2, receptor_nan)


def test_flat_bottom_distance_restraint() -> None:
    distances = torch.tensor([3.0, 3.2, 2.7, 2.0, 4.0], dtype=torch.float32)
    penalties = flat_bottom_distance_restraint(
        distances,
        target_distance=3.0,
        distance_tolerance=0.5,
        restraint_weight=10.0
    )

    assert torch.allclose(penalties[0], torch.tensor(0.0))
    assert torch.allclose(penalties[1], torch.tensor(0.0))
    assert torch.allclose(penalties[2], torch.tensor(0.0))
    assert torch.allclose(penalties[3], torch.tensor(2.5))
    assert torch.allclose(penalties[4], torch.tensor(2.5))

    # Invalid arguments
    with pytest.raises(ValueError, match="target_distance"):
        flat_bottom_distance_restraint(distances, target_distance=-1.0, distance_tolerance=0.5, restraint_weight=10.0)

    with pytest.raises(ValueError, match="distance_tolerance"):
        flat_bottom_distance_restraint(distances, target_distance=3.0, distance_tolerance=-0.5, restraint_weight=10.0)

    with pytest.raises(ValueError, match="distance_tolerance"):
        flat_bottom_distance_restraint(distances, target_distance=3.0, distance_tolerance=4.0, restraint_weight=10.0)

    with pytest.raises(ValueError, match="weight"):
        flat_bottom_distance_restraint(distances, target_distance=3.0, distance_tolerance=0.5, restraint_weight=-1.0)


def test_flat_bottom_restraint_gradient_flow() -> None:
    # Test gradient direction for values above upper tolerance limit
    coords = torch.tensor([[[0.0, 0.0, 4.0]]], dtype=torch.float32, requires_grad=True)
    receptor_coord = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    distances = interaction_distances(coords, 0, receptor_coord)
    penalties = flat_bottom_distance_restraint(
        distances,
        target_distance=3.0,
        distance_tolerance=0.5,
        restraint_weight=10.0
    )

    loss = penalties.sum()
    loss.backward()

    assert coords.grad is not None
    z_grad = coords.grad[0, 0, 2].item()
    assert z_grad > 0.0
    assert abs(coords.grad[0, 0, 0].item()) < 1e-6
    assert abs(coords.grad[0, 0, 1].item()) < 1e-6

    # Test gradient direction for values below lower tolerance limit
    coords2 = torch.tensor([[[0.0, 0.0, 2.0]]], dtype=torch.float32, requires_grad=True)
    distances2 = interaction_distances(coords2, 0, receptor_coord)
    penalties2 = flat_bottom_distance_restraint(
        distances2,
        target_distance=3.0,
        distance_tolerance=0.5,
        restraint_weight=10.0
    )
    loss2 = penalties2.sum()
    loss2.backward()

    assert coords2.grad is not None
    z_grad2 = coords2.grad[0, 0, 2].item()
    assert z_grad2 < 0.0
