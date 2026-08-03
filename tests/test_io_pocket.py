from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest
from rdkit import Chem

from anchor_dock.core.io import extract_pocket_around_residue, extract_pocket_around_residues

AtomSpec = tuple[str, str, int, str, str, bool, tuple[float, float, float]]
ResidueKey = tuple[str, int, str, str, bool]


def _protein(atoms: Iterable[AtomSpec]) -> Chem.Mol:
    atom_specs = list(atoms)
    editable = Chem.RWMol()
    conformer = Chem.Conformer(len(atom_specs))
    for serial, (element, atom_name, number, residue, chain, hetero, position) in enumerate(
        atom_specs,
        start=1,
    ):
        insertion = ""
        if len(residue) == 4:
            residue, insertion = residue[:3], residue[3]
        atom_index = editable.AddAtom(Chem.Atom(element))
        info = Chem.AtomPDBResidueInfo(
            atom_name.rjust(4),
            serial,
            "",
            residue,
            number,
            chain,
            insertion,
            1.0,
            20.0,
            hetero,
            0,
            0,
        )
        editable.GetAtomWithIdx(atom_index).SetMonomerInfo(info)
        conformer.SetAtomPosition(atom_index, position)
    molecule = editable.GetMol()
    molecule.AddConformer(conformer, assignId=True)
    return molecule


def _residue_atoms(molecule: Chem.Mol) -> dict[ResidueKey, set[str]]:
    result: dict[ResidueKey, set[str]] = {}
    for atom in molecule.GetAtoms():
        info = atom.GetPDBResidueInfo()
        assert info is not None
        key = (
            info.GetResidueName().strip(),
            info.GetResidueNumber(),
            info.GetInsertionCode().strip(),
            info.GetChainId().strip(),
            info.GetIsHeteroAtom(),
        )
        result.setdefault(key, set()).add(info.GetName().strip())
    return result


def _union_protein() -> Chem.Mol:
    return _protein(
        [
            ("N", "N", 1, "ALA", "A", False, (0.0, 0.0, 0.0)),
            ("C", "CA", 1, "ALA", "A", False, (0.5, 0.0, 0.0)),
            ("O", "OG", 2, "SER", "A", False, (1.5, 0.0, 0.0)),
            ("C", "CB", 2, "SER", "A", False, (8.0, 0.0, 0.0)),
            ("C", "CA", 3, "VAL", "A", False, (10.0, 0.0, 0.0)),
            ("N", "N", 4, "GLYA", "B", False, (20.0, 0.0, 0.0)),
            ("C", "CA", 4, "GLYA", "B", False, (20.5, 0.0, 0.0)),
            ("O", "OG1", 5, "THR", "B", False, (21.5, 0.0, 0.0)),
            ("C", "CB", 5, "THR", "B", False, (29.0, 0.0, 0.0)),
            ("C", "CA", 6, "LEU", "B", False, (10.0, 1.0, 0.0)),
        ]
    )


def test_extract_pocket_around_residues_unions_complete_residues() -> None:
    protein = _union_protein()

    pocket = extract_pocket_around_residues(
        protein,
        (spec for spec in ("ALA1:A", "GLY4A:B")),
        cutoff=2.0,
    )

    residues = _residue_atoms(pocket)
    assert residues == {
        ("ALA", 1, "", "A", False): {"N", "CA"},
        ("SER", 2, "", "A", False): {"OG", "CB"},
        ("GLY", 4, "A", "B", False): {"N", "CA"},
        ("THR", 5, "", "B", False): {"OG1", "CB"},
    }
    assert np.asarray(pocket.GetConformer().GetPositions()).shape == (8, 3)


def test_extract_pocket_around_residues_supports_blank_chain_and_deduplicates() -> None:
    protein = _protein(
        [
            ("C", "CA", 10, "ALA", "", False, (0.0, 0.0, 0.0)),
            ("N", "N", 11, "ASN", "", False, (1.0, 0.0, 0.0)),
            ("C", "CA", 10, "ALA", "A", False, (20.0, 0.0, 0.0)),
            ("C", "CA", 12, "PRO", "A", False, (21.0, 0.0, 0.0)),
        ]
    )

    blank_chain = extract_pocket_around_residues(protein, ["ALA10:", "ala10:"], cutoff=2.0)
    assert set(_residue_atoms(blank_chain)) == {
        ("ALA", 10, "", "", False),
        ("ASN", 11, "", "", False),
    }

    wildcard = extract_pocket_around_residue(protein, "ALA10", cutoff=2.0)
    assert set(_residue_atoms(wildcard)) == {
        ("ALA", 10, "", "", False),
        ("ASN", 11, "", "", False),
        ("ALA", 10, "", "A", False),
        ("PRO", 12, "", "A", False),
    }


def test_extract_pocket_around_residues_preserves_heteroatom_option() -> None:
    protein = _protein(
        [
            ("C", "CA", 1, "ALA", "A", False, (0.0, 0.0, 0.0)),
            ("O", "O", 20, "HOH", "A", True, (1.0, 0.0, 0.0)),
            ("C", "C1", 21, "LIG", "A", True, (1.5, 0.0, 0.0)),
        ]
    )

    with_heteroatoms = extract_pocket_around_residues(protein, ["ALA1:A"], cutoff=2.0)
    assert set(_residue_atoms(with_heteroatoms)) == {
        ("ALA", 1, "", "A", False),
        ("HOH", 20, "", "A", True),
        ("LIG", 21, "", "A", True),
    }

    without_heteroatoms = extract_pocket_around_residues(
        protein,
        ["ALA1:A"],
        cutoff=2.0,
        include_heteroatoms=False,
    )
    assert set(_residue_atoms(without_heteroatoms)) == {("ALA", 1, "", "A", False)}


def test_extract_pocket_around_residues_fails_if_any_target_is_missing() -> None:
    protein = _union_protein()

    with pytest.raises(ValueError, match=r"residue LYS999:A not found"):
        extract_pocket_around_residues(protein, ["ALA1:A", "LYS999:A"], cutoff=2.0)


def test_extract_pocket_around_residues_requires_nonempty_ordered_iterable() -> None:
    protein = _union_protein()

    with pytest.raises(ValueError, match="at least one"):
        extract_pocket_around_residues(protein, [], cutoff=2.0)
    with pytest.raises(TypeError, match="not a string"):
        extract_pocket_around_residues(protein, "ALA1:A", cutoff=2.0)
