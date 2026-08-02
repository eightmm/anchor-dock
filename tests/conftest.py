from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

CYS_BLOCK = (
    "ATOM      1  N   CYS A 145       0.000  -1.500   0.000  1.00 20.00           N  \n"
    "ATOM      2  CA  CYS A 145       0.000   0.000   0.000  1.00 20.00           C  \n"
    "ATOM      3  C   CYS A 145       1.500   0.500   0.000  1.00 20.00           C  \n"
    "ATOM      4  O   CYS A 145       2.300  -0.300   0.000  1.00 20.00           O  \n"
    "ATOM      5  CB  CYS A 145      -0.700   0.800   1.100  1.00 20.00           C  \n"
    "ATOM      6  SG  CYS A 145      -1.900   1.700   1.800  1.00 20.00           S  \n"
    "ATOM      7  N   ALA A 146       1.800   1.800   0.000  1.00 20.00           N  \n"
    "ATOM      8  CA  ALA A 146       3.200   2.200   0.000  1.00 20.00           C  \n"
    "ATOM      9  C   ALA A 146       3.800   1.200   1.000  1.00 20.00           C  \n"
    "ATOM     10  O   ALA A 146       5.000   1.100   1.100  1.00 20.00           O  \n"
    "ATOM     11  CB  ALA A 146       3.400   3.700   0.300  1.00 20.00           C  \n"
)


def _shifted_cys_block(serial_start: int, residue: int, shift: float) -> str:
    lines = []
    atoms = [
        ("N", "N", 0.0, -1.5, 0.0),
        ("CA", "C", 0.0, 0.0, 0.0),
        ("C", "C", 1.5, 0.5, 0.0),
        ("O", "O", 2.3, -0.3, 0.0),
        ("CB", "C", -0.7, 0.8, 1.1),
        ("SG", "S", -1.9, 1.7, 1.8),
    ]
    for offset, (name, element, x, y, z) in enumerate(atoms):
        serial = serial_start + offset
        lines.append(
            f"ATOM  {serial:5d} {name:>4s} CYS A{residue:4d}    "
            f"{x + shift:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}  \n"
        )
    return "".join(lines)


@pytest.fixture
def cys_pdb(tmp_path: Path) -> Path:
    path = tmp_path / "cys.pdb"
    path.write_text(CYS_BLOCK + "TER\nEND\n")
    return path


@pytest.fixture
def ambiguous_cys_pdb(tmp_path: Path) -> Path:
    path = tmp_path / "two_cys.pdb"
    path.write_text(_shifted_cys_block(1, 145, 0.0) + _shifted_cys_block(7, 166, 12.0) + "TER\nEND\n")
    return path


@pytest.fixture
def reference_sdf(tmp_path: Path) -> Path:
    mol = Chem.AddHs(Chem.MolFromSmiles("CCOc1ccccc1"))
    assert AllChem.EmbedMolecule(mol, randomSeed=11) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    path = tmp_path / "reference.sdf"
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()
    return path
