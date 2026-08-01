from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def cys_pdb(tmp_path: Path) -> Path:
    path = tmp_path / "cys_pocket.pdb"
    path.write_text(
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
        "TER\nEND\n"
    )
    return path
