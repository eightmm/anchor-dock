import sys
from pathlib import Path

from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anchor_dock.reference import auto_select_mcs_mapping


def _mol(smiles: str):
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def test_auto_selects_single_for_dominant_contiguous_mcs():
    ref = _mol("CC(C)Cc1ccccc1")
    query = _mol("CC(C)Cc1ccccc1O")

    choice = auto_select_mcs_mapping(ref, query)

    assert choice["mode"] == "single"
    assert len(choice["mapping"]) >= 6


def test_auto_selects_multi_for_symmetric_reference():
    ref = _mol("c1ccccc1Cc2ccccc2")
    query = _mol("c1ccccc1")

    choice = auto_select_mcs_mapping(ref, query)

    assert choice["mode"] == "multi"
    assert len(choice["mappings"]) >= 2
