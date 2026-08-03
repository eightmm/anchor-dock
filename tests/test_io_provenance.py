from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from anchor_dock.core.io import load_ligand, load_reference_ligand


def _write_sdf(path: Path, mols: list[Chem.Mol]) -> None:
    writer = Chem.SDWriter(str(path))
    for mol in mols:
        writer.write(mol)
    writer.close()


def _embedded_ethanol() -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(mol, randomSeed=3) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def test_load_ligand_rejects_multi_record_sdf(tmp_path: Path) -> None:
    path = tmp_path / "two.sdf"
    _write_sdf(path, [_embedded_ethanol(), _embedded_ethanol()])
    with pytest.raises(ValueError, match="dock_batch"):
        load_ligand(path)


def test_load_reference_ligand_rejects_multi_record_sdf(tmp_path: Path) -> None:
    path = tmp_path / "two.sdf"
    _write_sdf(path, [_embedded_ethanol(), _embedded_ethanol()])
    with pytest.raises(ValueError, match="dock_batch"):
        load_reference_ligand(path)


def test_single_sdf_reader_counts_malformed_second_record_and_stops(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "many.sdf"
    path.write_text("supplier is replaced by the test")
    consumed = 0

    class TwoRecordsThenFail:
        def __iter__(self):
            nonlocal consumed
            consumed += 1
            yield _embedded_ethanol()
            consumed += 1
            yield None
            raise AssertionError("single-molecule loading must stop after the second SDF record")

    def fake_supplier(source: str, *, removeHs: bool):
        del source, removeHs
        return TwoRecordsThenFail()

    monkeypatch.setattr("anchor_dock.core.io.Chem.SDMolSupplier", fake_supplier)
    with pytest.raises(ValueError, match="more than one.*dock_batch"):
        load_ligand(path)
    assert consumed == 2


def test_load_ligand_rejects_zero_record_sdf(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "empty.sdf"
    path.write_text("supplier is replaced by the test")
    monkeypatch.setattr("anchor_dock.core.io.Chem.SDMolSupplier", lambda *args, **kwargs: iter(()))
    with pytest.raises(ValueError, match="found 0"):
        load_ligand(path)


_MALFORMED_SDF_RECORD = "bad\n     RDKit          2D\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\nM  END\n$$$$\n"


def test_load_ligand_single_malformed_record_keeps_read_failure(tmp_path: Path) -> None:
    path = tmp_path / "malformed.sdf"
    path.write_text(_MALFORMED_SDF_RECORD)
    with pytest.raises(ValueError, match="failed to read molecule"):
        load_ligand(path)


def test_load_reference_ligand_single_malformed_record_keeps_read_failure(tmp_path: Path) -> None:
    path = tmp_path / "malformed.sdf"
    path.write_text(_MALFORMED_SDF_RECORD)
    with pytest.raises(ValueError, match="failed to read molecule"):
        load_reference_ligand(path)
