from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from anchor_dock import dock_free
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


def _fake_sparse_conformer_generator(mol, device, *, num_confs, rmsd_threshold, add_hydrogens, random_seed, **kwargs):
    working = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    conformer_ids = list(AllChem.EmbedMultipleConfs(working, numConfs=2, params=params))
    assert len(conformer_ids) == 2
    heavy = Chem.RemoveHs(working)
    heavy_conformers = list(heavy.GetConformers())
    assert len(heavy_conformers) == 2
    target_ids = [7, 19]
    for conformer, new_id in zip(heavy_conformers, target_ids, strict=True):
        conformer.SetId(new_id)
    return heavy, target_ids


def test_free_docking_tags_true_conformer_and_representative_index(
    monkeypatch,
    cys_pdb: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("anchor_dock.free.generate_conformers_and_cluster", _fake_sparse_conformer_generator)
    result = dock_free(
        cys_pdb,
        "CCO",
        tmp_path / "free-provenance",
        num_confs=2,
        num_starts=4,
        optimize=False,
        top_k=None,
        device="cpu",
        verbose=False,
    )
    poses = [mol for mol in Chem.SDMolSupplier(result["output_file"]) if mol is not None]
    assert poses
    for pose in poses:
        pose_id = pose.GetProp("AnchorDock_Pose_ID")
        start_index = int(pose_id.split("_")[1])
        representative_index = start_index % 2
        expected_conformer_id = [7, 19][representative_index]
        assert pose.GetProp("AnchorDock_Source_Representative_Index") == str(representative_index)
        assert pose.GetProp("AnchorDock_Source_Conformer") == str(expected_conformer_id)
