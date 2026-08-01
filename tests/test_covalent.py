from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from rdkit import Chem

from anchor_dock import dock_covalent
from anchor_dock.core.io import extract_pocket_around_residue
from anchor_dock.covalent.adduct import create_adduct_template, create_intermolecular_exclusion_mask
from anchor_dock.covalent.anchor import AnchorPoint, detect_warheads, find_reactive_residues


@pytest.fixture
def cys_anchor() -> AnchorPoint:
    return AnchorPoint(
        residue_name="CYS",
        residue_num=145,
        chain_id="A",
        atom_name="SG",
        coord=np.array([1.82, 0.0, 0.0]),
        bond_vector=np.array([1.0, 0.0, 0.0]),
        bond_length=1.82,
        cb_coord=np.zeros(3),
    )


@pytest.mark.parametrize(
    ("smiles", "warhead_type"),
    [
        ("C=CC(=O)N", "acrylamide"),
        ("CC=O", "aldehyde"),
        ("O=C(C(=O)N)C", "alpha_ketoamide"),
        ("OCC1CO1", "epoxide"),
        ("N#Cc1ccccc1", "aryl_nitrile"),
        ("ClCC(=O)N", "chloroacetamide"),
        ("B(O)O", "boronic_acid"),
    ],
)
def test_warhead_detection_and_sanitized_adduct(smiles: str, warhead_type: str, cys_anchor: AnchorPoint) -> None:
    mol = Chem.MolFromSmiles(smiles)
    hits = detect_warheads(mol)
    assert hits and hits[0].warhead_type == warhead_type
    adduct, cb_idx, nuc_idx, reactive_idx = create_adduct_template(mol, hits[0], cys_anchor)
    assert cb_idx is not None
    assert adduct.GetBondBetweenAtoms(nuc_idx, reactive_idx) is not None
    assert Chem.MolToSmiles(adduct)


def test_pocket_extraction_preserves_pdb_metadata(cys_pdb: Path) -> None:
    protein = Chem.MolFromPDBFile(str(cys_pdb), sanitize=False, removeHs=True)
    anchors = find_reactive_residues(protein, "CYS145:A")
    assert len(anchors) == 1 and anchors[0].atom_name == "SG"
    pocket = extract_pocket_around_residue(protein, "CYS145:A", cutoff=12.0)
    assert pocket.GetNumAtoms() == protein.GetNumAtoms()
    assert all(atom.GetPDBResidueInfo() is not None for atom in pocket.GetAtoms())
    assert find_reactive_residues(pocket, "CYS145:A")


def test_exclusion_mask_is_pairwise() -> None:
    ligand = Chem.MolFromSmiles("CCO")
    protein = Chem.MolFromSmiles("CCN")
    mask = create_intermolecular_exclusion_mask(ligand, protein, {0}, {2}, "cpu")
    assert mask.shape == (1, 3, 3)
    assert mask[0, 0].all()
    assert mask[0, :, 2].all()
    assert not mask[0, 1, 1]


def test_end_to_end_covalent_mode(cys_pdb: Path, tmp_path: Path) -> None:
    result = dock_covalent(
        protein_pdb=str(cys_pdb),
        query_ligand="C=CC(=O)NCC",
        reactive_residue="CYS145:A",
        output_dir=str(tmp_path / "out"),
        num_confs=5,
        rmsd_threshold=0.5,
        rotation_scan_step=180,
        rotation_top_k=3,
        optimize=True,
        opt_steps=2,
        opt_batch_size=2,
        device="cpu",
        verbose=False,
    )
    assert result["mode"] == "covalent"
    assert result["warhead_type"] == "acrylamide"
    assert result["num_poses"] > 0
    output = Path(result["output_file"])
    assert output.exists()
    poses = [mol for mol in Chem.SDMolSupplier(str(output), removeHs=False) if mol is not None]
    assert len(poses) == result["num_poses"]
    assert poses[0].GetProp("AnchorDock_Mode") == "covalent"
    assert poses[0].HasProp("CovVina_Warhead_Type")
    assert np.isfinite(float(result["best_score"]))
