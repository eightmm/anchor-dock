from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from rdkit import Chem

from anchor_dock import dock_covalent
from anchor_dock.covalent.adduct import (
    create_adduct_template,
    create_covalent_exclusion_mask,
)
from anchor_dock.covalent.anchor import AnchorPoint, detect_warheads, select_reactive_anchor
from anchor_dock.covalent.pipeline import _prepare_covalent_receptor, clear_covalent_context_cache


@pytest.fixture
def cys_anchor() -> AnchorPoint:
    support = np.zeros(3)
    coordinate = np.array([1.0, 0.0, 0.0])
    return AnchorPoint("CYS", 145, "", "A", "SG", "CB", coordinate, support, coordinate, 1.82, 16)


@pytest.mark.parametrize(
    ("smiles", "warhead_type"),
    [
        ("C=CC(=O)N", "acrylamide"), ("C=CC(=O)O", "acrylic_acid"),
        ("C=CC(=O)OC", "acrylate"), ("C=CC(=O)C", "enone"),
        ("C=CS(=O)(=O)N", "vinyl_sulfonamide"), ("C=CS(=O)(=O)C", "vinyl_sulfone"),
        ("O=C1C=CC(=O)N1", "maleimide"), ("ClCC(=O)N", "chloroacetamide"),
        ("BrCC(=O)N", "bromoacetamide"), ("ICC(=O)N", "iodoacetamide"),
        ("FCC(=O)N", "fluoroacetamide"), ("ClC(F)C(=O)N", "chlorofluoroacetamide"),
        ("CC1CO1", "epoxide"), ("CC1CN1", "aziridine"), ("CC1CS1", "thiirane"),
        ("N#Cc1ccccc1", "aryl_nitrile"), ("N#CCC", "alkyl_nitrile"),
        ("C#CC(=O)N", "propiolamide"), ("CC#CC(=O)N", "propargylamide"),
        ("N#CC=CC(=O)N", "cyanoacrylamide"), ("CSSC", "disulfide"),
        ("FS(=O)(=O)c1ccccc1", "sulfonyl_fluoride"), ("CC(=O)C(=O)N", "alpha_ketoamide"),
        ("CC=O", "aldehyde"), ("CN=C=S", "isothiocyanate"),
        ("CC(=O)ON1C(=O)CCC1=O", "nhs_ester"), ("CC(=O)OC(F)(F)F", "tfe_ester"),
        ("CC(=O)F", "acyl_fluoride"), ("CB(O)O", "boronic_acid"),
        ("CP(=O)(O)O", "phosphonate"),
    ],
)
def test_warhead_products_are_connected_and_sanitized(
    smiles: str,
    warhead_type: str,
    cys_anchor: AnchorPoint,
) -> None:
    mol = Chem.MolFromSmiles(smiles)
    hits = detect_warheads(mol)
    assert hits and hits[0].warhead_type == warhead_type
    adduct, _, nucleophile, reactive = create_adduct_template(mol, hits[0], cys_anchor)
    assert adduct.GetBondBetweenAtoms(nucleophile, reactive) is not None
    assert len(Chem.GetMolFrags(adduct)) == 1
    assert Chem.MolToSmiles(adduct)


def test_ambiguous_automatic_residue_is_rejected(ambiguous_cys_pdb: Path) -> None:
    protein = Chem.MolFromPDBFile(str(ambiguous_cys_pdb), sanitize=False, removeHs=True)
    with pytest.raises(ValueError, match="ambiguous"):
        select_reactive_anchor(protein)


def test_covalent_exclusion_is_narrow() -> None:
    ligand = Chem.MolFromSmiles("CCO")
    receptor = Chem.MolFromSmiles("CCN")
    mask = create_covalent_exclusion_mask(
        ligand,
        receptor,
        pseudo_atom_indices={0},
        reactive_atom_idx=1,
        receptor_nucleophile_idx=2,
        device="cpu",
    )
    assert mask[0, 0].all()
    assert mask[0, 1, 2]
    assert not mask[0, 2, 2]
    assert not mask[0, 1, 1]


def test_covalent_pipeline_preserves_bond_length_and_removes_legacy_tags(
    cys_pdb: Path,
    tmp_path: Path,
) -> None:
    result = dock_covalent(
        cys_pdb,
        "C=CC(=O)NCC",
        "CYS145:A",
        tmp_path / "out",
        num_confs=5,
        rmsd_threshold=0.5,
        rotation_scan_step=180,
        rotation_top_k=3,
        optimize=True,
        opt_steps=2,
        opt_batch_size=3,
        device="cpu",
        verbose=False,
    )
    assert result["covalent_bond_length"] == pytest.approx(1.82)
    poses = [mol for mol in Chem.SDMolSupplier(result["output_file"], removeHs=False) if mol is not None]
    assert poses
    properties = set(poses[0].GetPropNames())
    assert "AnchorDock_Covalent_Bond_Length" in properties
    assert not any(name.startswith("CovVina_") for name in properties)
    assert torch.isfinite(torch.tensor(result["best_score"]))


def test_covalent_receptor_context_is_cached(cys_pdb: Path) -> None:
    clear_covalent_context_cache()
    first = _prepare_covalent_receptor(cys_pdb, "CYS145:A", 12.0, True, "cpu")
    second = _prepare_covalent_receptor(cys_pdb, "CYS145:A", 12.0, True, "cpu")
    assert first is second
