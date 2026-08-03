from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from rdkit import Chem

from anchor_dock import dock_covalent
from anchor_dock.core.features import compute_atom_features
from anchor_dock.core.io import ReceptorContext, receptor_context_from_mol
from anchor_dock.core.scoring import VINA_CONFIG, prepare_interaction_matrices
from anchor_dock.covalent.adduct import (
    create_adduct_template,
    create_covalent_exclusion_mask,
    find_receptor_nucleophile_index,
    select_formed_bond_geometry,
)
from anchor_dock.covalent.anchor import REACTIVE_RESIDUES, AnchorPoint, detect_warheads, select_reactive_anchor
from anchor_dock.covalent.pipeline import (
    COVALENT_RECEPTOR_TYPING_VERSION,
    _prepare_covalent_receptor,
    _product_state_receptor_context,
    clear_covalent_context_cache,
)
from anchor_dock.covalent.pipeline import dock_covalent as dock_covalent_canonical


@pytest.fixture
def cys_anchor() -> AnchorPoint:
    support = np.zeros(3)
    coordinate = np.array([1.0, 0.0, 0.0])
    return AnchorPoint("CYS", 145, "", "A", "SG", "CB", coordinate, support, coordinate, 1.82, 16)


def _minimal_reactive_receptor(residue: str) -> tuple[ReceptorContext, AnchorPoint]:
    config = REACTIVE_RESIDUES[residue]
    element = Chem.GetPeriodicTable().GetElementSymbol(config.atomic_number)
    lines = [
        f"ATOM      1 {config.support_atom_name:>4} {residue:>3} A   1       0.000   0.000   0.000  1.00 20.00           C  ",
        f"ATOM      2 {config.atom_name:>4} {residue:>3} A   1       1.500   0.000   0.000  1.00 20.00          {element:>2}  ",
    ]
    if residue == "HIS":
        lines.append("ATOM      3  CD2 HIS A   1       2.000   1.200   0.000  1.00 20.00           C  ")
    lines.extend(["TER", "END", ""])
    block = "\n".join(lines)
    molecule = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=True)
    assert molecule is not None
    anchor = select_reactive_anchor(molecule, f"{residue}1:A")
    receptor = receptor_context_from_mol(
        molecule,
        "cpu",
        source_path="test.pdb",
        source_fingerprint="sha256:test",
    )
    return receptor, anchor


@pytest.mark.parametrize(
    ("smiles", "warhead_type"),
    [
        ("C=CC(=O)N", "acrylamide"),
        ("C=CC(=O)O", "acrylic_acid"),
        ("C=CC(=O)OC", "acrylate"),
        ("C=CC(=O)C", "enone"),
        ("C=CS(=O)(=O)N", "vinyl_sulfonamide"),
        ("C=CS(=O)(=O)C", "vinyl_sulfone"),
        ("O=C1C=CC(=O)N1", "maleimide"),
        ("ClCC(=O)N", "chloroacetamide"),
        ("BrCC(=O)N", "bromoacetamide"),
        ("ICC(=O)N", "iodoacetamide"),
        ("FCC(=O)N", "fluoroacetamide"),
        ("ClC(F)C(=O)N", "chlorofluoroacetamide"),
        ("C1CO1", "epoxide"),
        ("C1CN1", "aziridine"),
        ("C1CS1", "thiirane"),
        ("N#Cc1ccccc1", "aryl_nitrile"),
        ("N#CCC", "alkyl_nitrile"),
        ("C#CC(=O)N", "propiolamide"),
        ("C=C(C#N)C(=O)N", "cyanoacrylamide"),
        ("CSSC", "disulfide"),
        ("FS(=O)(=O)c1ccccc1", "sulfonyl_fluoride"),
        ("CC(=O)C(=O)N", "alpha_ketoamide"),
        ("CC=O", "aldehyde"),
        ("CN=C=S", "isothiocyanate"),
        ("CC(=O)ON1C(=O)CCC1=O", "nhs_ester"),
        ("CC(=O)OC(F)(F)F", "tfe_ester"),
        ("CC(=O)F", "acyl_fluoride"),
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
    if warhead_type == "chlorofluoroacetamide":
        assert any(atom.GetAtomicNum() == 9 for atom in adduct.GetAtoms())


def test_ring_opening_detects_both_carbons_and_prefers_less_substituted() -> None:
    parent = Chem.MolFromSmiles("C1CO1")
    parent_hits = [hit for hit in detect_warheads(parent) if hit.warhead_type == "epoxide"]
    assert {hit.reactive_atom_idx for hit in parent_hits} == {0, 1}

    substituted = Chem.MolFromSmiles("CC1CO1")
    hits = [hit for hit in detect_warheads(substituted) if hit.warhead_type == "epoxide"]
    assert len(hits) == 2
    assert (
        substituted.GetAtomWithIdx(hits[0].reactive_atom_idx).GetDegree()
        < substituted.GetAtomWithIdx(hits[1].reactive_atom_idx).GetDegree()
    )


@pytest.mark.parametrize("smiles", ["C=C(C#N)C(=O)N", "N#CC=CC(=O)N"])
def test_cyanoacrylamide_maps_michael_beta_carbon(
    smiles: str,
    cys_anchor: AnchorPoint,
) -> None:
    mol = Chem.MolFromSmiles(smiles)
    hit = detect_warheads(mol)[0]
    assert hit.warhead_type == "cyanoacrylamide"
    reactive = mol.GetAtomWithIdx(hit.reactive_atom_idx)
    alkene_bond = next(bond for bond in reactive.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE)
    alpha = alkene_bond.GetOtherAtom(reactive)
    assert any(
        neighbor.GetAtomicNum() == 6 and any(bond.GetBondType() == Chem.BondType.TRIPLE for bond in neighbor.GetBonds())
        for neighbor in alpha.GetNeighbors()
    ) or any(
        neighbor.GetAtomicNum() == 6 and any(bond.GetBondType() == Chem.BondType.TRIPLE for bond in neighbor.GetBonds())
        for neighbor in reactive.GetNeighbors()
    )

    adduct, _, nucleophile_idx, adduct_reactive_idx = create_adduct_template(mol, hit, cys_anchor)
    assert adduct.GetBondBetweenAtoms(nucleophile_idx, adduct_reactive_idx) is not None
    assert all(
        bond.GetBondType() != Chem.BondType.DOUBLE for bond in adduct.GetAtomWithIdx(adduct_reactive_idx).GetBonds()
    )


def test_untyped_boronic_acid_is_not_advertised_as_supported() -> None:
    assert detect_warheads(Chem.MolFromSmiles("CB(O)O")) == []


@pytest.mark.parametrize(
    ("smiles", "residue", "atomic_number", "carbon_length"),
    [
        ("CP(=O)(O)O", "SER", 8, 1.43),
        ("FS(=O)(=O)c1ccccc1", "SER", 8, 1.43),
        ("FS(=O)(=O)c1ccccc1", "CYS", 16, 1.82),
    ],
)
def test_noncarbon_electrophiles_use_pair_specific_bond_target(
    smiles: str,
    residue: str,
    atomic_number: int,
    carbon_length: float,
) -> None:
    anchor = AnchorPoint(
        residue,
        1,
        "",
        "A",
        "X",
        "Y",
        np.ones(3),
        np.zeros(3),
        np.ones(3),
        carbon_length,
        atomic_number,
    )
    ligand = Chem.MolFromSmiles(smiles)
    hit = detect_warheads(ligand)[0]
    adduct, _, nucleophile_idx, reactive_idx = create_adduct_template(ligand, hit, anchor)
    geometry = select_formed_bond_geometry(
        adduct,
        nucleophile_idx,
        reactive_idx,
        preferred_carbon_length=anchor.bond_length,
    )
    assert geometry.source == "rdkit_distance_geometry_bounds_midpoint"
    assert geometry.lower_bound <= geometry.target <= geometry.upper_bound
    assert geometry.target != pytest.approx(carbon_length)


@pytest.mark.parametrize(
    ("residue", "smiles", "warhead_type", "expected_type", "expected_donor", "expected_acceptor"),
    [
        ("CYS", "C=CC(=O)N", "acrylamide", "S_P", False, False),
        ("SER", "CC(=O)F", "acyl_fluoride", "O_A", False, True),
        ("THR", "CP(=O)(O)O", "phosphonate", "O_A", False, True),
        ("TYR", "FS(=O)(=O)c1ccccc1", "sulfonyl_fluoride", "O_A", False, True),
        ("HIS", "C1CO1", "epoxide", "N_P", False, False),
        ("LYS", "CC(=O)F", "acyl_fluoride", "N_D", True, False),
        ("LYS", "N#Cc1ccccc1", "aryl_nitrile", "N_D", True, False),
        ("LYS", "CN=C=S", "isothiocyanate", "N_D", True, False),
        ("LYS", "C1CO1", "epoxide", "N_DA", True, True),
        ("LYS", "CC=O", "aldehyde", "N_DA", True, True),
    ],
)
def test_covalent_product_state_retypes_receptor_nucleophile(
    residue: str,
    smiles: str,
    warhead_type: str,
    expected_type: str,
    expected_donor: bool,
    expected_acceptor: bool,
) -> None:
    receptor, anchor = _minimal_reactive_receptor(residue)
    ligand = Chem.MolFromSmiles(smiles)
    hit = next(hit for hit in detect_warheads(ligand) if hit.warhead_type == warhead_type)
    adduct, _, _, reactive_idx = create_adduct_template(ligand, hit, anchor)
    receptor_nucleophile_idx = find_receptor_nucleophile_index(receptor.mol, anchor)
    original_donor = receptor.features["donor"].clone()
    original_acceptor = receptor.features["acceptor"].clone()

    product, change = _product_state_receptor_context(
        receptor,
        anchor,
        receptor_nucleophile_idx,
        adduct,
        reactive_idx,
    )

    assert product is not receptor
    assert product.source_fingerprint == receptor.source_fingerprint
    assert product.structure_fingerprint != receptor.structure_fingerprint
    assert product.atom_typing_version.endswith(COVALENT_RECEPTOR_TYPING_VERSION)
    assert product.features["xs_types"][receptor_nucleophile_idx] == expected_type
    assert bool(product.features["donor"][receptor_nucleophile_idx]) is expected_donor
    assert bool(product.features["acceptor"][receptor_nucleophile_idx]) is expected_acceptor
    assert change["after"] == {
        "xs_type": expected_type,
        "donor": expected_donor,
        "acceptor": expected_acceptor,
    }
    assert torch.equal(receptor.features["donor"], original_donor)
    assert torch.equal(receptor.features["acceptor"], original_acceptor)


def test_ser_product_state_removes_spurious_remote_acceptor_hbond() -> None:
    receptor, anchor = _minimal_reactive_receptor("SER")
    ligand = Chem.MolFromSmiles("COCCOC(=O)F")
    hit = next(hit for hit in detect_warheads(ligand) if hit.warhead_type == "acyl_fluoride")
    adduct, support_idx, nucleophile_idx, reactive_idx = create_adduct_template(ligand, hit, anchor)
    receptor_nucleophile_idx = find_receptor_nucleophile_index(receptor.mol, anchor)
    product, _ = _product_state_receptor_context(
        receptor,
        anchor,
        receptor_nucleophile_idx,
        adduct,
        reactive_idx,
    )
    distances = Chem.GetDistanceMatrix(adduct)
    remote_oxygen = max(
        (atom.GetIdx() for atom in adduct.GetAtoms() if atom.GetAtomicNum() == 8),
        key=lambda atom_idx: distances[reactive_idx, atom_idx],
    )
    exclusion = create_covalent_exclusion_mask(
        adduct,
        receptor.mol,
        pseudo_atom_indices={support_idx, nucleophile_idx},
        reactive_atom_idx=reactive_idx,
        receptor_nucleophile_idx=receptor_nucleophile_idx,
        device="cpu",
    )
    ligand_features = compute_atom_features(adduct, "cpu")
    reactant_matrices = prepare_interaction_matrices(ligand_features, receptor.features, VINA_CONFIG, "cpu")
    product_matrices = prepare_interaction_matrices(ligand_features, product.features, VINA_CONFIG, "cpu")

    assert not bool(exclusion[0, remote_oxygen, receptor_nucleophile_idx])
    assert reactant_matrices["hbond"][remote_oxygen, receptor_nucleophile_idx] == 1
    assert product_matrices["hbond"][remote_oxygen, receptor_nucleophile_idx] == 0


def test_lys_product_typing_is_warhead_specific_without_context_pollution() -> None:
    receptor, anchor = _minimal_reactive_receptor("LYS")
    receptor_nucleophile_idx = find_receptor_nucleophile_index(receptor.mol, anchor)

    def product_for(smiles: str, warhead_type: str) -> ReceptorContext:
        ligand = Chem.MolFromSmiles(smiles)
        hit = next(hit for hit in detect_warheads(ligand) if hit.warhead_type == warhead_type)
        adduct, _, _, reactive_idx = create_adduct_template(ligand, hit, anchor)
        product, _ = _product_state_receptor_context(
            receptor,
            anchor,
            receptor_nucleophile_idx,
            adduct,
            reactive_idx,
        )
        return product

    acylated = product_for("CC(=O)F", "acyl_fluoride")
    ring_opened = product_for("C1CO1", "epoxide")
    acylated_again = product_for("CC(=O)F", "acyl_fluoride")

    assert receptor.features["xs_types"][receptor_nucleophile_idx] == "N_D"
    assert acylated.features["xs_types"][receptor_nucleophile_idx] == "N_D"
    assert ring_opened.features["xs_types"][receptor_nucleophile_idx] == "N_DA"
    assert acylated_again.structure_fingerprint == acylated.structure_fingerprint


def test_ambiguous_automatic_residue_is_rejected(ambiguous_cys_pdb: Path) -> None:
    protein = Chem.MolFromPDBFile(str(ambiguous_cys_pdb), sanitize=False, removeHs=True)
    with pytest.raises(ValueError, match="ambiguous"):
        select_reactive_anchor(protein)


def test_covalent_exclusion_matches_formed_bond_graph_distances() -> None:
    ligand = Chem.MolFromSmiles("CCC.CS")
    receptor = Chem.MolFromSmiles("NCCC")
    mask = create_covalent_exclusion_mask(
        ligand,
        receptor,
        pseudo_atom_indices={3, 4},
        reactive_atom_idx=2,
        receptor_nucleophile_idx=0,
        device="cpu",
    )[0]
    assert mask[3].all() and mask[4].all()
    assert mask[2, :3].all() and not mask[2, 3]
    assert mask[1, :2].all() and not mask[1, 2]
    assert mask[0, 0] and not mask[0, 1]


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
    assert poses[0].GetProp("AnchorDock_Version") == "0.5.0"
    assert poses[0].GetProp("AnchorDock_Score_Semantics") == "adduct_conditioned_pose_ranking"
    assert poses[0].HasProp("AnchorDock_Scorer_Fingerprint")
    assert poses[0].HasProp("AnchorDock_Receptor_Structure_Fingerprint")
    assert poses[0].HasProp("AnchorDock_Receptor_Reactant_Structure_Fingerprint")
    assert poses[0].HasProp("AnchorDock_Receptor_Source_Fingerprint")
    assert poses[0].GetProp("AnchorDock_Covalent_Receptor_Typing_State") == "product"
    assert poses[0].GetProp("AnchorDock_Covalent_Receptor_Typing_Version") == COVALENT_RECEPTOR_TYPING_VERSION
    assert poses[0].GetProp("AnchorDock_Receptor_Atom_Typing_Version").endswith(COVALENT_RECEPTOR_TYPING_VERSION)
    assert poses[0].GetProp("AnchorDock_Receptor_Reactant_Atom_Typing_Version") == "inferred-xs-v2"
    assert poses[0].HasProp("AnchorDock_Covalent_Receptor_Typing_Changes")
    assert poses[0].HasProp("AnchorDock_Intramolecular_Reference")
    assert poses[0].HasProp("AnchorDock_Canonical_Ligand_Atoms")
    assert not poses[0].HasProp("AnchorDock_Original_Ligand_Atoms")
    assert poses[0].HasProp("AnchorDock_Pseudo_Support_Atom_Index")
    assert poses[0].HasProp("AnchorDock_Pseudo_Nucleophile_Atom_Index")
    assert not any(name.startswith("CovVina_") for name in properties)
    assert torch.isfinite(torch.tensor(result["best_score"]))
    support_idx = int(poses[0].GetProp("AnchorDock_Pseudo_Support_Atom_Index"))
    nucleophile_idx = int(poses[0].GetProp("AnchorDock_Pseudo_Nucleophile_Atom_Index"))
    reactive_idx = int(poses[0].GetProp("AnchorDock_Adduct_Reactive_Atom_Index"))
    coordinates = poses[0].GetConformer().GetPositions()
    support_vector = coordinates[support_idx] - coordinates[nucleophile_idx]
    reactive_vector = coordinates[reactive_idx] - coordinates[nucleophile_idx]
    cosine = float(
        np.dot(support_vector, reactive_vector) / (np.linalg.norm(support_vector) * np.linalg.norm(reactive_vector))
    )
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    assert angle < 170.0
    lower, upper = result["support_reactive_distance_bounds"]
    one_three = float(np.linalg.norm(coordinates[support_idx] - coordinates[reactive_idx]))
    assert lower - 0.05 <= one_three <= upper + 0.05
    assert result["canonical_ligand_atoms"] == Chem.MolFromSmiles(result["canonical_smiles"]).GetNumAtoms()
    assert result["receptor_structure_scope"] == "extracted_pocket"
    assert result["receptor_structure_fingerprint"].startswith("sha256:")
    assert result["receptor_reactant_structure_fingerprint"].startswith("sha256:")
    assert result["receptor_structure_fingerprint"] != result["receptor_reactant_structure_fingerprint"]
    assert result["receptor_source_fingerprint"].startswith("sha256:")
    assert result["covalent_receptor_typing_state"] == "product"
    assert result["covalent_receptor_typing_version"] == COVALENT_RECEPTOR_TYPING_VERSION
    assert result["receptor_atom_typing_version"].endswith(COVALENT_RECEPTOR_TYPING_VERSION)
    assert result["receptor_reactant_atom_typing_version"] == "inferred-xs-v2"
    assert result["covalent_receptor_typing_changes"][0]["atom_name"] == "SG"


def test_covalent_noncarbon_electrophile_pipeline_uses_pair_specific_target(
    cys_pdb: Path,
    tmp_path: Path,
) -> None:
    result = dock_covalent(
        cys_pdb,
        "FS(=O)(=O)c1ccccc1",
        "CYS145:A",
        tmp_path / "sulfonyl",
        num_confs=4,
        rotation_scan_step=0,
        optimize=False,
        top_k=1,
        device="cpu",
        verbose=False,
    )
    lower, upper = result["covalent_bond_distance_bounds"]
    assert result["covalent_bond_target_source"] == "rdkit_distance_geometry_bounds_midpoint"
    assert lower <= result["covalent_bond_length"] <= upper
    assert result["electrophile_atomic_number"] == 16
    assert result["nucleophile_atomic_number"] == 16


def test_covalent_receptor_context_is_cached(cys_pdb: Path) -> None:
    clear_covalent_context_cache()
    first = _prepare_covalent_receptor(cys_pdb, "CYS145:A", 12.0, True, "cpu")
    second = _prepare_covalent_receptor(cys_pdb, "CYS145:A", 12.0, True, "cpu")
    assert first is second


@pytest.mark.parametrize("residue", ["CYS", "SER", "THR", "TYR", "LYS", "HIS"])
def test_native_connectivity_accepted_for_all_six_residues(residue: str) -> None:
    _, anchor = _minimal_reactive_receptor(residue)
    assert anchor.residue_name == residue
    assert anchor.atom_name == REACTIVE_RESIDUES[residue].atom_name


def test_disulfide_and_extra_heavy_neighbor_rejection() -> None:
    disulfide_pdb = (
        "ATOM      1  CB  CYS A   1       0.000   0.000   0.000  1.00 20.00           C  \n"
        "ATOM      2  SG  CYS A   1       1.500   0.000   0.000  1.00 20.00           S  \n"
        "ATOM      3  SG  CYS A   2       2.800   0.000   0.000  1.00 20.00           S  \n"
        "ATOM      4  CB  CYS A   2       4.300   0.000   0.000  1.00 20.00           C  \n"
        "TER\nEND\n"
    )
    mol_disulfide = Chem.MolFromPDBBlock(disulfide_pdb, sanitize=False, removeHs=True)
    assert mol_disulfide is not None
    with pytest.raises(ValueError, match="no supported reactive residue found"):
        select_reactive_anchor(mol_disulfide, "CYS1:A")

    extra_lys_pdb = (
        "ATOM      1  CE  LYS A   1       0.000   0.000   0.000  1.00 20.00           C  \n"
        "ATOM      2  NZ  LYS A   1       1.500   0.000   0.000  1.00 20.00           N  \n"
        "ATOM      3  C1  LYS A   1       2.000   1.200   0.000  1.00 20.00           C  \n"
        "TER\nEND\n"
    )
    mol_extra_lys = Chem.MolFromPDBBlock(extra_lys_pdb, sanitize=False, removeHs=True)
    assert mol_extra_lys is not None
    with pytest.raises(ValueError, match="no supported reactive residue found"):
        select_reactive_anchor(mol_extra_lys, "LYS1:A")


def test_incomplete_his_rejection() -> None:
    his_no_cd2 = (
        "ATOM      1  CE1 HIS A   1       0.000   0.000   0.000  1.00 20.00           C  \n"
        "ATOM      2  NE2 HIS A   1       1.500   0.000   0.000  1.00 20.00           N  \n"
        "TER\nEND\n"
    )
    mol_no_cd2 = Chem.MolFromPDBBlock(his_no_cd2, sanitize=False, removeHs=True)
    assert mol_no_cd2 is not None
    with pytest.raises(ValueError, match="no supported reactive residue found"):
        select_reactive_anchor(mol_no_cd2, "HIS1:A")

    his_unbonded_cd2 = (
        "ATOM      1  CE1 HIS A   1       0.000   0.000   0.000  1.00 20.00           C  \n"
        "ATOM      2  NE2 HIS A   1       1.500   0.000   0.000  1.00 20.00           N  \n"
        "ATOM      3  CD2 HIS A   1      10.000   0.000   0.000  1.00 20.00           C  \n"
        "TER\nEND\n"
    )
    mol_unbonded = Chem.MolFromPDBBlock(his_unbonded_cd2, sanitize=False, removeHs=True)
    assert mol_unbonded is not None
    with pytest.raises(ValueError, match="no supported reactive residue found"):
        select_reactive_anchor(mol_unbonded, "HIS1:A")


def test_dock_covalent_early_rotation_validation(tmp_path: Path) -> None:
    non_existent = tmp_path / "non_existent_receptor.pdb"
    with pytest.raises(ValueError, match="rotation_scan_step must be in 1..360 or 0 to disable"):
        dock_covalent_canonical(non_existent, "C=CC(=O)N", "CYS1:A", rotation_scan_step=-1)

    with pytest.raises(ValueError, match="rotation_scan_step must be in 1..360 or 0 to disable"):
        dock_covalent_canonical(non_existent, "C=CC(=O)N", "CYS1:A", rotation_scan_step=361)

    with pytest.raises(ValueError, match="rotation_top_k must be positive when rotation scanning is enabled"):
        dock_covalent_canonical(non_existent, "C=CC(=O)N", "CYS1:A", rotation_scan_step=30, rotation_top_k=0)
