"""Warhead detection and protein-side covalent anchors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

# Ordered from more characteristic to more generic patterns.
WARHEAD_REGISTRY: tuple[tuple[str, int, str], ...] = (
    ("[CH2:1]=[CH]C(=O)[N,n]", 1, "acrylamide"),
    ("[CH2:1]=[CH]C(=O)[OH]", 1, "acrylic_acid"),
    ("[CH2:1]=[CH]C(=O)O[#6]", 1, "acrylate"),
    ("[CH2:1]=[CH]C(=O)[#6]", 1, "enone"),
    ("[CH2:1]=[CH]S(=O)(=O)[N,n]", 1, "vinyl_sulfonamide"),
    ("[CH2:1]=[CH]S(=O)(=O)[#6]", 1, "vinyl_sulfone"),
    ("O=C1[CH:1]=[CH]C(=O)[NH,N]1", 1, "maleimide"),
    ("Cl[CH2:1]C(=O)[N,n]", 1, "chloroacetamide"),
    ("Br[CH2:1]C(=O)[N,n]", 1, "bromoacetamide"),
    ("I[CH2:1]C(=O)[N,n]", 1, "iodoacetamide"),
    ("F[CH2:1]C(=O)[N,n]", 1, "fluoroacetamide"),
    ("Cl[C:1](F)C(=O)[N,n]", 1, "chlorofluoroacetamide"),
    ("[CH:1]1OC1", 1, "epoxide"),
    ("[CH:1]1NC1", 1, "aziridine"),
    ("[CH:1]1SC1", 1, "thiirane"),
    ("N#[C:1]c", 1, "aryl_nitrile"),
    ("N#[C:1]C([#6])", 1, "alkyl_nitrile"),
    ("[CH:1]#CC(=O)[N,n]", 1, "propiolamide"),
    ("[C:1]#CC(=O)[N,n]", 1, "propargylamide"),
    ("N#CC=[C:1]C(=O)[N,n]", 1, "cyanoacrylamide"),
    ("[S:1]S[#6]", 1, "disulfide"),
    ("F[S:1](=O)(=O)[c,C]", 1, "sulfonyl_fluoride"),
    ("O=[C:1]C(=O)[N,n]", 1, "alpha_ketoamide"),
    ("[CH1:1]=O", 1, "aldehyde"),
    ("[C:1](=N)=S", 1, "isothiocyanate"),
    ("[C:1](=O)On1c(=O)cccc1", 1, "nhs_ester"),
    ("[C:1](=O)OC(F)(F)F", 1, "tfe_ester"),
    ("[C:1](=O)F", 1, "acyl_fluoride"),
    ("[B:1]([OX2])[OX2]", 1, "boronic_acid"),
    ("[P:1](=O)([OH])[OH]", 1, "phosphonate"),
)


@dataclass(frozen=True)
class ResidueConfig:
    atom_name: str
    bond_length: float


REACTIVE_RESIDUES: dict[str, ResidueConfig] = {
    "CYS": ResidueConfig("SG", 1.82),
    "SER": ResidueConfig("OG", 1.43),
    "LYS": ResidueConfig("NZ", 1.47),
    "THR": ResidueConfig("OG1", 1.43),
    "TYR": ResidueConfig("OH", 1.43),
    "HIS": ResidueConfig("NE2", 1.47),
}

GOOD_COMPATIBILITY: dict[str, set[str]] = {
    "acrylamide": {"CYS"}, "acrylic_acid": {"CYS"}, "acrylate": {"CYS"},
    "enone": {"CYS"}, "vinyl_sulfonamide": {"CYS"}, "vinyl_sulfone": {"CYS"},
    "maleimide": {"CYS"}, "cyanoacrylamide": {"CYS"},
    "chloroacetamide": {"CYS"}, "bromoacetamide": {"CYS"}, "iodoacetamide": {"CYS"},
    "epoxide": {"CYS", "LYS", "HIS"}, "aziridine": {"CYS", "LYS", "HIS"},
    "thiirane": {"CYS"}, "aryl_nitrile": {"CYS", "LYS"}, "alkyl_nitrile": {"CYS", "LYS"},
    "propiolamide": {"CYS"}, "propargylamide": {"CYS"},
    "boronic_acid": {"SER", "THR", "TYR"}, "phosphonate": {"SER", "THR"},
    "sulfonyl_fluoride": {"CYS", "SER", "THR", "TYR", "LYS", "HIS"},
    "acyl_fluoride": {"CYS", "SER", "THR", "TYR", "LYS", "HIS"},
    "aldehyde": {"CYS", "SER", "LYS"}, "alpha_ketoamide": {"CYS", "SER", "LYS"},
    "isothiocyanate": {"CYS", "LYS", "HIS"}, "disulfide": {"CYS"},
    "nhs_ester": {"LYS", "SER", "CYS"}, "tfe_ester": {"LYS", "SER", "CYS"},
}

NO_COMPATIBILITY: dict[str, set[str]] = {
    "acrylamide": {"SER", "THR", "LYS"},
    "vinyl_sulfonamide": {"SER"},
    "boronic_acid": {"CYS", "LYS"},
    "phosphonate": {"CYS"},
}


@dataclass(frozen=True)
class WarheadHit:
    warhead_type: str
    reactive_atom_idx: int
    matched_atoms: tuple[int, ...]


@dataclass(frozen=True)
class AnchorPoint:
    residue_name: str
    residue_num: int
    chain_id: str
    atom_name: str
    coord: np.ndarray
    bond_vector: np.ndarray
    bond_length: float
    cb_coord: np.ndarray | None = None


def detect_warheads(mol: Chem.Mol) -> list[WarheadHit]:
    """Detect all supported electrophiles, preferring the most specific hit per atom."""
    candidates: list[WarheadHit] = []
    for smarts, map_num, name in WARHEAD_REGISTRY:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        mapped_idx = next((a.GetIdx() for a in pattern.GetAtoms() if a.GetAtomMapNum() == map_num), None)
        if mapped_idx is None:
            continue
        for match in mol.GetSubstructMatches(pattern):
            candidates.append(WarheadHit(name, match[mapped_idx], tuple(match)))
    candidates.sort(key=lambda hit: len(hit.matched_atoms), reverse=True)
    result: list[WarheadHit] = []
    seen: set[int] = set()
    for hit in candidates:
        if hit.reactive_atom_idx not in seen:
            result.append(hit)
            seen.add(hit.reactive_atom_idx)
    return result


def check_warhead_residue_compatibility(
    warhead_type: str,
    residue_name: str,
    strict: bool = False,
) -> tuple[bool, str]:
    residue_name = residue_name.upper()
    if residue_name in NO_COMPATIBILITY.get(warhead_type, set()):
        return False, f"{warhead_type} is not chemically compatible with {residue_name}"
    if residue_name in GOOD_COMPATIBILITY.get(warhead_type, set()):
        return True, f"{warhead_type}/{residue_name} is a supported combination"
    message = f"{warhead_type}/{residue_name} has limited precedent"
    return (not strict), (message if not strict else message + " and strict mode rejects it")


def _parse_residue_spec(spec: str | None) -> tuple[str | None, int | None, str | None]:
    if spec is None:
        return None, None, None
    head, *chain = spec.split(":", maxsplit=1)
    letters = "".join(ch for ch in head if not ch.isdigit() and ch != "-").upper()
    number = "".join(ch for ch in head if ch.isdigit() or ch == "-")
    return letters or None, int(number) if number not in {"", "-"} else None, chain[0] if chain else None


def find_reactive_residues(pocket_mol: Chem.Mol, residue_spec: str | None = None) -> list[AnchorPoint]:
    """Locate supported residue nucleophiles in a PDB-derived RDKit molecule."""
    if pocket_mol.GetNumConformers() == 0:
        raise ValueError("Protein molecule has no conformer")
    target_name, target_number, target_chain = _parse_residue_spec(residue_spec)
    residues: dict[tuple[str, int, str], dict[str, int]] = {}
    for atom in pocket_mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None or info.GetIsHeteroAtom():
            continue
        key = (info.GetResidueName().strip(), info.GetResidueNumber(), info.GetChainId().strip())
        residues.setdefault(key, {})[info.GetName().strip()] = atom.GetIdx()

    positions = pocket_mol.GetConformer().GetPositions()
    anchors: list[AnchorPoint] = []
    for (name, number, chain), atom_map in residues.items():
        if target_name is not None and name != target_name:
            continue
        if target_number is not None and number != target_number:
            continue
        if target_chain is not None and chain != target_chain:
            continue
        config = REACTIVE_RESIDUES.get(name)
        if config is None or config.atom_name not in atom_map:
            continue
        coord = positions[atom_map[config.atom_name]].copy()
        cb_coord = None
        direction = np.array([0.0, 0.0, 1.0], dtype=float)
        if "CB" in atom_map:
            cb_coord = positions[atom_map["CB"]].copy()
            vector = coord - cb_coord
            if np.linalg.norm(vector) > 1e-8:
                direction = vector / np.linalg.norm(vector)
        anchors.append(AnchorPoint(name, number, chain, config.atom_name, coord, direction, config.bond_length, cb_coord))
    return anchors


def create_covalent_coordmap(
    cb_atom_idx: int | None,
    nucleophile_atom_idx: int,
    anchor: AnchorPoint,
) -> dict[int, Point3D]:
    coord_map: dict[int, Point3D] = {}
    if cb_atom_idx is not None and anchor.cb_coord is not None:
        coord_map[cb_atom_idx] = Point3D(*map(float, anchor.cb_coord))
    coord_map[nucleophile_atom_idx] = Point3D(*map(float, anchor.coord))
    return coord_map
