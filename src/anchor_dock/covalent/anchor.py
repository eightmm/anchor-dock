"""Warhead detection and deterministic protein-side covalent anchors."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

# Ordered from more specific to more generic patterns. The mapped atom is the
# ligand electrophile that receives the protein nucleophile bond.
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
    ("[C;r3:1]1[O;r3][C;r3]1", 1, "epoxide"),
    ("[C;r3:1]1[N;r3][C;r3]1", 1, "aziridine"),
    ("[C;r3:1]1[S;r3][C;r3]1", 1, "thiirane"),
    ("N#[C:1]c", 1, "aryl_nitrile"),
    ("N#[C:1]C([#6])", 1, "alkyl_nitrile"),
    ("[CH:1]#CC(=O)[N,n]", 1, "propiolamide"),
    ("[CH2:1]=[C](C#N)C(=O)[N,n]", 1, "cyanoacrylamide"),
    ("N#C[CH:1]=[CH]C(=O)[N,n]", 1, "cyanoacrylamide"),
    ("[S:1]S[#6]", 1, "disulfide"),
    ("F[S:1](=O)(=O)[c,C]", 1, "sulfonyl_fluoride"),
    ("O=[C:1]C(=O)[N,n]", 1, "alpha_ketoamide"),
    ("[CH1:1]=O", 1, "aldehyde"),
    ("[C:1](=N)=S", 1, "isothiocyanate"),
    ("[C:1](=O)ON1C(=O)CCC1=O", 1, "nhs_ester"),
    ("[C:1](=O)OC(F)(F)F", 1, "tfe_ester"),
    ("[C:1](=O)F", 1, "acyl_fluoride"),
    ("[P:1](=O)([OH])[OH]", 1, "phosphonate"),
)

_RING_OPENING_WARHEADS = {"epoxide", "aziridine", "thiirane"}


@dataclass(frozen=True)
class ResidueConfig:
    atom_name: str
    support_atom_name: str
    bond_length: float
    atomic_number: int
    expected_neighbor_names: tuple[str, ...] = ()


REACTIVE_RESIDUES: dict[str, ResidueConfig] = {
    "CYS": ResidueConfig("SG", "CB", 1.82, 16, ("CB",)),
    "SER": ResidueConfig("OG", "CB", 1.43, 8, ("CB",)),
    "THR": ResidueConfig("OG1", "CB", 1.43, 8, ("CB",)),
    "TYR": ResidueConfig("OH", "CZ", 1.43, 8, ("CZ",)),
    "LYS": ResidueConfig("NZ", "CE", 1.47, 7, ("CE",)),
    "HIS": ResidueConfig("NE2", "CE1", 1.47, 7, ("CE1", "CD2")),
}

GOOD_COMPATIBILITY: dict[str, set[str]] = {
    "acrylamide": {"CYS"},
    "acrylic_acid": {"CYS"},
    "acrylate": {"CYS"},
    "enone": {"CYS"},
    "vinyl_sulfonamide": {"CYS"},
    "vinyl_sulfone": {"CYS"},
    "maleimide": {"CYS"},
    "cyanoacrylamide": {"CYS"},
    "chloroacetamide": {"CYS"},
    "bromoacetamide": {"CYS"},
    "iodoacetamide": {"CYS"},
    "fluoroacetamide": {"CYS"},
    "chlorofluoroacetamide": {"CYS"},
    "epoxide": {"CYS", "LYS", "HIS"},
    "aziridine": {"CYS", "LYS", "HIS"},
    "thiirane": {"CYS"},
    "aryl_nitrile": {"CYS", "LYS"},
    "alkyl_nitrile": {"CYS", "LYS"},
    "propiolamide": {"CYS"},
    "phosphonate": {"SER", "THR"},
    "sulfonyl_fluoride": {"CYS", "SER", "THR", "TYR", "LYS", "HIS"},
    "acyl_fluoride": {"CYS", "SER", "THR", "TYR", "LYS", "HIS"},
    "aldehyde": {"CYS", "SER", "LYS"},
    "alpha_ketoamide": {"CYS", "SER", "LYS"},
    "isothiocyanate": {"CYS", "LYS", "HIS"},
    "disulfide": {"CYS"},
    "nhs_ester": {"LYS", "SER", "CYS"},
    "tfe_ester": {"LYS", "SER", "CYS"},
}

NO_COMPATIBILITY: dict[str, set[str]] = {
    "acrylamide": {"SER", "THR", "LYS"},
    "vinyl_sulfonamide": {"SER"},
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
    insertion_code: str
    chain_id: str
    atom_name: str
    support_atom_name: str
    coord: np.ndarray
    support_coord: np.ndarray
    bond_vector: np.ndarray
    bond_length: float
    atomic_number: int

    @property
    def residue_id(self) -> str:
        insertion = self.insertion_code or ""
        chain = f":{self.chain_id}" if self.chain_id else ""
        return f"{self.residue_name}{self.residue_num}{insertion}{chain}"


def detect_warheads(mol: Chem.Mol) -> list[WarheadHit]:
    """Detect supported electrophiles, keeping the most specific hit per atom."""
    candidates: list[WarheadHit] = []
    for smarts, map_num, name in WARHEAD_REGISTRY:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        mapped_idx = next((atom.GetIdx() for atom in pattern.GetAtoms() if atom.GetAtomMapNum() == map_num), None)
        if mapped_idx is None:
            continue
        for match in mol.GetSubstructMatches(pattern, uniquify=False):
            candidates.append(WarheadHit(name, int(match[mapped_idx]), tuple(int(value) for value in match)))
    candidates.sort(
        key=lambda hit: (
            -len(hit.matched_atoms),
            mol.GetAtomWithIdx(hit.reactive_atom_idx).GetDegree() if hit.warhead_type in _RING_OPENING_WARHEADS else 0,
            hit.reactive_atom_idx,
        )
    )
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
    *,
    strict: bool = False,
) -> tuple[bool, str]:
    residue_name = residue_name.upper()
    if residue_name in NO_COMPATIBILITY.get(warhead_type, set()):
        return False, f"{warhead_type} is not chemically compatible with {residue_name}"
    if residue_name in GOOD_COMPATIBILITY.get(warhead_type, set()):
        return True, f"{warhead_type}/{residue_name} is a supported combination"
    message = f"{warhead_type}/{residue_name} has limited precedent"
    return (not strict), (message if not strict else message + "; strict mode rejects it")


def _parse_residue_spec(spec: str | None) -> tuple[str | None, int | None, str | None, str | None]:
    if spec is None:
        return None, None, None, None
    match = re.fullmatch(r"([A-Za-z]+)(-?\d+)([A-Za-z]?)(?::([^:]+))?", spec.strip())
    if not match:
        raise ValueError(f"invalid residue specifier {spec!r}; expected CYS145:A or CYS145A:A")
    return match.group(1).upper(), int(match.group(2)), match.group(3) or None, match.group(4) or None


def find_reactive_residues(protein_mol: Chem.Mol, residue_spec: str | None = None) -> list[AnchorPoint]:
    """Locate all supported nucleophiles matching an optional residue specifier."""
    if protein_mol.GetNumConformers() == 0:
        raise ValueError("protein molecule has no conformer")
    target_name, target_number, target_insertion, target_chain = _parse_residue_spec(residue_spec)
    residues: dict[tuple[str, int, str, str], dict[str, int]] = {}
    for atom in protein_mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None or info.GetIsHeteroAtom():
            continue
        key = (
            info.GetResidueName().strip().upper(),
            info.GetResidueNumber(),
            info.GetInsertionCode().strip(),
            info.GetChainId().strip(),
        )
        atom_name = info.GetName().strip().upper()
        current = residues.setdefault(key, {}).get(atom_name)
        if current is None:
            residues[key][atom_name] = atom.GetIdx()
        else:
            current_info = protein_mol.GetAtomWithIdx(current).GetPDBResidueInfo()
            if current_info is not None and info.GetOccupancy() > current_info.GetOccupancy():
                residues[key][atom_name] = atom.GetIdx()

    positions = protein_mol.GetConformer().GetPositions()
    anchors: list[AnchorPoint] = []
    for (name, number, insertion, chain), atoms in sorted(residues.items()):
        if target_name is not None and name != target_name:
            continue
        if target_number is not None and number != target_number:
            continue
        if target_insertion is not None and insertion != target_insertion:
            continue
        if target_chain is not None and chain != target_chain:
            continue
        config = REACTIVE_RESIDUES.get(name)
        if config is None or config.atom_name not in atoms or config.support_atom_name not in atoms:
            continue
        if any(nb_name not in atoms for nb_name in config.expected_neighbor_names):
            continue

        nucleophile_idx = atoms[config.atom_name]
        nucleophile_atom = protein_mol.GetAtomWithIdx(nucleophile_idx)
        if nucleophile_atom.GetAtomicNum() != config.atomic_number:
            continue

        expected_neighbor_indices = {atoms[nb_name] for nb_name in config.expected_neighbor_names}
        actual_heavy_neighbor_indices = {
            nb.GetIdx() for nb in nucleophile_atom.GetNeighbors() if nb.GetAtomicNum() != 1
        }
        if actual_heavy_neighbor_indices != expected_neighbor_indices:
            continue

        coord = positions[nucleophile_idx].copy()
        support_coord = positions[atoms[config.support_atom_name]].copy()
        vector = coord - support_coord
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            continue
        anchors.append(
            AnchorPoint(
                name,
                number,
                insertion,
                chain,
                config.atom_name,
                config.support_atom_name,
                coord,
                support_coord,
                vector / norm,
                config.bond_length,
                config.atomic_number,
            )
        )
    return anchors


def select_reactive_anchor(protein_mol: Chem.Mol, residue_spec: str | None = None) -> AnchorPoint:
    """Select one anchor; ambiguous automatic detection is rejected."""
    anchors = find_reactive_residues(protein_mol, residue_spec)
    if not anchors:
        raise ValueError(f"no supported reactive residue found for {residue_spec or 'automatic selection'}")
    if residue_spec is None and len(anchors) != 1:
        candidates = ", ".join(anchor.residue_id for anchor in anchors[:20])
        suffix = " ..." if len(anchors) > 20 else ""
        raise ValueError(
            f"automatic reactive-residue selection is ambiguous ({len(anchors)} candidates: {candidates}{suffix}); "
            "provide reactive_residue explicitly"
        )
    if len(anchors) > 1:
        candidates = ", ".join(anchor.residue_id for anchor in anchors)
        raise ValueError(f"residue specifier {residue_spec!r} is ambiguous: {candidates}")
    return anchors[0]


def create_covalent_coordmap(
    support_atom_idx: int,
    nucleophile_atom_idx: int,
    anchor: AnchorPoint,
) -> dict[int, Point3D]:
    """Return the two exact protein-side coordinates for adduct embedding.

    The ligand electrophile is deliberately not constrained here. Constraining
    all three atoms to the protein bond axis would force an unphysical 180°
    support-nucleophile-electrophile angle. ETKDG instead chooses a
    topology-compatible approach angle; the new bond length is normalized
    afterwards without deforming the ligand.
    """
    return {
        support_atom_idx: Point3D(*map(float, anchor.support_coord)),
        nucleophile_atom_idx: Point3D(*map(float, anchor.coord)),
    }
