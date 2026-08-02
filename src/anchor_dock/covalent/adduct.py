"""Covalent adduct topology construction and scoring exclusions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Geometry import Point3D

from .anchor import AnchorPoint, WarheadHit

LEAVING_GROUP_MATCH_POSITIONS: dict[str, tuple[int, ...]] = {
    "chloroacetamide": (0,),
    "bromoacetamide": (0,),
    "iodoacetamide": (0,),
    "fluoroacetamide": (0,),
    "chlorofluoroacetamide": (0,),
    "acyl_fluoride": (2,),
    "sulfonyl_fluoride": (0,),
}

MICHAEL_WARHEADS = {
    "acrylamide",
    "acrylic_acid",
    "acrylate",
    "enone",
    "vinyl_sulfonamide",
    "vinyl_sulfone",
    "maleimide",
    "cyanoacrylamide",
}
RING_OPENING_WARHEADS = {"epoxide", "aziridine", "thiirane"}
TRIPLE_BOND_WARHEADS = {"aryl_nitrile", "alkyl_nitrile", "propiolamide"}
CARBONYL_ADDITION_WARHEADS = {"aldehyde", "alpha_ketoamide"}


@dataclass(frozen=True)
class FormedBondGeometry:
    target: float
    lower_bound: float
    upper_bound: float
    source: str
    nucleophile_atomic_number: int
    electrophile_atomic_number: int


def _replace_bond(editable: Chem.RWMol, atom_a: int, atom_b: int, bond_type: Chem.BondType | None) -> None:
    editable.RemoveBond(atom_a, atom_b)
    if bond_type is not None:
        editable.AddBond(atom_a, atom_b, bond_type)


def _branch_indices(editable: Chem.RWMol, root_idx: int, start_idx: int) -> set[int]:
    """Return the atom branch reached from ``start_idx`` without crossing ``root_idx``."""
    branch: set[int] = set()
    stack = [start_idx]
    while stack:
        atom_idx = stack.pop()
        if atom_idx == root_idx or atom_idx in branch:
            continue
        branch.add(atom_idx)
        stack.extend(
            neighbor.GetIdx()
            for neighbor in editable.GetAtomWithIdx(atom_idx).GetNeighbors()
            if neighbor.GetIdx() != root_idx and neighbor.GetIdx() not in branch
        )
    return branch


def _find_single_bond_neighbor(
    editable: Chem.RWMol,
    reactive_idx: int,
    atomic_number: int,
) -> int | None:
    for bond in editable.GetAtomWithIdx(reactive_idx).GetBonds():
        neighbor_idx = bond.GetOtherAtomIdx(reactive_idx)
        if (
            bond.GetBondType() == Chem.BondType.SINGLE
            and editable.GetAtomWithIdx(neighbor_idx).GetAtomicNum() == atomic_number
        ):
            return neighbor_idx
    return None


def _reduce_first_bond(
    editable: Chem.RWMol,
    reactive_idx: int,
    source_type: Chem.BondType,
    target_type: Chem.BondType,
    *,
    allowed_atomic_numbers: set[int] | None = None,
) -> bool:
    atom = editable.GetAtomWithIdx(reactive_idx)
    for bond in list(atom.GetBonds()):
        other = bond.GetOtherAtomIdx(reactive_idx)
        if bond.GetBondType() != source_type:
            continue
        if allowed_atomic_numbers and editable.GetAtomWithIdx(other).GetAtomicNum() not in allowed_atomic_numbers:
            continue
        _replace_bond(editable, reactive_idx, other, target_type)
        return True
    return False


def create_adduct_template(
    ligand_mol: Chem.Mol,
    warhead: WarheadHit,
    anchor: AnchorPoint,
) -> tuple[Chem.Mol, int, int, int]:
    """Apply a reaction-class topology transform and attach protein anchor atoms."""
    editable = Chem.RWMol(Chem.RemoveHs(ligand_mol))
    positions = LEAVING_GROUP_MATCH_POSITIONS.get(warhead.warhead_type, ())
    matched = list(warhead.matched_atoms)
    remove_indices = [matched[pos] for pos in positions if pos < len(matched)]
    original_reactive = warhead.reactive_atom_idx

    if warhead.warhead_type in MICHAEL_WARHEADS:
        if not _reduce_first_bond(editable, original_reactive, Chem.BondType.DOUBLE, Chem.BondType.SINGLE):
            raise ValueError(f"Could not identify Michael double bond for {warhead.warhead_type}")
    elif warhead.warhead_type in RING_OPENING_WARHEADS:
        atom = editable.GetAtomWithIdx(original_reactive)
        broken = False
        for bond in list(atom.GetBonds()):
            other = bond.GetOtherAtomIdx(original_reactive)
            if editable.GetAtomWithIdx(other).GetAtomicNum() not in {7, 8, 16}:
                continue
            if any(
                original_reactive in ring and other in ring and len(ring) == 3
                for ring in editable.GetRingInfo().AtomRings()
            ):
                _replace_bond(editable, original_reactive, other, None)
                broken = True
                break
        if not broken:
            raise ValueError(f"Could not identify three-membered ring bond for {warhead.warhead_type}")
    elif warhead.warhead_type in TRIPLE_BOND_WARHEADS:
        if not _reduce_first_bond(editable, original_reactive, Chem.BondType.TRIPLE, Chem.BondType.DOUBLE):
            raise ValueError(f"Could not identify reactive triple bond for {warhead.warhead_type}")
    elif warhead.warhead_type in CARBONYL_ADDITION_WARHEADS:
        if not _reduce_first_bond(
            editable,
            original_reactive,
            Chem.BondType.DOUBLE,
            Chem.BondType.SINGLE,
            allowed_atomic_numbers={8},
        ):
            raise ValueError(f"Could not identify carbonyl for {warhead.warhead_type}")
    elif warhead.warhead_type == "isothiocyanate":
        if not _reduce_first_bond(
            editable,
            original_reactive,
            Chem.BondType.DOUBLE,
            Chem.BondType.SINGLE,
            allowed_atomic_numbers={7},
        ):
            raise ValueError("Could not reduce isothiocyanate C=N bond")
    elif warhead.warhead_type == "disulfide":
        distal_sulfur = _find_single_bond_neighbor(editable, original_reactive, 16)
        if distal_sulfur is None:
            raise ValueError("Could not identify disulfide leaving branch")
        remove_indices.extend(_branch_indices(editable, original_reactive, distal_sulfur))
    elif warhead.warhead_type == "phosphonate":
        leaving_oxygen = _find_single_bond_neighbor(editable, original_reactive, 8)
        if leaving_oxygen is None:
            raise ValueError("Could not identify phosphonate hydroxyl leaving group")
        remove_indices.extend(_branch_indices(editable, original_reactive, leaving_oxygen))
    elif warhead.warhead_type in {"nhs_ester", "tfe_ester"}:
        leaving_oxygen = _find_single_bond_neighbor(editable, original_reactive, 8)
        if leaving_oxygen is None:
            raise ValueError(f"Could not identify ester leaving group for {warhead.warhead_type}")
        remove_indices.extend(_branch_indices(editable, original_reactive, leaving_oxygen))
    elif not positions:
        raise ValueError(f"Unsupported adduct transform for {warhead.warhead_type}")

    remove_indices = sorted(set(remove_indices), reverse=True)
    if original_reactive in remove_indices:
        raise ValueError("Adduct transform attempted to remove the reactive atom")
    for atom_idx in remove_indices:
        editable.RemoveAtom(atom_idx)
    shift = sum(idx < original_reactive for idx in remove_indices)
    reactive_idx = original_reactive - shift

    support_idx = editable.AddAtom(Chem.Atom(6))
    nuc_idx = editable.AddAtom(Chem.Atom(anchor.atomic_number))
    editable.AddBond(support_idx, nuc_idx, Chem.BondType.SINGLE)
    editable.AddBond(nuc_idx, reactive_idx, Chem.BondType.SINGLE)

    adduct = editable.GetMol()
    try:
        Chem.SanitizeMol(adduct)
    except Exception as exc:
        raise ValueError(f"Invalid {warhead.warhead_type} adduct topology: {exc}") from exc
    return adduct, support_idx, nuc_idx, reactive_idx


def covalent_one_three_bounds(
    mol: Chem.Mol,
    support_idx: int,
    reactive_idx: int,
) -> tuple[float, float]:
    """Return RDKit's lower/upper distance bounds for the formed-bond 1-3 pair."""
    bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    low_idx = min(support_idx, reactive_idx)
    high_idx = max(support_idx, reactive_idx)
    lower = float(bounds[high_idx, low_idx])
    upper = float(bounds[low_idx, high_idx])
    return lower, upper


def select_formed_bond_geometry(
    mol: Chem.Mol,
    nucleophile_idx: int,
    reactive_idx: int,
    *,
    preferred_carbon_length: float,
) -> FormedBondGeometry:
    """Choose a formed-bond target compatible with the actual atom pair.

    Validated residue-specific targets are retained for carbon electrophiles.
    For phosphorus, sulfur, and other electrophiles, RDKit's topology-derived
    distance-geometry interval supplies a pair-specific midpoint instead of
    reusing an unrelated C-N/C-O/C-S length.
    """
    bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    low_idx = min(nucleophile_idx, reactive_idx)
    high_idx = max(nucleophile_idx, reactive_idx)
    lower = float(bounds[high_idx, low_idx])
    upper = float(bounds[low_idx, high_idx])
    nucleophile_atomic_number = mol.GetAtomWithIdx(nucleophile_idx).GetAtomicNum()
    electrophile_atomic_number = mol.GetAtomWithIdx(reactive_idx).GetAtomicNum()
    if electrophile_atomic_number == 6:
        target = float(preferred_carbon_length)
        source = "residue_carbon_reference"
    else:
        target = 0.5 * (lower + upper)
        source = "rdkit_distance_geometry_bounds_midpoint"
    return FormedBondGeometry(
        target,
        lower,
        upper,
        source,
        nucleophile_atomic_number,
        electrophile_atomic_number,
    )


def normalize_covalent_conformer(
    mol: Chem.Mol,
    conformer_id: int,
    *,
    support_idx: int,
    nucleophile_idx: int,
    reactive_idx: int,
    bond_length: float,
    bounds_tolerance: float = 0.05,
    max_bond_correction: float | None = None,
) -> bool:
    """Set formed-bond length and protein-side 1-3 geometry with a rigid branch.

    Translation sets the formed-bond length. A subsequent rotation of the same
    complete branch about the protein nucleophile sets the
    support-nucleophile-electrophile angle while preserving every ligand-local
    bond, angle, and nucleophile-to-ligand-neighbor distance.
    """
    conformer = mol.GetConformer(conformer_id)
    positions = conformer.GetPositions()
    direction = positions[reactive_idx] - positions[nucleophile_idx]
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1e-8:
        return False
    nucleophile = positions[nucleophile_idx]
    target = nucleophile + direction * (float(bond_length) / norm)
    shift = target - positions[reactive_idx]
    if max_bond_correction is not None and float(np.linalg.norm(shift)) > max_bond_correction:
        return False

    branch = _branch_indices(Chem.RWMol(mol), nucleophile_idx, reactive_idx)
    if support_idx in branch or nucleophile_idx in branch or reactive_idx not in branch:
        return False
    for atom_idx in branch:
        xyz = positions[atom_idx] + shift
        conformer.SetAtomPosition(atom_idx, Point3D(*map(float, xyz)))

    positions = conformer.GetPositions()
    support_vector = positions[support_idx] - positions[nucleophile_idx]
    support_length = float(np.linalg.norm(support_vector))
    if support_length <= 1e-8:
        return False
    one_three_lower, one_three_upper = covalent_one_three_bounds(mol, support_idx, reactive_idx)
    one_three_target = 0.5 * (one_three_lower + one_three_upper)
    cosine = (support_length**2 + float(bond_length) ** 2 - one_three_target**2) / (
        2.0 * support_length * float(bond_length)
    )
    if not -1.0 - 1e-8 <= cosine <= 1.0 + 1e-8:
        return False
    cosine = float(np.clip(cosine, -1.0, 1.0))
    sine = float(np.sqrt(max(0.0, 1.0 - cosine**2)))
    support_axis = support_vector / support_length
    current_axis = positions[reactive_idx] - positions[nucleophile_idx]
    current_axis /= np.linalg.norm(current_axis)
    perpendicular = current_axis - np.dot(current_axis, support_axis) * support_axis
    if float(np.linalg.norm(perpendicular)) <= 1e-8:
        trial = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(trial, support_axis))) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        perpendicular = trial - np.dot(trial, support_axis) * support_axis
    perpendicular /= np.linalg.norm(perpendicular)
    desired_axis = cosine * support_axis + sine * perpendicular
    rotation = _rotation_between_vectors(current_axis, desired_axis)
    for atom_idx in branch:
        xyz = nucleophile + rotation @ (positions[atom_idx] - nucleophile)
        conformer.SetAtomPosition(atom_idx, Point3D(*map(float, xyz)))

    updated = conformer.GetPositions()
    formed_length = float(np.linalg.norm(updated[reactive_idx] - updated[nucleophile_idx]))
    one_three = float(np.linalg.norm(updated[reactive_idx] - updated[support_idx]))
    return (
        abs(formed_length - float(bond_length)) <= 1e-6
        and one_three_lower - bounds_tolerance <= one_three <= one_three_upper + bounds_tolerance
    )


def _rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a stable 3D rotation taking one unit vector to another."""
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine <= 1e-10:
        if cosine > 0.0:
            return np.eye(3)
        trial = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(trial, source))) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        axis = trial - np.dot(trial, source) * source
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    skew = np.array([[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]])
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine**2))


def find_receptor_nucleophile_index(receptor_mol: Chem.Mol, anchor: AnchorPoint) -> int:
    """Locate the receptor atom represented by the adduct pseudo nucleophile."""
    matches: list[int] = []
    for atom in receptor_mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        if (
            info.GetResidueName().strip().upper() == anchor.residue_name
            and info.GetResidueNumber() == anchor.residue_num
            and info.GetInsertionCode().strip() == anchor.insertion_code
            and info.GetChainId().strip() == anchor.chain_id
            and info.GetName().strip().upper() == anchor.atom_name
        ):
            matches.append(atom.GetIdx())
    if len(matches) != 1:
        raise ValueError(f"expected one receptor atom for {anchor.residue_id} {anchor.atom_name}, found {len(matches)}")
    return matches[0]


def create_covalent_exclusion_mask(
    ligand_mol: Chem.Mol,
    receptor_mol: Chem.Mol,
    *,
    pseudo_atom_indices: set[int],
    reactive_atom_idx: int,
    receptor_nucleophile_idx: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Exclude pseudo rows and formed-bond 1-2/1-3/1-4 cross pairs.

    The intramolecular scorer excludes graph distances up to three bonds. For
    the cross-molecule bond between the ligand electrophile and receptor
    nucleophile, the equivalent condition is ``ligand_distance +
    receptor_distance <= 2``. More distant pairs remain active contacts.
    """
    mask = torch.zeros(
        ligand_mol.GetNumAtoms(),
        receptor_mol.GetNumAtoms(),
        dtype=torch.bool,
        device=torch.device(device),
    )
    for atom_idx in pseudo_atom_indices:
        if 0 <= atom_idx < mask.shape[0]:
            mask[atom_idx, :] = True
    if 0 <= reactive_atom_idx < mask.shape[0] and 0 <= receptor_nucleophile_idx < mask.shape[1]:
        ligand_distances = Chem.GetDistanceMatrix(ligand_mol)
        receptor_distances = Chem.GetDistanceMatrix(receptor_mol)
        for ligand_idx in range(mask.shape[0]):
            if ligand_idx in pseudo_atom_indices:
                continue
            ligand_distance = ligand_distances[ligand_idx, reactive_atom_idx]
            if ligand_distance > 2:
                continue
            for receptor_idx in range(mask.shape[1]):
                receptor_distance = receptor_distances[receptor_nucleophile_idx, receptor_idx]
                if ligand_distance + receptor_distance <= 2:
                    mask[ligand_idx, receptor_idx] = True
    return mask.unsqueeze(0)
