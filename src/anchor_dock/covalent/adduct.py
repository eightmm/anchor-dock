"""Covalent adduct topology construction and scoring exclusions."""

from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem

from .anchor import AnchorPoint, WarheadHit

LEAVING_GROUP_MATCH_POSITIONS: dict[str, tuple[int, ...]] = {
    "chloroacetamide": (0,), "bromoacetamide": (0,), "iodoacetamide": (0,),
    "fluoroacetamide": (0,), "chlorofluoroacetamide": (0, 2),
    "acyl_fluoride": (2,), "sulfonyl_fluoride": (0,),
}

MICHAEL_WARHEADS = {
    "acrylamide", "acrylic_acid", "acrylate", "enone", "vinyl_sulfonamide",
    "vinyl_sulfone", "maleimide", "cyanoacrylamide",
}
RING_OPENING_WARHEADS = {"epoxide", "aziridine", "thiirane"}
TRIPLE_BOND_WARHEADS = {"aryl_nitrile", "alkyl_nitrile", "propiolamide", "propargylamide"}
CARBONYL_ADDITION_WARHEADS = {"aldehyde", "alpha_ketoamide"}


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
) -> tuple[Chem.Mol, int | None, int, int]:
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
            if any(original_reactive in ring and other in ring and len(ring) == 3 for ring in editable.GetRingInfo().AtomRings()):
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
            editable, original_reactive, Chem.BondType.DOUBLE, Chem.BondType.SINGLE,
            allowed_atomic_numbers={8},
        ):
            raise ValueError(f"Could not identify carbonyl for {warhead.warhead_type}")
    elif warhead.warhead_type == "isothiocyanate":
        if not _reduce_first_bond(
            editable, original_reactive, Chem.BondType.DOUBLE, Chem.BondType.SINGLE,
            allowed_atomic_numbers={7},
        ):
            raise ValueError("Could not reduce isothiocyanate C=N bond")
    elif warhead.warhead_type == "disulfide":
        distal_sulfur = _find_single_bond_neighbor(editable, original_reactive, 16)
        if distal_sulfur is None:
            raise ValueError("Could not identify disulfide leaving branch")
        remove_indices.extend(_branch_indices(editable, original_reactive, distal_sulfur))
    elif warhead.warhead_type == "boronic_acid":
        editable.GetAtomWithIdx(original_reactive).SetFormalCharge(-1)
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

    cb_idx = editable.AddAtom(Chem.Atom(6)) if anchor.cb_coord is not None else None
    nucleophile_atomic_number = {"SG": 16, "OG": 8, "OG1": 8, "OH": 8, "NZ": 7, "NE2": 7}[anchor.atom_name]
    nuc_idx = editable.AddAtom(Chem.Atom(nucleophile_atomic_number))
    if cb_idx is not None:
        editable.AddBond(cb_idx, nuc_idx, Chem.BondType.SINGLE)
    editable.AddBond(nuc_idx, reactive_idx, Chem.BondType.SINGLE)

    adduct = editable.GetMol()
    try:
        Chem.SanitizeMol(adduct)
    except Exception as exc:
        raise ValueError(f"Invalid {warhead.warhead_type} adduct topology: {exc}") from exc
    return adduct, cb_idx, nuc_idx, reactive_idx


def get_protein_exclusion_atom_indices(
    pocket_mol: Chem.Mol,
    anchor: AnchorPoint,
    n_hop_exclude: int = 0,
) -> set[int]:
    positions = pocket_mol.GetConformer().GetPositions()
    if positions.shape[0] == 0:
        return set()
    distances = np.linalg.norm(positions - anchor.coord, axis=1)
    closest = int(distances.argmin())
    if distances[closest] > 0.5:
        return set()
    result = {closest}
    for _ in range(n_hop_exclude):
        result.update(n.GetIdx() for idx in tuple(result) for n in pocket_mol.GetAtomWithIdx(idx).GetNeighbors())
    return result


def create_intermolecular_exclusion_mask(
    ligand_mol: Chem.Mol,
    protein_mol: Chem.Mol,
    ligand_exclude_indices: set[int],
    protein_exclude_atom_indices: set[int],
    device: torch.device | str,
) -> torch.Tensor:
    mask = torch.zeros(
        ligand_mol.GetNumAtoms(), protein_mol.GetNumAtoms(), dtype=torch.bool, device=torch.device(device)
    )
    for idx in ligand_exclude_indices:
        if 0 <= idx < mask.shape[0]:
            mask[idx, :] = True
    for idx in protein_exclude_atom_indices:
        if 0 <= idx < mask.shape[1]:
            mask[:, idx] = True
    return mask.unsqueeze(0)
