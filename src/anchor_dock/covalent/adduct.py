"""Covalent adduct topology construction and scoring exclusions."""

from __future__ import annotations

from rdkit import Chem
import torch
import numpy as np

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
        if not _reduce_first_bond(
            editable, original_reactive, Chem.BondType.SINGLE, Chem.BondType.SINGLE,
            allowed_atomic_numbers={16},
        ):
            raise ValueError("Could not identify disulfide bond")
        # Exchange requires the distal sulfur fragment to leave. Remove the matched distal S-C side when unambiguous.
        reactive = editable.GetAtomWithIdx(original_reactive)
        sulfur_neighbors = [n.GetIdx() for n in reactive.GetNeighbors() if n.GetAtomicNum() == 16]
        if sulfur_neighbors:
            _replace_bond(editable, original_reactive, sulfur_neighbors[0], None)
    elif warhead.warhead_type == "boronic_acid":
        editable.GetAtomWithIdx(original_reactive).SetFormalCharge(-1)
    elif warhead.warhead_type in {"phosphonate", "nhs_ester", "tfe_ester"}:
        # Hypervalent P and acyl substitution are represented as an attached adduct.
        pass
    elif not positions:
        raise ValueError(f"Unsupported adduct transform for {warhead.warhead_type}")

    for atom_idx in sorted(remove_indices, reverse=True):
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
    conformer = pocket_mol.GetConformer()
    closest, distance = None, float("inf")
    for atom in pocket_mol.GetAtoms():
        current = np.linalg.norm(np.asarray(conformer.GetAtomPosition(atom.GetIdx()), dtype=float) - anchor.coord)
        if current < distance:
            closest, distance = atom.GetIdx(), current
    if closest is None or distance > 0.5:
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
