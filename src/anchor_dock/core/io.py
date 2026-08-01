"""Shared ligand and receptor I/O."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem

from .features import compute_vina_features


@dataclass(frozen=True)
class PocketBundle:
    mol: Chem.Mol
    coords: torch.Tensor
    features: dict[str, torch.Tensor]


_POCKET_CACHE: dict[tuple[str, int, int, str], PocketBundle] = {}


def process_query_ligand(query: str) -> tuple[Chem.Mol, str]:
    """Load an SDF or parse a SMILES, then canonicalize atom ordering."""
    if query.lower().endswith(".sdf"):
        supplier = Chem.SDMolSupplier(query, removeHs=True)
        source = supplier[0] if supplier else None
        if source is None:
            raise ValueError(f"Failed to load molecule from {query}")
        smiles = Chem.MolToSmiles(source)
    else:
        smiles = query
    parsed = Chem.MolFromSmiles(smiles)
    if parsed is None:
        raise ValueError(f"Failed to parse ligand: {query}")
    canonical = Chem.MolToSmiles(parsed)
    molecule = Chem.MolFromSmiles(canonical)
    if molecule is None:
        raise ValueError(f"Failed to canonicalize ligand: {query}")
    return molecule, canonical


def load_pocket_bundle(
    protein_pdb: str,
    device: torch.device | str,
    feature_builder: Callable[[Chem.Mol, torch.device], dict[str, torch.Tensor]] = compute_vina_features,
) -> PocketBundle:
    """Load and cache a receptor pocket by path metadata and target device."""
    device = torch.device(device)
    path = os.path.abspath(protein_pdb)
    stat = os.stat(path)
    key = (path, stat.st_mtime_ns, stat.st_size, str(device))
    if key in _POCKET_CACHE:
        return _POCKET_CACHE[key]
    mol = Chem.MolFromPDBFile(path, sanitize=False, removeHs=True)
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"Failed to load protein coordinates from {protein_pdb}")
    coords = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32, device=device)
    try:
        features = feature_builder(mol, device)
    except TypeError:
        features = feature_builder(mol)
    bundle = PocketBundle(mol=mol, coords=coords, features=features)
    _POCKET_CACHE[key] = bundle
    return bundle


def clear_pocket_cache() -> None:
    _POCKET_CACHE.clear()


def extract_pocket_around_residue(
    protein_mol: Chem.Mol,
    residue_spec: str,
    cutoff: float = 12.0,
    *,
    include_heteroatoms: bool = False,
) -> Chem.Mol:
    """Extract complete residues having any atom within ``cutoff`` of an anchor residue."""
    match = re.fullmatch(r"([A-Za-z]+)(-?\d+)(?::([^:]+))?", residue_spec.strip())
    if not match:
        raise ValueError(f"Invalid residue specifier: {residue_spec}")
    target_name, target_number, target_chain = match.group(1).upper(), int(match.group(2)), match.group(3)
    if protein_mol.GetNumConformers() == 0:
        raise ValueError("Protein molecule has no conformer")
    conformer = protein_mol.GetConformer()

    target_atoms: list[int] = []
    residue_members: dict[tuple[str, int, str, bool], list[int]] = {}
    for atom in protein_mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        hetero = bool(info.GetIsHeteroAtom())
        key = (
            info.GetResidueName().strip(),
            info.GetResidueNumber(),
            info.GetChainId().strip(),
            hetero,
        )
        residue_members.setdefault(key, []).append(atom.GetIdx())
        if key[0] == target_name and key[1] == target_number and (target_chain is None or key[2] == target_chain):
            target_atoms.append(atom.GetIdx())
    if not target_atoms:
        raise ValueError(f"Residue {residue_spec} not found")

    positions = conformer.GetPositions()
    target_coords = positions[target_atoms]
    selected: set[int] = set()
    for (name, number, chain, hetero), atom_indices in residue_members.items():
        if hetero and not include_heteroatoms:
            continue
        coords = positions[atom_indices]
        minimum = np.linalg.norm(coords[:, None, :] - target_coords[None, :, :], axis=-1).min()
        if minimum <= cutoff:
            selected.update(atom_indices)
    if not selected:
        raise ValueError(f"No pocket atoms within {cutoff} Å of {residue_spec}")

    ordered = sorted(selected)
    old_to_new: dict[int, int] = {}
    editable = Chem.RWMol()
    for old_idx in ordered:
        old_to_new[old_idx] = editable.AddAtom(Chem.Atom(protein_mol.GetAtomWithIdx(old_idx)))
    for bond in protein_mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin in old_to_new and end in old_to_new:
            editable.AddBond(old_to_new[begin], old_to_new[end], bond.GetBondType())
    pocket = editable.GetMol()
    pocket_conf = Chem.Conformer(len(ordered))
    for old_idx, new_idx in old_to_new.items():
        pocket_conf.SetAtomPosition(new_idx, conformer.GetAtomPosition(old_idx))
        old_info = protein_mol.GetAtomWithIdx(old_idx).GetPDBResidueInfo()
        if old_info is not None:
            copied_info = Chem.AtomPDBResidueInfo(
                old_info.GetName(),
                old_info.GetSerialNumber(),
                old_info.GetAltLoc(),
                old_info.GetResidueName(),
                old_info.GetResidueNumber(),
                old_info.GetChainId(),
                old_info.GetInsertionCode(),
                old_info.GetOccupancy(),
                old_info.GetTempFactor(),
                old_info.GetIsHeteroAtom(),
                old_info.GetSecondaryStructure(),
                old_info.GetSegmentNumber(),
            )
            pocket.GetAtomWithIdx(new_idx).SetMonomerInfo(copied_info)
    pocket.AddConformer(pocket_conf, assignId=True)
    return pocket
