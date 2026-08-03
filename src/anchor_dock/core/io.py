"""Ligand and receptor I/O with explicit, scorer-safe caching."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from .features import ATOM_TYPING_VERSION, compute_atom_features


@dataclass(frozen=True)
class ReceptorContext:
    """Prepared receptor coordinates and atom features."""

    mol: Chem.Mol
    coords: torch.Tensor
    features: dict[str, object]
    source_path: str | None
    device: torch.device
    structure_fingerprint: str
    source_fingerprint: str | None = None
    atom_typing_version: str = ATOM_TYPING_VERSION


_RECEPTOR_CACHE_MAX_SIZE = max(0, int(os.environ.get("ANCHOR_DOCK_RECEPTOR_CACHE_SIZE", "8")))
_RECEPTOR_CACHE: OrderedDict[tuple[str, str, str, str], ReceptorContext] = OrderedDict()


def file_content_fingerprint(path: str | os.PathLike[str]) -> str:
    """Return a content identity without exposing a machine-specific path."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def receptor_structure_fingerprint(
    coords: torch.Tensor,
    features: dict[str, object],
) -> str:
    """Fingerprint the exact receptor coordinates/features consumed by scorers."""
    digest = hashlib.sha256(b"anchor-dock-receptor-v1\0")
    values: dict[str, object] = {"coords": coords, **features}
    for key, value in sorted(values.items()):
        digest.update(key.encode("utf-8") + b"\0")
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii") + b"\0")
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
            digest.update(b"\0" + tensor.numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def choose_device(device: torch.device | str | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(device)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return result


def _read_single_molecule(path: Path, *, remove_hydrogens: bool) -> Chem.Mol:
    suffix = path.suffix.lower()
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), removeHs=remove_hydrogens)
        records = list(islice(supplier, 2))
        if len(records) != 1:
            record_count = "0" if not records else "more than one"
            raise ValueError(
                f"single-molecule input requires exactly one SDF record, found {record_count} in {path}; "
                "use dock_batch for multi-ligand SDF files"
            )
        mol = records[0]
    elif suffix == ".mol":
        mol = Chem.MolFromMolFile(str(path), removeHs=remove_hydrogens)
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=remove_hydrogens)
    elif suffix in {".pdb", ".ent"}:
        mol = Chem.MolFromPDBFile(str(path), sanitize=False, removeHs=remove_hydrogens)
    else:
        raise ValueError(f"unsupported molecule file type: {path.suffix}")
    if mol is None:
        raise ValueError(f"failed to read molecule from {path}")
    return mol


def load_ligand(
    ligand: str | os.PathLike[str] | Chem.Mol,
    *,
    add_hydrogens: bool = False,
) -> tuple[Chem.Mol, str]:
    """Load a ligand from RDKit, SMILES, InChI or a molecule file.

    The returned molecule is canonicalized through SMILES so atom ordering is
    deterministic across batch and single-ligand entry points.
    """
    if isinstance(ligand, Chem.Mol):
        source = Chem.Mol(ligand)
    else:
        text = os.fspath(ligand)
        path = Path(text)
        if path.is_file():
            source = _read_single_molecule(path, remove_hydrogens=True)
        elif text.startswith("InChI="):
            source = Chem.MolFromInchi(text)
        else:
            source = Chem.MolFromSmiles(text)
    if source is None:
        raise ValueError(f"failed to parse ligand: {ligand}")
    heavy_source = Chem.RemoveHs(source)
    if len(Chem.GetMolFrags(heavy_source)) != 1:
        raise ValueError(
            "ligand must contain exactly one connected component; salts and mixtures are not silently desalted"
        )
    canonical = Chem.MolToSmiles(heavy_source, canonical=True, isomericSmiles=True)
    molecule = Chem.MolFromSmiles(canonical)
    if molecule is None:
        raise ValueError(f"failed to canonicalize ligand: {ligand}")
    if add_hydrogens:
        molecule = Chem.AddHs(molecule)
    return molecule, canonical


def load_reference_ligand(path: str | os.PathLike[str]) -> Chem.Mol:
    """Load a reference ligand while preserving its input coordinates."""
    molecule = _read_single_molecule(Path(path), remove_hydrogens=False)
    if molecule.GetNumConformers() == 0:
        raise ValueError(f"reference ligand has no coordinates: {path}")
    if len(Chem.GetMolFrags(Chem.RemoveHs(molecule))) != 1:
        raise ValueError(
            "reference ligand must contain exactly one connected component; "
            "salts and mixtures are not silently desalted"
        )
    return molecule


def receptor_context_from_mol(
    mol: Chem.Mol,
    device: torch.device | str,
    *,
    source_path: str | None = None,
    source_fingerprint: str | None = None,
) -> ReceptorContext:
    device = choose_device(device)
    if mol.GetNumConformers() == 0:
        raise ValueError("receptor molecule has no conformer")
    coords = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32, device=device)
    features = compute_atom_features(mol, device)
    return ReceptorContext(
        Chem.Mol(mol),
        coords,
        features,
        source_path,
        device,
        receptor_structure_fingerprint(coords, features),
        source_fingerprint,
    )


def load_receptor_context(
    protein_pdb: str | os.PathLike[str],
    device: torch.device | str | None = None,
) -> ReceptorContext:
    """Load and cache a PDB receptor using atom-typing-aware cache keys."""
    target_device = choose_device(device)
    path = os.path.abspath(os.fspath(protein_pdb))
    source_fingerprint = file_content_fingerprint(path)
    key = (path, source_fingerprint, str(target_device), ATOM_TYPING_VERSION)
    cached = _RECEPTOR_CACHE.get(key)
    if cached is not None:
        _RECEPTOR_CACHE.move_to_end(key)
        return cached
    mol = Chem.MolFromPDBFile(path, sanitize=False, removeHs=True)
    if mol is None:
        raise ValueError(f"failed to load receptor from {protein_pdb}")
    context = receptor_context_from_mol(
        mol,
        target_device,
        source_path=path,
        source_fingerprint=source_fingerprint,
    )
    if _RECEPTOR_CACHE_MAX_SIZE:
        for stale_key in list(_RECEPTOR_CACHE):
            if stale_key[0] == path and stale_key[2] == str(target_device) and stale_key != key:
                _RECEPTOR_CACHE.pop(stale_key)
        _RECEPTOR_CACHE[key] = context
        while len(_RECEPTOR_CACHE) > _RECEPTOR_CACHE_MAX_SIZE:
            _RECEPTOR_CACHE.popitem(last=False)
    return context


def clear_receptor_cache() -> None:
    _RECEPTOR_CACHE.clear()


def _parse_residue_spec(spec: str) -> tuple[str, int, str | None, str | None]:
    match = re.fullmatch(r"([A-Za-z]+)(-?\d+)([A-Za-z]?)?(?::([^:]*))?", spec.strip())
    if not match:
        raise ValueError(f"invalid residue specifier: {spec!r}; expected CYS145:A or CYS145A:A")
    residue = match.group(1).upper()
    number = int(match.group(2))
    insertion = match.group(3) or None
    chain = match.group(4)
    return residue, number, insertion, chain


def extract_pocket_around_residue(
    protein_mol: Chem.Mol,
    residue_spec: str,
    cutoff: float = 12.0,
    *,
    include_heteroatoms: bool = True,
) -> Chem.Mol:
    """Extract complete residues having any atom within ``cutoff`` of a residue."""
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    target_name, target_number, target_insertion, target_chain = _parse_residue_spec(residue_spec)
    if protein_mol.GetNumConformers() == 0:
        raise ValueError("protein molecule has no conformer")
    conformer = protein_mol.GetConformer()

    target_atoms: list[int] = []
    residue_members: dict[tuple[str, int, str, str, bool], list[int]] = {}
    for atom in protein_mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        key = (
            info.GetResidueName().strip().upper(),
            info.GetResidueNumber(),
            info.GetInsertionCode().strip(),
            info.GetChainId().strip(),
            bool(info.GetIsHeteroAtom()),
        )
        residue_members.setdefault(key, []).append(atom.GetIdx())
        if (
            key[0] == target_name
            and key[1] == target_number
            and (target_insertion is None or key[2] == target_insertion)
            and (target_chain is None or key[3] == target_chain)
        ):
            target_atoms.append(atom.GetIdx())
    if not target_atoms:
        raise ValueError(f"residue {residue_spec} not found")

    positions = conformer.GetPositions()
    target_coords = positions[target_atoms]
    selected: set[int] = set()
    for (*_, hetero), atom_indices in residue_members.items():
        if hetero and not include_heteroatoms:
            continue
        coords = positions[atom_indices]
        if np.linalg.norm(coords[:, None, :] - target_coords[None, :, :], axis=-1).min() <= cutoff:
            selected.update(atom_indices)
    if not selected:
        raise ValueError(f"no pocket atoms within {cutoff} Å of {residue_spec}")

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
