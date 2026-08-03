"""Fail-closed receptor and ligand selectors for interaction docking."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem

HARD_RAW_MATCH_CAP = 2048
_RESIDUE_PATTERN = re.compile(r"([A-Za-z]+)(-?\d+)([A-Za-z]?)(?::([^:\s]*))?")


class InteractionSelectionError(ValueError):
    """Base error for an interaction selector that cannot resolve exactly."""


class ReceptorAtomNotFoundError(InteractionSelectionError):
    """Raised when the requested receptor atom cannot be resolved exactly."""


class InvalidSmartsError(InteractionSelectionError):
    """Raised when ligand SMARTS does not define one valid mapped anchor."""


class MatchLimitExceededError(InteractionSelectionError):
    """Raised instead of silently truncating ligand anchor hypotheses."""


@dataclass(frozen=True)
class ReceptorAtomSelection:
    """Canonical provenance for one resolved standard-residue atom."""

    residue_id: str
    residue_name: str
    residue_number: int
    atom_name: str
    rdkit_index: int
    pdb_serial: int
    element: str
    chain: str
    insertion_code: str
    occupancy: float
    coordinate: tuple[float, float, float]


@dataclass(frozen=True)
class LigandAnchorMatch:
    """One distinct canonical ligand atom selected by mapped SMARTS."""

    match_index: int
    ligand_atom_index: int
    representative_match: tuple[int, ...]
    element: str
    formal_charge: int


@dataclass(frozen=True)
class _ResidueQuery:
    name: str
    number: int
    insertion_code: str
    chain: str | None


@dataclass(frozen=True)
class _PdbAtomRecord:
    residue_name: str
    residue_number: int
    insertion_code: str
    chain: str
    atom_name: str
    altloc: str
    serial: int
    element: str
    occupancy: float
    coordinate: tuple[float, float, float]

    @property
    def residue_id(self) -> str:
        return f"{self.residue_name}{self.residue_number}{self.insertion_code}:{self.chain}"


def parse_receptor_residue(value: str) -> tuple[str, int, str, str | None]:
    """Parse ``RESNAME<number><insertion>:<chain>``; an omitted chain is a unique-match query."""
    if not isinstance(value, str):
        raise TypeError("receptor_residue must be a string")
    match = _RESIDUE_PATTERN.fullmatch(value.strip())
    if match is None:
        raise InteractionSelectionError(
            f"invalid receptor residue {value!r}; expected ASP189:A, ASP189A:A, ASP189, or ASP189:"
        )
    name = match.group(1).upper()
    number = int(match.group(2))
    insertion = (match.group(3) or "").upper()
    chain = match.group(4)
    return name, number, insertion, chain


def _query(value: str) -> _ResidueQuery:
    name, number, insertion, chain = parse_receptor_residue(value)
    return _ResidueQuery(name, number, insertion, chain)


def _is_hydrogen(element: str, atom_name: str) -> bool:
    normalized = element.strip().upper()
    if normalized in {"H", "D", "T"}:
        return True
    name = atom_name.strip().upper().lstrip("0123456789")
    return not normalized and name.startswith(("H", "D", "T"))


def _parse_target_record(line: str, line_number: int) -> _PdbAtomRecord:
    if len(line) < 54:
        raise InteractionSelectionError(f"malformed PDB ATOM record at line {line_number}")
    try:
        coordinate = (
            float(line[30:38].strip()),
            float(line[38:46].strip()),
            float(line[46:54].strip()),
        )
        occupancy_text = line[54:60].strip()
        occupancy = float(occupancy_text) if occupancy_text else 1.0
        record = _PdbAtomRecord(
            residue_name=line[17:20].strip().upper(),
            residue_number=int(line[22:26].strip()),
            insertion_code=line[26].strip().upper(),
            chain=line[21].strip(),
            atom_name=line[12:16].strip().upper(),
            altloc=line[16].strip(),
            serial=int(line[6:11].strip()),
            element=(line[76:78].strip().upper() if len(line) >= 78 else ""),
            occupancy=occupancy,
            coordinate=coordinate,
        )
    except ValueError as exc:
        raise InteractionSelectionError(f"malformed PDB ATOM record at line {line_number}") from exc
    if not all(math.isfinite(value) for value in (*record.coordinate, record.occupancy)):
        raise InteractionSelectionError(f"target PDB atom has non-finite coordinates or occupancy at line {line_number}")
    return record


def _matching_pdb_records(
    pdb_path: Path,
    residue: _ResidueQuery,
    atom_name: str,
) -> list[_PdbAtomRecord]:
    matches: list[_PdbAtomRecord] = []
    with pdb_path.open(encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line[:6].strip() != "ATOM":
                continue
            if len(line) < 27:
                raise InteractionSelectionError(f"malformed PDB ATOM record at line {line_number}")
            try:
                residue_number = int(line[22:26].strip())
            except ValueError as exc:
                raise InteractionSelectionError(f"malformed PDB ATOM record at line {line_number}") from exc
            if (
                line[17:20].strip().upper() != residue.name
                or residue_number != residue.number
                or line[26].strip().upper() != residue.insertion_code
                or line[12:16].strip().upper() != atom_name
            ):
                continue
            record = _parse_target_record(line, line_number)
            if residue.chain is not None and record.chain != residue.chain:
                continue
            matches.append(record)
    return matches


def _resolve_rdkit_index(receptor: Chem.Mol, record: _PdbAtomRecord) -> int:
    if receptor.GetNumConformers() != 1:
        raise ReceptorAtomNotFoundError("receptor must contain exactly one conformer")
    candidates: list[int] = []
    for atom in receptor.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None or info.GetIsHeteroAtom():
            continue
        if (
            info.GetSerialNumber() == record.serial
            and info.GetName().strip().upper() == record.atom_name
            and info.GetResidueName().strip().upper() == record.residue_name
            and info.GetResidueNumber() == record.residue_number
            and info.GetInsertionCode().strip().upper() == record.insertion_code
            and info.GetChainId().strip() == record.chain
        ):
            candidates.append(atom.GetIdx())
    if len(candidates) != 1:
        raise ReceptorAtomNotFoundError(
            f"PDB atom {record.residue_id}/{record.atom_name} serial {record.serial} "
            f"mapped to {len(candidates)} RDKit atoms; expected exactly one"
        )
    index = candidates[0]
    atom = receptor.GetAtomWithIdx(index)
    if atom.GetAtomicNum() <= 1:
        raise InteractionSelectionError("receptor interaction atom must be a heavy atom")
    position = receptor.GetConformer().GetAtomPosition(index)
    resolved = (float(position.x), float(position.y), float(position.z))
    if not all(math.isfinite(value) for value in resolved):
        raise InteractionSelectionError("resolved RDKit receptor coordinate is not finite")
    if max(abs(first - second) for first, second in zip(resolved, record.coordinate, strict=True)) > 1e-3:
        raise InteractionSelectionError("PDB and RDKit receptor coordinates disagree for the selected atom")
    if record.element and atom.GetSymbol().upper() != record.element:
        raise InteractionSelectionError("PDB and RDKit receptor elements disagree for the selected atom")
    return index


def select_receptor_atom(
    pdb_path: str | Path,
    rdkit_receptor: Chem.Mol,
    receptor_residue: str,
    receptor_atom: str,
) -> ReceptorAtomSelection:
    """Resolve one standard-residue heavy atom without choosing an altloc or duplicate."""
    if not isinstance(receptor_atom, str) or not receptor_atom.strip():
        raise InteractionSelectionError("receptor_atom must be a non-empty PDB atom name")
    atom_name = receptor_atom.strip().upper()
    residue = _query(receptor_residue)
    path = Path(pdb_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDB file not found: {path}")
    records = _matching_pdb_records(path, residue, atom_name)
    if not records:
        raise ReceptorAtomNotFoundError(f"no ATOM record matches {receptor_residue}/{atom_name}")
    if any(record.altloc for record in records):
        locations = sorted({record.altloc for record in records if record.altloc})
        raise InteractionSelectionError(
            f"alternate locations are not supported for {receptor_residue}/{atom_name}: {locations}"
        )
    if len(records) != 1:
        choices = sorted(f"{record.residue_id}/serial={record.serial}" for record in records)
        raise InteractionSelectionError(
            f"ambiguous or duplicate receptor atom {receptor_residue}/{atom_name}: {choices}"
        )
    record = records[0]
    if _is_hydrogen(record.element, record.atom_name):
        raise InteractionSelectionError("receptor interaction atom must be a heavy atom")
    if not isinstance(rdkit_receptor, Chem.Mol):
        raise TypeError("rdkit_receptor must be an RDKit molecule")
    rdkit_index = _resolve_rdkit_index(rdkit_receptor, record)
    element = rdkit_receptor.GetAtomWithIdx(rdkit_index).GetSymbol()
    return ReceptorAtomSelection(
        residue_id=record.residue_id,
        residue_name=record.residue_name,
        residue_number=record.residue_number,
        atom_name=record.atom_name,
        rdkit_index=rdkit_index,
        pdb_serial=record.serial,
        element=element,
        chain=record.chain,
        insertion_code=record.insertion_code,
        occupancy=record.occupancy,
        coordinate=record.coordinate,
    )


def select_ligand_anchors(
    ligand: Chem.Mol,
    ligand_smarts: str,
    max_matches: int = 16,
) -> list[LigandAnchorMatch]:
    """Enumerate all distinct heavy atoms selected by the sole SMARTS ``:1`` atom."""
    if not isinstance(ligand, Chem.Mol):
        raise TypeError("ligand must be an RDKit molecule")
    if any(atom.GetAtomicNum() == 1 for atom in ligand.GetAtoms()):
        raise InteractionSelectionError("ligand selector requires the canonical heavy-atom molecule")
    if not isinstance(max_matches, int) or isinstance(max_matches, bool) or max_matches <= 0:
        raise ValueError("max_matches must be a positive integer")
    if not isinstance(ligand_smarts, str) or not ligand_smarts.strip():
        raise InvalidSmartsError("ligand_smarts must be a non-empty SMARTS string")
    query = Chem.MolFromSmarts(ligand_smarts)
    if query is None:
        raise InvalidSmartsError(f"invalid ligand SMARTS: {ligand_smarts!r}")
    mapped = [atom for atom in query.GetAtoms() if atom.GetAtomMapNum()]
    if len(mapped) != 1 or mapped[0].GetAtomMapNum() != 1:
        labels = [atom.GetAtomMapNum() for atom in mapped]
        raise InvalidSmartsError(f"ligand SMARTS must contain exactly one mapped atom and it must be :1; found {labels}")
    mapped_query_atom = mapped[0]
    if mapped_query_atom.GetAtomicNum() == 1:
        raise InvalidSmartsError("the SMARTS :1 atom must select a heavy atom")
    raw_matches = ligand.GetSubstructMatches(
        query,
        uniquify=False,
        useChirality=True,
        maxMatches=HARD_RAW_MATCH_CAP + 1,
    )
    if len(raw_matches) > HARD_RAW_MATCH_CAP:
        raise MatchLimitExceededError(
            f"SMARTS produced more than {HARD_RAW_MATCH_CAP} raw matches; refine the pattern"
        )
    query_index = mapped_query_atom.GetIdx()
    by_target: dict[int, tuple[int, ...]] = {}
    for raw_match in raw_matches:
        match = tuple(int(value) for value in raw_match)
        target = match[query_index]
        if ligand.GetAtomWithIdx(target).GetAtomicNum() <= 1:
            raise InteractionSelectionError("the resolved ligand interaction atom must be heavy")
        current = by_target.get(target)
        if current is None or match < current:
            by_target[target] = match
    if not by_target:
        raise InteractionSelectionError(f"ligand SMARTS {ligand_smarts!r} has no match")
    if len(by_target) > max_matches:
        raise MatchLimitExceededError(
            f"SMARTS resolved {len(by_target)} distinct ligand atoms, exceeding max_matches={max_matches}; "
            "refine the pattern or raise the explicit bound"
        )
    result: list[LigandAnchorMatch] = []
    for match_index, atom_index in enumerate(sorted(by_target)):
        atom = ligand.GetAtomWithIdx(atom_index)
        result.append(
            LigandAnchorMatch(
                match_index=match_index,
                ligand_atom_index=atom_index,
                representative_match=by_target[atom_index],
                element=atom.GetSymbol(),
                formal_charge=atom.GetFormalCharge(),
            )
        )
    return result
