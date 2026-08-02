"""Deterministic MCS mapping for reference-guided docking."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from rdkit import Chem
from rdkit.Chem import rdFMCS

Mapping = list[tuple[int, int]]


@dataclass(frozen=True)
class MCSSelection:
    mode: str
    mappings: tuple[tuple[tuple[int, int], ...], ...]
    reason: str
    simple_size: int
    cross_size: int


def _mcs_parameters(timeout: int, *, match_chirality: bool = False) -> rdFMCS.MCSParameters:
    parameters = rdFMCS.MCSParameters()
    parameters.Timeout = max(1, int(timeout))
    parameters.AtomTyper = rdFMCS.AtomCompare.CompareElements
    parameters.BondTyper = rdFMCS.BondCompare.CompareOrderExact
    parameters.AtomCompareParameters.RingMatchesRingOnly = True
    parameters.AtomCompareParameters.CompleteRingsOnly = True
    parameters.AtomCompareParameters.MatchValences = True
    parameters.AtomCompareParameters.MatchChiralTag = match_chirality
    parameters.BondCompareParameters.RingMatchesRingOnly = True
    parameters.BondCompareParameters.CompleteRingsOnly = True
    parameters.BondCompareParameters.MatchStereo = match_chirality
    return parameters


def _canonical_mapping(mapping: Mapping) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(ref_idx), int(query_idx)) for ref_idx, query_idx in mapping))


def _deduplicate_mappings(mappings: list[Mapping], max_mappings: int) -> list[Mapping]:
    unique: dict[tuple[tuple[int, int], ...], Mapping] = {}
    for mapping in mappings:
        canonical = _canonical_mapping(mapping)
        unique.setdefault(canonical, list(canonical))
    ordered = sorted(unique.values(), key=lambda mapping: (-len(mapping), _canonical_mapping(mapping)))
    return ordered[:max_mappings]


def find_simple_mcs_mappings(
    reference: Chem.Mol,
    query: Chem.Mol,
    *,
    min_atoms: int = 3,
    timeout: int = 10,
    max_mappings: int = 64,
    match_chirality: bool = False,
) -> list[Mapping]:
    """Find all unique placements of the largest contiguous MCS in both molecules."""
    reference_heavy = Chem.RemoveHs(reference)
    query_heavy = Chem.RemoveHs(query)
    result = rdFMCS.FindMCS([reference_heavy, query_heavy], _mcs_parameters(timeout, match_chirality=match_chirality))
    if not result.smartsString:
        return []
    pattern = Chem.MolFromSmarts(result.smartsString)
    if pattern is None or pattern.GetNumAtoms() < min_atoms:
        return []
    reference_matches = reference_heavy.GetSubstructMatches(pattern, uniquify=True, useChirality=match_chirality)
    query_matches = query_heavy.GetSubstructMatches(pattern, uniquify=True, useChirality=match_chirality)
    mappings = [list(zip(ref_match, query_match, strict=True)) for ref_match, query_match in product(reference_matches, query_matches)]
    return _deduplicate_mappings(mappings, max_mappings)


def _find_disjoint_fragments(
    reference: Chem.Mol,
    query: Chem.Mol,
    *,
    min_fragment_size: int,
    max_fragments: int,
    timeout: int,
    match_chirality: bool,
) -> list[Chem.Mol]:
    reference_copy = Chem.RWMol(Chem.RemoveHs(reference))
    query_copy = Chem.RWMol(Chem.RemoveHs(query))
    fragments: list[Chem.Mol] = []
    parameters = _mcs_parameters(timeout, match_chirality=match_chirality)
    parameters.AtomCompareParameters.MatchIsotope = True

    for fragment_idx in range(max_fragments):
        result = rdFMCS.FindMCS([reference_copy, query_copy], parameters)
        if not result.smartsString:
            break
        pattern = Chem.MolFromSmarts(result.smartsString)
        if pattern is None or pattern.GetNumAtoms() < min_fragment_size:
            break
        reference_matches = reference_copy.GetSubstructMatches(pattern, uniquify=True)
        query_matches = query_copy.GetSubstructMatches(pattern, uniquify=True)
        if not reference_matches or not query_matches:
            break
        fragments.append(pattern)
        # Mark consumed atoms with molecule-specific isotopes so subsequent MCS
        # searches cannot rematch masked atoms while original indices stay intact.
        for atom_idx in reference_matches[0]:
            atom = reference_copy.GetAtomWithIdx(atom_idx)
            atom.SetAtomicNum(0)
            atom.SetIsotope(900 + fragment_idx)
        for atom_idx in query_matches[0]:
            atom = query_copy.GetAtomWithIdx(atom_idx)
            atom.SetAtomicNum(0)
            atom.SetIsotope(1900 + fragment_idx)
    return fragments


def find_cross_mcs_mappings(
    reference: Chem.Mol,
    query: Chem.Mol,
    *,
    min_fragment_size: int = 5,
    max_fragments: int = 3,
    timeout: int = 10,
    max_mappings: int = 64,
    allow_partial: bool = True,
    match_chirality: bool = False,
) -> list[Mapping]:
    """Combine non-overlapping MCS fragments with bounded combinatorics."""
    reference_heavy = Chem.RemoveHs(reference)
    query_heavy = Chem.RemoveHs(query)
    fragments = _find_disjoint_fragments(
        reference,
        query,
        min_fragment_size=min_fragment_size,
        max_fragments=max_fragments,
        timeout=timeout,
        match_chirality=match_chirality,
    )
    if not fragments:
        return []

    placements: list[list[tuple[tuple[int, ...], tuple[int, ...]]]] = []
    for fragment in fragments:
        ref_matches = reference_heavy.GetSubstructMatches(fragment, uniquify=True, useChirality=match_chirality)
        query_matches = query_heavy.GetSubstructMatches(fragment, uniquify=True, useChirality=match_chirality)
        placements.append(list(product(ref_matches, query_matches)))

    mappings: list[Mapping] = []

    def visit(
        fragment_idx: int,
        current: Mapping,
        used_reference: set[int],
        used_query: set[int],
    ) -> None:
        if len(mappings) >= max_mappings * 4:
            return
        if fragment_idx == len(placements):
            if current:
                mappings.append(current.copy())
            return
        if allow_partial:
            visit(fragment_idx + 1, current, used_reference, used_query)
        for ref_match, query_match in placements[fragment_idx]:
            ref_atoms = set(ref_match)
            query_atoms = set(query_match)
            if ref_atoms & used_reference or query_atoms & used_query:
                continue
            pairs = list(zip(ref_match, query_match, strict=True))
            visit(
                fragment_idx + 1,
                current + pairs,
                used_reference | ref_atoms,
                used_query | query_atoms,
            )

    visit(0, [], set(), set())
    return _deduplicate_mappings(mappings, max_mappings)


def select_mcs_mappings(
    reference: Chem.Mol,
    query: Chem.Mol,
    *,
    mode: str = "auto",
    min_atoms: int = 3,
    min_fragment_size: int = 5,
    max_fragments: int = 3,
    timeout: int = 10,
    max_mappings: int = 64,
    match_chirality: bool = False,
) -> MCSSelection:
    """Select single, multi or cross mappings using all candidate families."""
    simple = find_simple_mcs_mappings(
        reference,
        query,
        min_atoms=min_atoms,
        timeout=timeout,
        max_mappings=max_mappings,
        match_chirality=match_chirality,
    )
    if not simple:
        raise ValueError("no MCS found between reference and query")
    cross = find_cross_mcs_mappings(
        reference,
        query,
        min_fragment_size=min_fragment_size,
        max_fragments=max_fragments,
        timeout=timeout,
        max_mappings=max_mappings,
        match_chirality=match_chirality,
    )
    simple_size = len(simple[0])
    cross_size = len(cross[0]) if cross else 0

    requested = mode.lower()
    if requested == "single":
        selected, resolved, reason = simple[:1], "single", "explicit contiguous MCS"
    elif requested == "multi":
        selected, resolved, reason = simple, "multi", f"{len(simple)} contiguous placements"
    elif requested == "cross":
        if not cross:
            raise ValueError("cross MCS search produced no valid fragment combination")
        selected, resolved, reason = cross, "cross", f"{len(cross)} disjoint-fragment combinations"
    elif requested == "auto":
        if cross and cross_size > simple_size:
            selected, resolved = cross, "cross"
            reason = f"cross mapping increased anchor size from {simple_size} to {cross_size} atoms"
        elif len(simple) > 1:
            selected, resolved = simple, "multi"
            reason = f"largest contiguous MCS has {len(simple)} unique placements"
        else:
            selected, resolved, reason = simple[:1], "single", "one dominant contiguous MCS"
    else:
        raise ValueError("mode must be auto, single, multi or cross")

    return MCSSelection(
        resolved,
        tuple(_canonical_mapping(mapping) for mapping in selected),
        reason,
        simple_size,
        cross_size,
    )
