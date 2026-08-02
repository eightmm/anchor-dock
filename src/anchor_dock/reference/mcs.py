"""Deterministic MCS mapping for reference-guided docking."""

from __future__ import annotations

import heapq
from collections.abc import Iterator
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import rdFMCS

Mapping = list[tuple[int, int]]

_MAX_SUBSTRUCT_MATCHES = 4096
_MAX_ALTERNATIVE_COMPONENT_PAIRS = 16
_MAX_ALTERNATIVE_CUTS = 8
_MAX_ALTERNATIVE_COMPONENTS_PER_MOLECULE = 16
_MAX_ALTERNATIVE_PACKING_NODES = 100_000


@dataclass(frozen=True)
class MCSSelection:
    mode: str
    mappings: tuple[tuple[tuple[int, int], ...], ...]
    reason: str
    simple_size: int
    cross_size: int
    candidate_complete: bool = True
    max_size_proven: bool = True
    candidate_limit: int = 64


@dataclass(frozen=True)
class _MCSFragment:
    pattern: Chem.Mol
    reference_seed: tuple[int, ...]
    query_seed: tuple[int, ...]


@dataclass(frozen=True)
class _CutComponent:
    molecule: Chem.Mol
    original_indices: tuple[int, ...]


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


def _deduplicate_mappings(
    mappings: list[Mapping],
    max_mappings: int,
    *,
    preserve_order: bool = False,
) -> list[Mapping]:
    unique: dict[tuple[tuple[int, int], ...], Mapping] = {}
    for mapping in mappings:
        canonical = _canonical_mapping(mapping)
        unique.setdefault(canonical, list(canonical))
    ordered = list(unique.values())
    if not preserve_order:
        ordered.sort(key=lambda mapping: (-len(mapping), _canonical_mapping(mapping)))
    return ordered[:max_mappings]


def _candidate_budget(max_mappings: int) -> int:
    return min(_MAX_SUBSTRUCT_MATCHES, max(64, max_mappings * 8))


def _validate_max_mappings(max_mappings: int) -> None:
    if max_mappings <= 0:
        raise ValueError("max_mappings must be positive")
    if max_mappings > _MAX_SUBSTRUCT_MATCHES:
        raise ValueError(f"max_mappings must be <= {_MAX_SUBSTRUCT_MATCHES}")


def _iter_occurrence_pairs(
    reference_matches: tuple[tuple[int, ...], ...],
    query_matches: tuple[tuple[int, ...], ...],
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Yield occurrence pairs with early breadth on both molecules."""
    if not reference_matches or not query_matches:
        return
    width = max(len(reference_matches), len(query_matches))
    seen: set[tuple[int, int]] = set()
    for offset in range(width):
        for index in range(width):
            pair = (index % len(reference_matches), (index + offset) % len(query_matches))
            if pair in seen:
                continue
            seen.add(pair)
            yield reference_matches[pair[0]], query_matches[pair[1]]
            if len(seen) == len(reference_matches) * len(query_matches):
                return
    for reference_index, reference_match in enumerate(reference_matches):
        for query_index, query_match in enumerate(query_matches):
            if (reference_index, query_index) not in seen:
                yield reference_match, query_match


def _pattern_mapping_candidates(
    reference: Chem.Mol,
    query: Chem.Mol,
    pattern: Chem.Mol,
    *,
    match_chirality: bool,
    max_candidates: int,
) -> tuple[list[Mapping], bool]:
    """Enumerate occurrence pairs modulo common pattern automorphisms.

    ``uniquify=True`` identifies distinct atom-set occurrences in each target.
    Applying every bounded self-automorphism of the MCS pattern to one side then
    recovers symmetry-related atom correspondences without constructing the
    redundant all-automorphism Cartesian product.
    """
    pattern.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(pattern)
    match_limit = max_candidates + 1
    reference_matches_raw = reference.GetSubstructMatches(
        pattern,
        uniquify=True,
        useChirality=match_chirality,
        maxMatches=match_limit,
    )
    query_matches_raw = query.GetSubstructMatches(
        pattern,
        uniquify=True,
        useChirality=match_chirality,
        maxMatches=match_limit,
    )
    automorphisms_raw = pattern.GetSubstructMatches(
        pattern,
        uniquify=False,
        useChirality=match_chirality,
        useQueryQueryMatches=True,
        maxMatches=match_limit,
    )
    identity = tuple(range(pattern.GetNumAtoms()))
    reference_matches = tuple(sorted(set(reference_matches_raw)))[:max_candidates]
    query_matches = tuple(sorted(set(query_matches_raw)))[:max_candidates]
    automorphisms = tuple(sorted(set(automorphisms_raw) or {identity}, key=lambda value: (value != identity, value)))[
        :max_candidates
    ]

    unique: dict[tuple[tuple[int, int], ...], Mapping] = {}
    for automorphism in automorphisms:
        for reference_match, query_match in _iter_occurrence_pairs(reference_matches, query_matches):
            mapping = [
                (reference_match[index], query_match[automorphism[index]]) for index in range(pattern.GetNumAtoms())
            ]
            canonical = _canonical_mapping(mapping)
            unique.setdefault(canonical, list(canonical))
            if len(unique) >= max_candidates:
                return list(unique.values()), True
    maybe_truncated = any(
        len(matches) > max_candidates for matches in (reference_matches_raw, query_matches_raw, automorphisms_raw)
    )
    return list(unique.values()), maybe_truncated


def _simple_mcs_search(
    reference: Chem.Mol,
    query: Chem.Mol,
    *,
    min_atoms: int,
    timeout: int,
    max_mappings: int,
    match_chirality: bool,
) -> tuple[list[Mapping], bool]:
    _validate_max_mappings(max_mappings)
    reference_heavy = Chem.RemoveHs(reference)
    query_heavy = Chem.RemoveHs(query)
    result = rdFMCS.FindMCS(
        [reference_heavy, query_heavy],
        _mcs_parameters(timeout, match_chirality=match_chirality),
    )
    if result.canceled:
        raise TimeoutError(f"contiguous MCS search timed out after {timeout}s; partial result discarded")
    if not result.smartsString:
        return [], True
    pattern = Chem.MolFromSmarts(result.smartsString)
    if pattern is None or pattern.GetNumAtoms() < min_atoms:
        return [], True
    mappings, maybe_truncated = _pattern_mapping_candidates(
        reference_heavy,
        query_heavy,
        pattern,
        match_chirality=match_chirality,
        max_candidates=_candidate_budget(max_mappings),
    )
    selected = _deduplicate_mappings(mappings, max_mappings, preserve_order=True)
    complete = not maybe_truncated and len(mappings) <= max_mappings
    return selected, complete


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
    mappings, _ = _simple_mcs_search(
        reference,
        query,
        min_atoms=min_atoms,
        timeout=timeout,
        max_mappings=max_mappings,
        match_chirality=match_chirality,
    )
    return mappings


def _find_disjoint_fragments(
    reference: Chem.Mol,
    query: Chem.Mol,
    *,
    min_fragment_size: int,
    max_fragments: int,
    timeout: int,
    match_chirality: bool,
) -> list[_MCSFragment]:
    reference_copy = Chem.RWMol(Chem.RemoveHs(reference))
    query_copy = Chem.RWMol(Chem.RemoveHs(query))
    reference_original_indices = list(range(reference_copy.GetNumAtoms()))
    query_original_indices = list(range(query_copy.GetNumAtoms()))
    fragments: list[_MCSFragment] = []
    parameters = _mcs_parameters(timeout, match_chirality=match_chirality)

    for _ in range(max_fragments):
        result = rdFMCS.FindMCS([reference_copy, query_copy], parameters)
        if result.canceled:
            raise TimeoutError(f"cross MCS search timed out after {timeout}s; partial result discarded")
        if not result.smartsString:
            break
        pattern = Chem.MolFromSmarts(result.smartsString)
        if pattern is None or pattern.GetNumAtoms() < min_fragment_size:
            break
        reference_matches = reference_copy.GetSubstructMatches(
            pattern,
            uniquify=True,
            useChirality=match_chirality,
            maxMatches=1,
        )
        query_matches = query_copy.GetSubstructMatches(
            pattern,
            uniquify=True,
            useChirality=match_chirality,
            maxMatches=1,
        )
        if not reference_matches or not query_matches:
            break
        reference_seed = tuple(reference_original_indices[index] for index in reference_matches[0])
        query_seed = tuple(query_original_indices[index] for index in query_matches[0])
        fragments.append(_MCSFragment(pattern, reference_seed, query_seed))
        # Remove consumed atoms from the search copies. Element/isotope masking
        # is not reliable in rdFMCS: masked atoms can be selected again and the
        # generated SMARTS may then fail to match the other molecule. Original
        # indices are recovered later by matching each fragment to the untouched
        # input molecules.
        for editable, original_indices, match in (
            (reference_copy, reference_original_indices, reference_matches[0]),
            (query_copy, query_original_indices, query_matches[0]),
        ):
            for atom_idx in sorted(match, reverse=True):
                editable.RemoveAtom(atom_idx)
                del original_indices[atom_idx]
            editable.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(editable)
    return fragments


def _single_cut_components(
    molecule: Chem.Mol,
    other: Chem.Mol,
    *,
    min_atoms: int,
) -> list[_CutComponent]:
    """Return large components exposed by cutting unmatched articulation atoms.

    Cross-fragment anchors commonly sit on opposite sides of a linker whose
    element is absent from the other ligand.  Removing only those unmatched
    articulation atoms gives a small, deterministic set of alternative
    connected search regions without enumerating arbitrary molecular cuts.
    """
    other_elements = {atom.GetAtomicNum() for atom in other.GetAtoms()}
    adjacency = {
        atom.GetIdx(): tuple(neighbor.GetIdx() for neighbor in atom.GetNeighbors()) for atom in molecule.GetAtoms()
    }
    discovery = [-1] * molecule.GetNumAtoms()
    low = [-1] * molecule.GetNumAtoms()
    parent = [-1] * molecule.GetNumAtoms()
    subtree_size = [0] * molecule.GetNumAtoms()
    clock = 0
    ranked_cuts: list[tuple[tuple[int, int, int], int]] = []

    for fragment_indices in Chem.GetMolFrags(molecule):
        fragment_size = len(fragment_indices)
        if fragment_size < 2 * min_atoms + 1:
            continue
        root = fragment_indices[0]
        discovery[root] = clock
        low[root] = clock
        clock += 1
        subtree_size[root] = 1
        # Explicit frames avoid Python's recursion limit for large molecules.
        stack: list[tuple[int, int]] = [(root, 0)]
        while stack:
            atom_idx, neighbor_offset = stack[-1]
            neighbors = adjacency[atom_idx]
            if neighbor_offset < len(neighbors):
                neighbor_idx = neighbors[neighbor_offset]
                stack[-1] = (atom_idx, neighbor_offset + 1)
                if discovery[neighbor_idx] < 0:
                    parent[neighbor_idx] = atom_idx
                    discovery[neighbor_idx] = clock
                    low[neighbor_idx] = clock
                    clock += 1
                    subtree_size[neighbor_idx] = 1
                    stack.append((neighbor_idx, 0))
                elif neighbor_idx != parent[atom_idx]:
                    low[atom_idx] = min(low[atom_idx], discovery[neighbor_idx])
                continue

            stack.pop()
            parent_idx = parent[atom_idx]
            if parent_idx >= 0:
                subtree_size[parent_idx] += subtree_size[atom_idx]
                low[parent_idx] = min(low[parent_idx], low[atom_idx])
            atom = molecule.GetAtomWithIdx(atom_idx)
            if atom.GetAtomicNum() in other_elements or atom.GetDegree() < 2:
                continue
            separated_sizes = [
                subtree_size[neighbor_idx]
                for neighbor_idx in neighbors
                if parent[neighbor_idx] == atom_idx and low[neighbor_idx] >= discovery[atom_idx]
            ]
            remainder = fragment_size - 1 - sum(separated_sizes)
            component_sizes = separated_sizes + ([remainder] if remainder else [])
            qualifying = sorted((size for size in component_sizes if size >= min_atoms), reverse=True)
            if len(qualifying) < 2:
                continue
            # Prefer cuts exposing the largest two useful regions.  The index
            # tie-break keeps the bounded selection deterministic.
            rank = (-sum(qualifying[:2]), -qualifying[1], atom_idx)
            ranked_cuts.append((rank, atom_idx))

    selected_cuts = [atom_idx for _, atom_idx in sorted(ranked_cuts)[:_MAX_ALTERNATIVE_CUTS]]
    unique: dict[tuple[int, ...], _CutComponent] = {}
    for removed_index in selected_cuts:
        editable = Chem.RWMol(molecule)
        editable.RemoveAtom(removed_index)
        editable.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(editable)
        atom_mappings: list[tuple[int, ...]] = []
        try:
            component_molecules = Chem.GetMolFrags(
                editable,
                asMols=True,
                sanitizeFrags=True,
                fragsMolAtomMapping=atom_mappings,
            )
        except Exception:
            continue
        if len(component_molecules) < 2:
            continue
        original_indices = list(range(molecule.GetNumAtoms()))
        del original_indices[removed_index]
        for component, current_indices in zip(component_molecules, atom_mappings, strict=True):
            if component.GetNumAtoms() < min_atoms:
                continue
            original = tuple(original_indices[index] for index in current_indices)
            unique.setdefault(original, _CutComponent(component, original))
        if len(unique) >= _MAX_ALTERNATIVE_COMPONENTS_PER_MOLECULE:
            break
    return sorted(
        unique.values(),
        key=lambda component: (-component.molecule.GetNumAtoms(), component.original_indices),
    )[:_MAX_ALTERNATIVE_COMPONENTS_PER_MOLECULE]


def _alternative_fragment_candidates(
    reference: Chem.Mol,
    query: Chem.Mol,
    *,
    min_fragment_size: int,
    timeout: int,
    match_chirality: bool,
    candidate_budget: int,
) -> tuple[list[Mapping], bool]:
    """Find bounded alternative fragments exposed by unmatched linker cuts."""
    reference_components = _single_cut_components(reference, query, min_atoms=min_fragment_size)
    query_components = _single_cut_components(query, reference, min_atoms=min_fragment_size)
    pair_count = len(reference_components) * len(query_components)

    def pair_key(pair: tuple[_CutComponent, _CutComponent]) -> tuple[object, ...]:
        return (
            -min(pair[0].molecule.GetNumAtoms(), pair[1].molecule.GetNumAtoms()),
            pair[0].original_indices,
            pair[1].original_indices,
        )

    component_pairs = heapq.nsmallest(
        _MAX_ALTERNATIVE_COMPONENT_PAIRS,
        (
            (reference_component, query_component)
            for reference_component in reference_components
            for query_component in query_components
        ),
        key=pair_key,
    )
    pair_search_complete = pair_count <= _MAX_ALTERNATIVE_COMPONENT_PAIRS
    unique: dict[tuple[tuple[int, int], ...], Mapping] = {}
    for reference_component, query_component in component_pairs:
        result = rdFMCS.FindMCS(
            [reference_component.molecule, query_component.molecule],
            _mcs_parameters(timeout, match_chirality=match_chirality),
        )
        if result.canceled:
            raise TimeoutError(f"alternative cross MCS search timed out after {timeout}s; partial result discarded")
        if not result.smartsString:
            continue
        pattern = Chem.MolFromSmarts(result.smartsString)
        if pattern is None or pattern.GetNumAtoms() < min_fragment_size:
            continue
        local_mappings, local_truncated = _pattern_mapping_candidates(
            reference_component.molecule,
            query_component.molecule,
            pattern,
            match_chirality=match_chirality,
            max_candidates=candidate_budget,
        )
        pair_search_complete &= not local_truncated
        for local_mapping in local_mappings:
            mapping = [
                (
                    reference_component.original_indices[reference_index],
                    query_component.original_indices[query_index],
                )
                for reference_index, query_index in local_mapping
            ]
            canonical = _canonical_mapping(mapping)
            unique.setdefault(canonical, list(canonical))
            if len(unique) >= candidate_budget:
                return list(unique.values()), False
    return list(unique.values()), pair_search_complete


def _best_disjoint_fragment_mapping(
    candidates: list[Mapping],
    *,
    max_fragments: int,
) -> Mapping:
    """Select a maximum-size bounded packing from connected fragment mappings."""
    ordered = _deduplicate_mappings(candidates, len(candidates))
    best: Mapping = []
    visited_nodes = 0
    exhausted = False

    def visit(
        start: int,
        selected_count: int,
        current: Mapping,
        used_reference: set[int],
        used_query: set[int],
    ) -> None:
        nonlocal best, visited_nodes, exhausted
        visited_nodes += 1
        if visited_nodes > _MAX_ALTERNATIVE_PACKING_NODES:
            exhausted = True
            return
        current_canonical = _canonical_mapping(current)
        best_canonical = _canonical_mapping(best)
        if len(current_canonical) > len(best_canonical) or (
            len(current_canonical) == len(best_canonical) and current_canonical < best_canonical
        ):
            best = list(current_canonical)
        if selected_count >= max_fragments or exhausted:
            return
        remaining_slots = max_fragments - selected_count
        optimistic = len(current) + sum(len(mapping) for mapping in ordered[start : start + remaining_slots])
        if optimistic < len(best):
            return
        for index in range(start, len(ordered)):
            mapping = ordered[index]
            reference_atoms = {reference_index for reference_index, _ in mapping}
            query_atoms = {query_index for _, query_index in mapping}
            if reference_atoms & used_reference or query_atoms & used_query:
                continue
            visit(
                index + 1,
                selected_count + 1,
                current + mapping,
                used_reference | reference_atoms,
                used_query | query_atoms,
            )
            if exhausted:
                return

    visit(0, 0, [], set(), set())
    if exhausted:
        raise RuntimeError(
            "alternative cross MCS fragment packing exceeded its bounded node budget; partial packing discarded"
        )
    return best


def _iter_fragment_subsets(
    fragments: list[_MCSFragment],
    *,
    allow_partial: bool,
) -> Iterator[tuple[int, ...]]:
    """Yield non-empty fragment subsets by decreasing possible anchor size."""
    count = len(fragments)
    full_mask = (1 << count) - 1
    if not allow_partial:
        yield tuple(range(count))
        return
    weights = [fragment.pattern.GetNumAtoms() for fragment in fragments]
    heap: list[tuple[int, int]] = [(-sum(weights), full_mask)]
    seen = {full_mask}
    while heap:
        _, mask = heapq.heappop(heap)
        subset = tuple(index for index in range(count) if mask & (1 << index))
        yield subset
        for index in subset:
            child = mask & ~(1 << index)
            if child == 0 or child in seen:
                continue
            seen.add(child)
            size = sum(weights[item] for item in range(count) if child & (1 << item))
            heapq.heappush(heap, (-size, child))


def _cross_mcs_search(
    reference: Chem.Mol,
    query: Chem.Mol,
    *,
    min_fragment_size: int = 5,
    max_fragments: int = 3,
    timeout: int = 10,
    max_mappings: int = 64,
    allow_partial: bool = True,
    match_chirality: bool = False,
) -> tuple[list[Mapping], bool]:
    """Combine non-overlapping MCS fragments with bounded combinatorics."""
    _validate_max_mappings(max_mappings)
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
        return [], True

    placement_budget = _candidate_budget(max_mappings)
    placements: list[list[Mapping]] = []
    placements_maybe_truncated = False
    seed_mapping: Mapping = []
    packing_candidates: list[Mapping] = []
    for fragment in fragments:
        reference_seed_matches = reference_heavy.GetSubstructMatches(
            fragment.pattern,
            uniquify=False,
            useChirality=match_chirality,
            maxMatches=_MAX_SUBSTRUCT_MATCHES + 1,
        )
        query_seed_matches = query_heavy.GetSubstructMatches(
            fragment.pattern,
            uniquify=False,
            useChirality=match_chirality,
            maxMatches=_MAX_SUBSTRUCT_MATCHES + 1,
        )
        if fragment.reference_seed not in reference_seed_matches or fragment.query_seed not in query_seed_matches:
            raise RuntimeError("cross MCS fragment seed could not be validated in the input molecules")
        fragment_seed = list(zip(fragment.reference_seed, fragment.query_seed, strict=True))
        seed_mapping.extend(fragment_seed)
        packing_candidates.append(fragment_seed)
        candidates, maybe_truncated = _pattern_mapping_candidates(
            reference_heavy,
            query_heavy,
            fragment.pattern,
            match_chirality=match_chirality,
            max_candidates=placement_budget,
        )
        placements.append(candidates)
        placements_maybe_truncated |= maybe_truncated

    seed_canonical = _canonical_mapping(seed_mapping)
    if len({pair[0] for pair in seed_canonical}) != len(seed_canonical) or len(
        {pair[1] for pair in seed_canonical}
    ) != len(seed_canonical):
        raise RuntimeError("cross MCS fragment seeds overlap in the input molecules")
    unique: dict[tuple[tuple[int, int], ...], Mapping] = {seed_canonical: list(seed_canonical)}
    if allow_partial:
        alternative_candidates, _ = _alternative_fragment_candidates(
            reference_heavy,
            query_heavy,
            min_fragment_size=min_fragment_size,
            timeout=timeout,
            match_chirality=match_chirality,
            candidate_budget=placement_budget,
        )
        packing_candidates.extend(alternative_candidates)
        alternative_mapping = _best_disjoint_fragment_mapping(
            packing_candidates,
            max_fragments=max_fragments,
        )
        if alternative_mapping:
            alternative_canonical = _canonical_mapping(alternative_mapping)
            unique.setdefault(alternative_canonical, list(alternative_canonical))
    node_budget = min(1_000_000, max(10_000, max_mappings * 4096))
    visited_nodes = 0
    exhausted_budget = False

    def visit_subset(
        subset: tuple[int, ...],
        offset: int,
        current: Mapping,
        used_reference: set[int],
        used_query: set[int],
    ) -> None:
        nonlocal visited_nodes, exhausted_budget
        visited_nodes += 1
        if visited_nodes > node_budget:
            exhausted_budget = True
            return
        if len(unique) >= max_mappings or exhausted_budget:
            return
        if offset == len(subset):
            canonical = _canonical_mapping(current)
            unique.setdefault(canonical, list(canonical))
            return
        for mapping in placements[subset[offset]]:
            reference_atoms = {reference_idx for reference_idx, _ in mapping}
            query_atoms = {query_idx for _, query_idx in mapping}
            if reference_atoms & used_reference or query_atoms & used_query:
                continue
            visit_subset(
                subset,
                offset + 1,
                current + mapping,
                used_reference | reference_atoms,
                used_query | query_atoms,
            )
            if len(unique) >= max_mappings or exhausted_budget:
                return

    for subset in _iter_fragment_subsets(fragments, allow_partial=allow_partial):
        visit_subset(subset, 0, [], set(), set())
        if len(unique) >= max_mappings:
            break
        if exhausted_budget:
            break

    if exhausted_budget or (placements_maybe_truncated and len(unique) < max_mappings):
        raise RuntimeError(
            "cross MCS symmetry search exceeded its bounded candidate budget "
            f"(visited_nodes={visited_nodes}, node_budget={node_budget}, "
            f"placement_options={[len(options) for options in placements]}); "
            "reduce max_fragments or use single/multi mode"
        )
    # Alternative fragment discovery is bounded and therefore does not prove
    # exhaustive coverage of all connected common-subgraph packings.
    complete = False
    return _deduplicate_mappings(list(unique.values()), max_mappings), complete


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
    mappings, _ = _cross_mcs_search(
        reference,
        query,
        min_fragment_size=min_fragment_size,
        max_fragments=max_fragments,
        timeout=timeout,
        max_mappings=max_mappings,
        allow_partial=allow_partial,
        match_chirality=match_chirality,
    )
    return mappings


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
    if min_atoms <= 0:
        raise ValueError("min_atoms must be positive")
    if min_fragment_size <= 0:
        raise ValueError("min_fragment_size must be positive")
    if max_fragments <= 0:
        raise ValueError("max_fragments must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    _validate_max_mappings(max_mappings)
    requested = mode.lower()
    if requested not in {"auto", "single", "multi", "cross"}:
        raise ValueError("mode must be auto, single, multi or cross")
    simple, simple_complete = _simple_mcs_search(
        reference,
        query,
        min_atoms=min_atoms,
        timeout=timeout,
        max_mappings=max_mappings,
        match_chirality=match_chirality,
    )
    if not simple:
        raise ValueError("no MCS found between reference and query")
    if requested in {"auto", "cross"}:
        cross, cross_complete = _cross_mcs_search(
            reference,
            query,
            min_fragment_size=min_fragment_size,
            max_fragments=max_fragments,
            timeout=timeout,
            max_mappings=max_mappings,
            match_chirality=match_chirality,
        )
    else:
        cross, cross_complete = [], True
    simple_size = len(simple[0])
    cross_size = len(cross[0]) if cross else 0

    if requested == "single":
        selected, resolved, reason = simple[:1], "single", "explicit contiguous MCS"
        candidate_complete = simple_complete
        max_size_proven = True
    elif requested == "multi":
        selected, resolved, reason = simple, "multi", f"{len(simple)} contiguous placements"
        candidate_complete = simple_complete
        max_size_proven = True
    elif requested == "cross":
        if not cross:
            raise ValueError("cross MCS search produced no valid fragment combination")
        selected, resolved, reason = cross, "cross", f"{len(cross)} disjoint-fragment combinations"
        candidate_complete = cross_complete
        max_size_proven = False
    elif requested == "auto":
        if cross and cross_size > simple_size:
            selected, resolved = cross, "cross"
            reason = f"cross mapping increased anchor size from {simple_size} to {cross_size} atoms"
            candidate_complete = cross_complete
            max_size_proven = False
        elif len(simple) > 1:
            selected, resolved = simple, "multi"
            reason = f"largest contiguous MCS has {len(simple)} unique placements"
            candidate_complete = simple_complete and cross_complete
            max_size_proven = cross_complete
        else:
            selected, resolved, reason = simple[:1], "single", "one dominant contiguous MCS"
            candidate_complete = simple_complete and cross_complete
            max_size_proven = cross_complete
    if not candidate_complete:
        if requested in {"auto", "cross"} and not cross_complete:
            reason += "; bounded cross-fragment candidate search"
        if resolved != "cross" and not simple_complete:
            reason += f"; contiguous candidate set capped at {max_mappings}"
    if not max_size_proven:
        reason += "; global cross-fragment maximum not proven"
    return MCSSelection(
        resolved,
        tuple(_canonical_mapping(mapping) for mapping in selected),
        reason,
        simple_size,
        cross_size,
        candidate_complete,
        max_size_proven,
        max_mappings,
    )
