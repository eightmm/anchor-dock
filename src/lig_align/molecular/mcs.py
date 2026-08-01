"""
Maximum Common Substructure (MCS) detection with multi-position and cross-matching support.

Supports:
1. Single-position (fast, original behavior)
2. Multi-position (all ref positions for single query MCS)
3. Cross-matching (multiple query fragments × multiple ref positions)

All are unified under find_mcs_with_positions() with different parameters.
"""

from rdkit import Chem
from rdkit.Chem import rdFMCS, RWMol
from typing import List, Tuple, Optional, Dict, Any


def find_all_mcs_positions(ref_mol: Chem.Mol,
                           query_mol: Chem.Mol,
                           min_atoms: int = 3) -> List[List[Tuple[int, int]]]:
    """
    Find ALL possible MCS alignments when query matches multiple positions in reference.

    Args:
        ref_mol: Reference molecule
        query_mol: Query molecule
        min_atoms: Minimum MCS size to consider (default: 3)

    Returns:
        List of mappings, where each mapping is [(ref_idx, query_idx), ...]
        Returns empty list if no valid MCS found.

    Example:
        Reference: Ph-CH2-Ph (two benzene rings)
        Query:     Ph (one benzene)

        Returns:
            [
                [(0,0), (1,1), (2,2), (3,3), (4,4), (5,5)],  # First ring
                [(7,0), (8,1), (9,2), (10,3), (11,4), (12,5)]  # Second ring
            ]
    """
    ref_no_h = Chem.RemoveHs(ref_mol)
    query_no_h = Chem.RemoveHs(query_mol)

    # Step 1: Find MCS pattern
    mcs_res = rdFMCS.FindMCS([ref_no_h, query_no_h],
                             atomCompare=rdFMCS.AtomCompare.CompareElements,
                             bondCompare=rdFMCS.BondCompare.CompareOrderExact,
                             ringMatchesRingOnly=True,
                             timeout=10)

    if mcs_res.canceled:
        print("Warning: MCS search reached timeout limit.")

    if not mcs_res.smartsString:
        print("Warning: No common substructure found.")
        return []

    mcs_mol = Chem.MolFromSmarts(mcs_res.smartsString)

    if mcs_mol.GetNumAtoms() < min_atoms:
        print(f"Warning: MCS too small ({mcs_mol.GetNumAtoms()} atoms)")
        return []

    # Step 2: Get ALL substructure matches in reference
    ref_matches = ref_no_h.GetSubstructMatches(mcs_mol, uniquify=True)

    # Step 3: Get query match (should be unique since query is smaller)
    query_matches = query_no_h.GetSubstructMatches(mcs_mol, uniquify=True)

    if len(query_matches) == 0:
        print("Warning: Failed to match MCS back to query molecule.")
        return []

    # Use first query match as canonical
    query_match = query_matches[0]

    # Step 4: Create mappings for each reference match
    all_mappings = []
    for ref_match in ref_matches:
        if len(ref_match) != len(query_match):
            continue

        mapping = list(zip(ref_match, query_match))
        all_mappings.append(mapping)

    # Deduplicate
    all_mappings = _deduplicate_mappings(all_mappings)

    print(f"Found {len(all_mappings)} possible MCS alignment position(s) "
          f"({len(query_match)} atoms each)")

    return all_mappings


def _deduplicate_mappings(mappings: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    """
    Remove duplicate mappings (same ref atoms, possibly different order).

    Args:
        mappings: List of mappings

    Returns:
        Deduplicated list of mappings
    """
    unique_mappings = []
    seen_ref_atoms = set()

    for mapping in mappings:
        ref_atoms = tuple(sorted([m[0] for m in mapping]))

        if ref_atoms not in seen_ref_atoms:
            seen_ref_atoms.add(ref_atoms)
            unique_mappings.append(mapping)

    return unique_mappings


def _find_multi_fragment_mcs(ref_mol: Chem.Mol,
                             query_mol: Chem.Mol,
                             min_fragment_size: int = 5,
                             max_fragments: int = 3) -> List[Tuple[str, int]]:
    """
    Find multiple MCS fragments iteratively by masking matched atoms.

    Args:
        ref_mol: Reference molecule
        query_mol: Query molecule
        min_fragment_size: Minimum atoms for a fragment to be considered
        max_fragments: Maximum number of fragments to find

    Returns:
        List of (mcs_smarts, fragment_size) for each fragment found
    """
    ref_no_h = Chem.RemoveHs(ref_mol)
    query_no_h = Chem.RemoveHs(query_mol)

    fragments = []

    ref_copy = RWMol(ref_no_h)
    query_copy = RWMol(query_no_h)

    for frag_idx in range(max_fragments):
        # Find MCS on remaining (non-masked) atoms
        mcs_res = rdFMCS.FindMCS([ref_copy, query_copy],
                                 atomCompare=rdFMCS.AtomCompare.CompareElements,
                                 bondCompare=rdFMCS.BondCompare.CompareOrderExact,
                                 ringMatchesRingOnly=True,
                                 timeout=10)

        if mcs_res.canceled or not mcs_res.smartsString:
            break

        mcs_mol = Chem.MolFromSmarts(mcs_res.smartsString)
        mcs_size = mcs_mol.GetNumAtoms()

        if mcs_size < min_fragment_size:
            break

        fragments.append((mcs_res.smartsString, mcs_size))

        # Mask matched atoms (change to dummy atom type 0)
        ref_matches = ref_copy.GetSubstructMatches(mcs_mol)
        query_matches = query_copy.GetSubstructMatches(mcs_mol)

        if not ref_matches or not query_matches:
            break

        for atom_idx in ref_matches[0]:
            ref_copy.GetAtomWithIdx(atom_idx).SetAtomicNum(0)
        for atom_idx in query_matches[0]:
            query_copy.GetAtomWithIdx(atom_idx).SetAtomicNum(0)

    return fragments


def _generate_cross_combinations(ref_mol: Chem.Mol,
                                 query_mol: Chem.Mol,
                                 fragments: List[Tuple[str, int]],
                                 allow_partial: bool) -> List[List[Tuple[int, int]]]:
    """
    Generate all valid cross-combinations of fragment alignments.

    Args:
        ref_mol: Reference molecule
        query_mol: Query molecule
        fragments: List of (smarts, size) for each MCS fragment
        allow_partial: If True, allow using subset of fragments

    Returns:
        List of mappings, each mapping is [(ref_idx, query_idx), ...]
    """
    if not fragments:
        return []

    ref_no_h = Chem.RemoveHs(ref_mol)
    query_no_h = Chem.RemoveHs(query_mol)

    # Find all positions for each fragment in both molecules
    ref_positions = []
    query_positions = []

    for smarts, size in fragments:
        frag_mol = Chem.MolFromSmarts(smarts)
        ref_pos = list(ref_no_h.GetSubstructMatches(frag_mol, uniquify=True))
        query_pos = list(query_no_h.GetSubstructMatches(frag_mol, uniquify=True))
        ref_positions.append(ref_pos)
        query_positions.append(query_pos)

    # Generate all valid combinations recursively
    combinations = []

    def generate_combos(frag_idx: int,
                       current_combo: List[Tuple[Tuple[int, ...], Tuple[int, ...]]],
                       used_ref_atoms: set,
                       used_query_atoms: set):
        if frag_idx == len(fragments):
            if current_combo:
                combinations.append(current_combo[:])
            return

        # Option 1: Skip this fragment (if partial allowed)
        if allow_partial:
            generate_combos(frag_idx + 1, current_combo, used_ref_atoms, used_query_atoms)

        # Option 2: Assign this fragment
        for ref_pos in ref_positions[frag_idx]:
            for query_pos in query_positions[frag_idx]:
                # Check for atom conflicts
                ref_atoms = set(ref_pos)
                query_atoms = set(query_pos)

                if ref_atoms & used_ref_atoms or query_atoms & used_query_atoms:
                    continue  # Overlapping atoms - skip

                # Valid assignment
                current_combo.append((ref_pos, query_pos))
                generate_combos(frag_idx + 1, current_combo,
                              used_ref_atoms | ref_atoms,
                              used_query_atoms | query_atoms)
                current_combo.pop()

    generate_combos(0, [], set(), set())

    # Convert combinations to standard mapping format
    mappings = []
    seen_atom_sets = set()

    for combo in combinations:
        mapping = []
        for ref_pos, query_pos in combo:
            for ref_idx, query_idx in zip(ref_pos, query_pos):
                mapping.append((ref_idx, query_idx))

        # Deduplicate
        atom_set = frozenset(mapping)
        if atom_set not in seen_atom_sets:
            seen_atom_sets.add(atom_set)
            mappings.append(mapping)

    # Sort by size (largest first)
    mappings.sort(key=len, reverse=True)

    return mappings


def find_mcs_with_positions(ref_mol: Chem.Mol,
                            query_mol: Chem.Mol,
                            return_all: bool = False,
                            min_atoms: int = 3,
                            cross_match: bool = False,
                            min_fragment_size: Optional[int] = None,
                            max_fragments: int = 3,
                            allow_partial: bool = True) -> List[List[Tuple[int, int]]]:
    """
    Unified MCS finder supporting single/multi-position and cross-matching.

    This is the ONE function you need for all MCS alignment scenarios:
    - Single position (fast, original)
    - Multi-position (symmetric ref)
    - Cross-matching (symmetric ref AND query)

    Args:
        ref_mol: Reference molecule
        query_mol: Query molecule
        return_all: If True, return all positions. If False, return only first.
        min_atoms: Minimum MCS size for simple mode (default: 3)
        cross_match: If True, enable cross-matching (multi-fragment)
        min_fragment_size: Min atoms per fragment for cross-match (default: min_atoms)
        max_fragments: Max fragments to find for cross-match (default: 3)
        allow_partial: Allow partial fragment matching for cross-match (default: True)

    Returns:
        List of mappings. Each mapping is [(ref_idx, query_idx), ...]

    Examples:
        >>> # Mode 1: Single position (original, fastest)
        >>> mappings = find_mcs_with_positions(ref, query, return_all=False)
        >>> mapping = mappings[0]  # One mapping

        >>> # Mode 2: Multi-position (symmetric ref)
        >>> mappings = find_mcs_with_positions(ref, query, return_all=True)
        >>> # Returns all positions where query matches ref

        >>> # Mode 3: Cross-matching (symmetric ref AND query)
        >>> mappings = find_mcs_with_positions(ref, query, cross_match=True,
        ...                                   min_fragment_size=5, max_fragments=2)
        >>> # Returns all cross-combinations of multi-fragment alignments
    """
    if cross_match:
        # Mode 3: Cross-matching multi-fragment MCS
        if min_fragment_size is None:
            min_fragment_size = max(min_atoms, 5)  # Default to 5 for fragments

        print(f"Finding cross-matching MCS (min_fragment={min_fragment_size}, "
              f"max_fragments={max_fragments}, partial={allow_partial})...")

        fragments = _find_multi_fragment_mcs(ref_mol, query_mol,
                                             min_fragment_size, max_fragments)

        if not fragments:
            print("  No fragments found")
            return []

        print(f"  Found {len(fragments)} fragment(s):")
        for i, (smarts, size) in enumerate(fragments):
            print(f"    Fragment {i+1}: {size} atoms")

        mappings = _generate_cross_combinations(ref_mol, query_mol, fragments, allow_partial)

        print(f"  Generated {len(mappings)} unique combination(s)")

        return mappings

    else:
        # Mode 1 & 2: Simple single/multi-position MCS
        all_mappings = find_all_mcs_positions(ref_mol, query_mol, min_atoms)

        if not all_mappings:
            return []

        if return_all:
            return all_mappings
        else:
            return [all_mappings[0]]


def auto_select_mcs_mapping(ref_mol: Chem.Mol,
                            query_mol: Chem.Mol,
                            min_atoms: int = 3,
                            min_fragment_size: int = 5,
                            max_fragments: int = 3,
                            allow_partial: bool = True) -> Dict[str, Any]:
    """
    Automatically choose between single, multi, and cross MCS modes.

    Decision rule:
    - If there are multiple simple placements for the same largest MCS, choose `multi`.
    - Otherwise, if cross-matching produces a larger total mapping than simple MCS, choose `cross`.
    - Otherwise, choose `single`.
    """
    simple_mappings = find_all_mcs_positions(ref_mol, query_mol, min_atoms=min_atoms)
    if not simple_mappings:
        raise ValueError("No MCS found between reference and query")

    best_simple = simple_mappings[0]
    best_simple_size = len(best_simple)

    if len(simple_mappings) > 1:
        return {
            "mode": "multi",
            "mapping": best_simple,
            "mappings": simple_mappings,
            "reason": f"found {len(simple_mappings)} symmetry-equivalent placements for the same MCS",
            "stats": {
                "simple_positions": len(simple_mappings),
                "simple_size": best_simple_size,
                "cross_size": None,
                "cross_positions": 0,
            },
        }

    cross_mappings = find_mcs_with_positions(
        ref_mol,
        query_mol,
        cross_match=True,
        min_atoms=min_atoms,
        min_fragment_size=min_fragment_size,
        max_fragments=max_fragments,
        allow_partial=allow_partial,
    )

    best_cross = cross_mappings[0] if cross_mappings else None
    best_cross_size = len(best_cross) if best_cross else 0

    if best_cross and best_cross_size > best_simple_size:
        return {
            "mode": "cross",
            "mapping": best_cross,
            "mappings": cross_mappings,
            "reason": f"multi-fragment cross-matching expanded the anchor from {best_simple_size} to {best_cross_size} atoms",
            "stats": {
                "simple_positions": len(simple_mappings),
                "simple_size": best_simple_size,
                "cross_size": best_cross_size,
                "cross_positions": len(cross_mappings),
            },
        }

    return {
        "mode": "single",
        "mapping": best_simple,
        "mappings": [best_simple],
        "reason": "found one dominant contiguous MCS without symmetry ambiguity",
        "stats": {
            "simple_positions": len(simple_mappings),
            "simple_size": best_simple_size,
            "cross_size": best_cross_size if best_cross else None,
            "cross_positions": len(cross_mappings),
        },
    }
