"""
Test unified MCS API - ONE function for all scenarios.

Demonstrates that find_mcs_with_positions() handles:
1. Single position (1:1)
2. Multi-position (1:N)
3. Cross-matching (N:M)

All with the same function, just different parameters!
"""

import sys

sys.path.insert(0, '/home/jaemin/project/protein-ligand/lig-align')

from rdkit import Chem

from src.lig_align.molecular.mcs import find_mcs_with_positions


def test_all_modes():
    """
    Test all 3 modes with ONE unified API
    """
    print("=" * 80)
    print("UNIFIED MCS API - ONE Function for All Scenarios")
    print("=" * 80)

    # Test molecules
    ref_smiles = "c1ccccc1Cc2ccccc2"  # Ph-CH2-Ph
    query_smiles = "c1ccccc1Cc2ccccc2"  # Same (symmetric case)

    ref_mol = Chem.MolFromSmiles(ref_smiles)
    ref_mol = Chem.AddHs(ref_mol)
    query_mol = Chem.MolFromSmiles(query_smiles)
    query_mol = Chem.AddHs(query_mol)

    print("\nTest molecules:")
    print(f"  Ref:   {ref_smiles} (symmetric)")
    print(f"  Query: {query_smiles} (symmetric)")

    # Mode 1: Single position (1:1) - FASTEST
    print("\n" + "=" * 80)
    print("Mode 1: Single Position (1:1) - Original Behavior")
    print("=" * 80)

    mappings = find_mcs_with_positions(
        ref_mol, query_mol,
        return_all=False  # Just first one
    )

    print(f"Result: {len(mappings)} mapping(s)")
    for i, mapping in enumerate(mappings):
        print(f"  Mapping {i+1}: {len(mapping)} atom pairs")

    # Mode 2: Multi-position (1:N) - Symmetric ref
    print("\n" + "=" * 80)
    print("Mode 2: Multi-Position (1:N) - All Ref Positions")
    print("=" * 80)

    mappings = find_mcs_with_positions(
        ref_mol, query_mol,
        return_all=True  # All positions
    )

    print(f"Result: {len(mappings)} mapping(s)")
    for i, mapping in enumerate(mappings):
        ref_atoms = sorted(set(m[0] for m in mapping))
        print(f"  Mapping {i+1}: {len(mapping)} atom pairs, ref atoms = {ref_atoms}")

    # Mode 3: Cross-matching (N:M) - Most comprehensive
    print("\n" + "=" * 80)
    print("Mode 3: Cross-Matching (N:M) - All Fragment Combinations")
    print("=" * 80)

    mappings = find_mcs_with_positions(
        ref_mol, query_mol,
        cross_match=True,        # Enable cross-matching
        min_fragment_size=5,     # Min 5 atoms per fragment
        max_fragments=2,         # Max 2 fragments
        allow_partial=True       # Allow using subset
    )

    print(f"\nResult: {len(mappings)} mapping(s)")
    for i, mapping in enumerate(mappings[:5]):  # Show first 5
        ref_atoms = sorted(set(m[0] for m in mapping))
        query_atoms = sorted(set(m[1] for m in mapping))
        print(f"  Mapping {i+1}: {len(mapping)} atom pairs")
        print(f"    Ref:   {ref_atoms}")
        print(f"    Query: {query_atoms}")

    if len(mappings) > 5:
        print(f"  ... ({len(mappings) - 5} more)")


def test_asymmetric_case():
    """
    Test with asymmetric query to show Mode 2 working
    """
    print("\n" + "=" * 80)
    print("Asymmetric Query Test (Ph-CH2-Ph ref vs Ph query)")
    print("=" * 80)

    ref_smiles = "c1ccccc1Cc2ccccc2"
    query_smiles = "c1ccccc1"  # Single benzene

    ref_mol = Chem.MolFromSmiles(ref_smiles)
    ref_mol = Chem.AddHs(ref_mol)
    query_mol = Chem.MolFromSmiles(query_smiles)
    query_mol = Chem.AddHs(query_mol)

    print(f"\nRef:   {ref_smiles}")
    print(f"Query: {query_smiles}")

    # Mode 2: Should find 2 positions (left ring, right ring)
    print("\nMode 2: Multi-position")
    mappings = find_mcs_with_positions(ref_mol, query_mol, return_all=True)

    print(f"Found {len(mappings)} position(s):")
    for i, mapping in enumerate(mappings):
        ref_atoms = sorted(set(m[0] for m in mapping))
        print(f"  Position {i+1}: ref atoms = {ref_atoms}")


def main():
    print("\n" + "=" * 80)
    print("UNIFIED MCS API DEMONSTRATION")
    print("=" * 80)
    print("""
Question: "1:1은 all:all의 하위집합이니까 코드 하나면 되는거 아니니?"

Answer: ✓ EXACTLY! One function handles everything!
    """)

    test_all_modes()
    test_asymmetric_case()

    print("\n" + "=" * 80)
    print("SUMMARY: One Function, Three Modes")
    print("=" * 80)
    print("""
find_mcs_with_positions(ref, query, ...) handles:

┌────────────────────────────────────────────────────────────┐
│ Mode 1: Single Position (1:1)                             │
│ ---------------------------------------------------------- │
│ find_mcs_with_positions(ref, query, return_all=False)     │
│                                                            │
│ - Fastest (original behavior)                             │
│ - Returns first matching position only                    │
│ - Use when: Speed matters, no symmetry issues             │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ Mode 2: Multi-Position (1:N)                              │
│ ---------------------------------------------------------- │
│ find_mcs_with_positions(ref, query, return_all=True)      │
│                                                            │
│ - Fast (single MCS search + multiple matches)             │
│ - Returns all ref positions where query matches           │
│ - Use when: Symmetric reference molecule                  │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ Mode 3: Cross-Matching (N:M)                              │
│ ---------------------------------------------------------- │
│ find_mcs_with_positions(ref, query,                       │
│     cross_match=True,                                      │
│     min_fragment_size=5,                                   │
│     max_fragments=2)                                       │
│                                                            │
│ - Most comprehensive                                       │
│ - Returns all fragment cross-combinations                 │
│ - Use when: Both ref AND query are symmetric              │
│ - Finds swapped alignments automatically                  │
└────────────────────────────────────────────────────────────┘

Benefits of unified API:
✓ One function to learn
✓ Consistent interface
✓ Easy to upgrade (return_all=False → True)
✓ Progressive complexity (add cross_match when needed)

Code simplicity:
- Old: 3 separate functions
- New: 1 function with parameters
- User gets exactly what they need with right params
    """)


if __name__ == "__main__":
    main()
