"""
Test all three MCS modes via both CLI and Python API.
"""
import subprocess
import sys

from rdkit import Chem


def test_mode1_single_cli():
    """Test Mode 1 (single position) via CLI"""
    print("\n" + "="*60)
    print("TEST 1: Mode 1 (Single Position) via CLI")
    print("="*60)

    cmd = [
        sys.executable, "scripts/run_pipeline.py",
        "-p", "examples/10gs/10gs_pocket.pdb",
        "-r", "examples/10gs/10gs_ligand.sdf",
        "-q", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
        "-o", "test_outputs/mcs_mode1",
        "-n", "50",
        "--mcs_mode", "single"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)

    assert result.returncode == 0, f"Mode 1 CLI failed: {result.stderr}"
    assert "MCS Mode: single" in result.stdout
    print("✓ Mode 1 (Single) CLI test passed")


def test_mode2_multi_cli():
    """Test Mode 2 (multi-position) via CLI with symmetric molecule"""
    print("\n" + "="*60)
    print("TEST 2: Mode 2 (Multi-Position) via CLI")
    print("="*60)

    # Dibenzyl as reference (symmetric)
    cmd = [
        sys.executable, "scripts/run_pipeline.py",
        "-p", "examples/10gs/10gs_pocket.pdb",
        "-r", "examples/10gs/10gs_ligand.sdf",
        "-q", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
        "-o", "test_outputs/mcs_mode2",
        "-n", "50",
        "--mcs_mode", "multi"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)

    assert result.returncode == 0, f"Mode 2 CLI failed: {result.stderr}"
    assert "MCS Mode: multi" in result.stdout
    assert "possible MCS alignment positions" in result.stdout
    print("✓ Mode 2 (Multi) CLI test passed")


def test_mode3_cross_cli():
    """Test Mode 3 (cross-matching) via CLI"""
    print("\n" + "="*60)
    print("TEST 3: Mode 3 (Cross-Matching) via CLI")
    print("="*60)

    cmd = [
        sys.executable, "scripts/run_pipeline.py",
        "-p", "examples/10gs/10gs_pocket.pdb",
        "-r", "examples/10gs/10gs_ligand.sdf",
        "-q", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
        "-o", "test_outputs/mcs_mode3",
        "-n", "50",
        "--mcs_mode", "cross",
        "--min_fragment_size", "5",
        "--max_fragments", "3"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)

    assert result.returncode == 0, f"Mode 3 CLI failed: {result.stderr}"
    assert "MCS Mode: cross" in result.stdout
    assert "cross-matching combinations" in result.stdout
    print("✓ Mode 3 (Cross) CLI test passed")


def test_python_api_all_modes():
    """Test all three modes via Python API"""
    print("\n" + "="*60)
    print("TEST 4: All Modes via Python API")
    print("="*60)

    import torch

    from lig_align.aligner import LigandAligner

    # Load molecules
    ref_suppl = Chem.SDMolSupplier("examples/10gs/10gs_ligand.sdf")
    ref_mol = ref_suppl[0]
    query_mol = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    query_mol = Chem.AddHs(query_mol)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    aligner = LigandAligner(device=device)

    # Mode 1: Single position
    print("\n--- Mode 1: Single Position ---")
    mapping1 = aligner.step2_find_mcs(ref_mol, query_mol,
                                      return_all_positions=False,
                                      cross_match=False)
    print(f"Mode 1 returned: {type(mapping1)}, length={len(mapping1)}")
    assert isinstance(mapping1, list), "Mode 1 should return list of tuples"
    assert isinstance(mapping1[0], tuple), "Mode 1 elements should be tuples"
    print(f"✓ Mode 1: Found {len(mapping1)} MCS atoms")

    # Mode 2: Multi-position
    print("\n--- Mode 2: Multi-Position ---")
    mappings2 = aligner.step2_find_mcs(ref_mol, query_mol,
                                       return_all_positions=True,
                                       cross_match=False)
    print(f"Mode 2 returned: {type(mappings2)}, length={len(mappings2)}")
    assert isinstance(mappings2, list), "Mode 2 should return list of mappings"
    assert isinstance(mappings2[0], list), "Mode 2 elements should be lists"
    print(f"✓ Mode 2: Found {len(mappings2)} possible alignment positions")

    # Mode 3: Cross-matching
    print("\n--- Mode 3: Cross-Matching ---")
    mappings3 = aligner.step2_find_mcs(ref_mol, query_mol,
                                       cross_match=True,
                                       min_fragment_size=5,
                                       max_fragments=3)
    print(f"Mode 3 returned: {type(mappings3)}, length={len(mappings3)}")
    assert isinstance(mappings3, list), "Mode 3 should return list of mappings"
    print(f"✓ Mode 3: Found {len(mappings3)} cross-matching combinations")

    print("\n✓ All Python API tests passed!")


if __name__ == "__main__":
    import os
    os.makedirs("test_outputs", exist_ok=True)

    # Test CLI modes
    test_mode1_single_cli()
    test_mode2_multi_cli()
    test_mode3_cross_cli()

    # Test Python API
    test_python_api_all_modes()

    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)
