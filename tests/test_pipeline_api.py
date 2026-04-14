"""
Test the high-level run_pipeline() API function.
"""
import os
import tempfile
import shutil


def test_basic_usage():
    """Test basic usage with minimal parameters"""
    print("\n" + "="*60)
    print("TEST 1: Basic Usage")
    print("="*60)

    from lig_align import run_pipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        results = run_pipeline(
            protein_pdb="examples/10gs/10gs_pocket.pdb",
            ref_ligand="examples/10gs/10gs_ligand.sdf",
            query_ligand="CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            output_dir=tmpdir,
            num_confs=50,  # Small for speed
            verbose=True
        )

        # Verify results structure
        assert "output_file" in results
        assert "best_score" in results
        assert "runtime" in results
        assert "num_representatives" in results
        assert "mcs_size" in results

        # Verify output file exists
        assert os.path.exists(results["output_file"])

        print(f"\n✓ Basic usage test passed")
        print(f"  Best score: {results['best_score']:.2f} kcal/mol")
        print(f"  Runtime: {results['runtime']:.2f}s")
        print(f"  MCS size: {results['mcs_size']} atoms")


def test_with_optimization():
    """Test with gradient optimization enabled"""
    print("\n" + "="*60)
    print("TEST 2: With Optimization")
    print("="*60)

    from lig_align import run_pipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        results = run_pipeline(
            protein_pdb="examples/10gs/10gs_pocket.pdb",
            ref_ligand="examples/10gs/10gs_ligand.sdf",
            query_ligand="CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            output_dir=tmpdir,
            num_confs=50,
            optimize=True,
            optimizer="adam",
            opt_steps=50,
            verbose=True
        )

        assert os.path.exists(results["output_file"])
        assert "predicted_poses.sdf" in results["output_file"]

        print(f"\n✓ Optimization test passed")
        print(f"  Optimized best score: {results['best_score']:.2f} kcal/mol")
        print(f"  Saved {results['num_poses']} optimized poses")


def test_mcs_modes():
    """Test different MCS modes"""
    print("\n" + "="*60)
    print("TEST 3: MCS Modes")
    print("="*60)

    from lig_align import run_pipeline

    modes = ["single", "multi", "cross"]

    for mode in modes:
        print(f"\n--- Testing MCS mode: {mode} ---")
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_pipeline(
                protein_pdb="examples/10gs/10gs_pocket.pdb",
                ref_ligand="examples/10gs/10gs_ligand.sdf",
                query_ligand="CC(C)Cc1ccc(cc1)C(C)C(=O)O",
                output_dir=tmpdir,
                num_confs=50,
                mcs_mode=mode,
                verbose=False
            )

            assert os.path.exists(results["output_file"])
            print(f"  ✓ Mode '{mode}': {results['mcs_positions']} position(s), score={results['best_score']:.2f}")

    print(f"\n✓ All MCS modes test passed")


def test_scoring_functions():
    """Test different scoring functions"""
    print("\n" + "="*60)
    print("TEST 4: Scoring Functions")
    print("="*60)

    from lig_align import run_pipeline

    scoring_functions = ["vina", "vina_lp", "vinardo"]
    scores = {}

    for sf in scoring_functions:
        print(f"\n--- Testing scoring function: {sf} ---")
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_pipeline(
                protein_pdb="examples/10gs/10gs_pocket.pdb",
                ref_ligand="examples/10gs/10gs_ligand.sdf",
                query_ligand="CC(C)Cc1ccc(cc1)C(C)C(=O)O",
                output_dir=tmpdir,
                num_confs=50,
                weight_preset=sf,
                verbose=False
            )

            scores[sf] = results['best_score']
            print(f"  ✓ {sf}: {results['best_score']:.2f} kcal/mol")

    print(f"\n✓ All scoring functions test passed")


def test_all_poses_output():
    """Test that all poses are saved sorted by energy"""
    print("\n" + "="*60)
    print("TEST 5: All Poses Output (sorted by energy)")
    print("="*60)

    from lig_align import run_pipeline
    from rdkit import Chem

    with tempfile.TemporaryDirectory() as tmpdir:
        results = run_pipeline(
            protein_pdb="examples/10gs/10gs_pocket.pdb",
            ref_ligand="examples/10gs/10gs_ligand.sdf",
            query_ligand="CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            output_dir=tmpdir,
            num_confs=50,
            verbose=False
        )

        suppl = Chem.SDMolSupplier(results["output_file"])
        mols = [mol for mol in suppl if mol is not None]
        assert len(mols) == results['num_representatives']

        scores = [float(mol.GetProp("Vina_Score")) for mol in mols]
        assert scores == sorted(scores), "Poses must be sorted by energy (ascending)"

        print(f"  ✓ Saved {len(mols)} poses, all sorted by energy")

    print(f"\n✓ All poses output test passed")


def test_batch_processing():
    """Test batch processing multiple molecules"""
    print("\n" + "="*60)
    print("TEST 6: Batch Processing")
    print("="*60)

    from lig_align import run_pipeline

    molecules = {
        "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    }

    batch_results = []

    for name, smiles in molecules.items():
        print(f"\n--- Processing {name} ---")
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_pipeline(
                protein_pdb="examples/10gs/10gs_pocket.pdb",
                ref_ligand="examples/10gs/10gs_ligand.sdf",
                query_ligand=smiles,
                output_dir=tmpdir,
                num_confs=50,
                verbose=False
            )

            batch_results.append({
                "name": name,
                "score": results['best_score'],
                "mcs_size": results['mcs_size']
            })
            print(f"  ✓ {name}: score={results['best_score']:.2f}, MCS={results['mcs_size']} atoms")

    assert len(batch_results) == 2
    print(f"\n✓ Batch processing test passed ({len(batch_results)} molecules)")


def test_all_parameters():
    """Test with all parameters specified"""
    print("\n" + "="*60)
    print("TEST 7: All Parameters")
    print("="*60)

    from lig_align import run_pipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        results = run_pipeline(
            # Required
            protein_pdb="examples/10gs/10gs_pocket.pdb",
            ref_ligand="examples/10gs/10gs_ligand.sdf",
            query_ligand="CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            output_dir=tmpdir,
            # Conformer generation
            num_confs=50,
            rmsd_threshold=1.0,
            # MCS options
            mcs_mode="single",
            min_fragment_size=5,
            max_fragments=3,
            # Force field
            mmff_optimize=True,
            # Optimization
            optimize=True,
            optimizer="adam",
            opt_steps=50,
            opt_lr=0.05,
            opt_batch_size=8,
            freeze_mcs=True,
            # Scoring
            weight_preset="vina",
            torsion_penalty=False,
            # Output
            # System
            device=None,
            verbose=False
        )

        assert os.path.exists(results["output_file"])
        print(f"  ✓ All parameters accepted and processed")
        print(f"  Best score: {results['best_score']:.2f} kcal/mol")

    print(f"\n✓ All parameters test passed")


if __name__ == "__main__":
    test_basic_usage()
    test_with_optimization()
    test_mcs_modes()
    test_scoring_functions()
    test_all_poses_output()
    test_batch_processing()
    test_all_parameters()

    print("\n" + "="*60)
    print("ALL API TESTS PASSED ✓")
    print("="*60)
    print("\nThe run_pipeline() API is ready for use in Jupyter notebooks!")
    print("See examples/notebook_api_examples.ipynb for usage examples.")
