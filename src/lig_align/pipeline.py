"""
High-level pipeline API for easy use in Jupyter notebooks and scripts.

This module provides a single function that wraps the entire pipeline
with all CLI options available as parameters.
"""
import os
import time
import torch
from typing import Optional, List, Tuple, Literal
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit.Geometry import Point3D

from .aligner import LigandAligner
from .scoring import compute_intramolecular_mask
from .scoring.vina_scoring import precompute_interaction_matrices
from .io import load_pocket_bundle, process_query_ligand
from .molecular.mcs import find_mcs_with_positions, auto_select_mcs_mapping
from .molecular.relax import relax_pose_with_fixed_core


def _resolve_mcs_mappings(
    mcs_mode: str,
    ref_mol: Chem.Mol,
    query_mol: Chem.Mol,
    aligner: LigandAligner,
    min_fragment_size: int,
    max_fragments: int,
    verbose: bool,
) -> Tuple[str, List[List[Tuple[int, int]]]]:
    """Resolve MCS mode and return all mappings to try."""
    if mcs_mode == "auto":
        auto_choice = auto_select_mcs_mapping(
            ref_mol, query_mol, min_atoms=3,
            min_fragment_size=min_fragment_size, max_fragments=max_fragments,
        )
        resolved_mode = auto_choice["mode"]
        all_mappings = auto_choice["mappings"]
        if verbose:
            print(f"MCS Mode: auto -> {resolved_mode}")
            print(f"  Reason: {auto_choice['reason']}")
    elif mcs_mode == "single":
        if verbose:
            print("MCS Mode: single")
        mapping = aligner.step2_find_mcs(ref_mol, query_mol, return_all_positions=False)
        resolved_mode = "single"
        all_mappings = [mapping]
    elif mcs_mode == "multi":
        if verbose:
            print("MCS Mode: multi")
        all_mappings = find_mcs_with_positions(ref_mol, query_mol, return_all=True, min_atoms=3)
        resolved_mode = "multi"
        if verbose:
            print(f"Found {len(all_mappings)} possible MCS alignment positions")
    elif mcs_mode == "cross":
        if verbose:
            print("MCS Mode: cross")
        all_mappings = find_mcs_with_positions(
            ref_mol, query_mol, cross_match=True,
            min_fragment_size=min_fragment_size, max_fragments=max_fragments,
        )
        resolved_mode = "cross"
        if verbose:
            print(f"Found {len(all_mappings)} cross-matching combinations")
    else:
        raise ValueError(f"Invalid mcs_mode: {mcs_mode}")

    if not all_mappings:
        raise ValueError("No MCS found between reference and query")

    return resolved_mode, all_mappings


def _generate_and_score_for_mapping(
    mapping: List[Tuple[int, int]],
    position_idx: int,
    query_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    aligner: LigandAligner,
    pocket_coords: torch.Tensor,
    pocket_features: dict,
    num_confs: int,
    rmsd_threshold: float,
    mmff_optimize: bool,
    weight_preset: str,
    num_rotatable_bonds: Optional[int],
    verbose: bool,
) -> Tuple[Chem.Mol, List[int], torch.Tensor, torch.Tensor]:
    """Generate conformers, relax, and score for a single MCS mapping.

    Returns:
        (query_mol_with_confs, rep_cids, aligned_coords, scores)
    """
    ref_conf = ref_mol.GetConformer()

    # Build coordinate constraints from this mapping
    coordMap = {}
    for ref_idx, query_idx in mapping:
        pos = ref_conf.GetAtomPosition(ref_idx)
        coordMap[query_idx] = Point3D(pos.x, pos.y, pos.z)

    # Generate conformers
    query_mol_copy = Chem.RWMol(query_mol)
    query_mol_copy, rep_cids = aligner.step1_generate_conformers(
        query_mol_copy, num_confs=num_confs, rmsd_threshold=rmsd_threshold, coordMap=coordMap
    )

    if len(rep_cids) == 0:
        if verbose:
            print(f"  Position {position_idx + 1}: conformer generation failed, skipping")
        return query_mol_copy, [], torch.empty(0), torch.empty(0)

    # Coordinate surgery + MMFF relaxation
    num_atoms = query_mol_copy.GetNumAtoms()
    aligned_coords = torch.zeros((len(rep_cids), num_atoms, 3))
    mcs_query_indices = {query_idx for _, query_idx in mapping}

    # Precompute MMFF properties once
    mmff_props = None
    if mmff_optimize:
        try:
            query_mol_copy.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(query_mol_copy)
        except Exception:
            pass
        mmff_props = AllChem.MMFFGetMoleculeProperties(query_mol_copy)

    for j, cid in enumerate(rep_cids):
        conf = query_mol_copy.GetConformer(cid)
        for ref_idx, query_idx in mapping:
            pos = ref_conf.GetAtomPosition(ref_idx)
            conf.SetAtomPosition(query_idx, Point3D(pos.x, pos.y, pos.z))

        if mmff_optimize:
            relax_pose_with_fixed_core(query_mol_copy, cid, mcs_query_indices, max_iters=500, mmff_props=mmff_props)

        aligned_coords[j] = torch.tensor(conf.GetPositions(), dtype=torch.float32)

    aligned_coords = aligned_coords.to(aligner.device)

    # Score
    query_features = aligner.compute_vina_features(query_mol_copy)
    intra_mask = compute_intramolecular_mask(query_mol_copy, aligner.device)
    precomputed = precompute_interaction_matrices(query_features, pocket_features, aligner.device)
    scores = aligner.step4_vina_scoring(
        aligned_coords, pocket_coords, query_features, pocket_features,
        num_rotatable_bonds, weight_preset,
        intramolecular_mask=intra_mask, precomputed_matrices=precomputed,
    )

    if verbose:
        best = scores.min().item() if len(scores) > 0 else float('inf')
        print(f"  Position {position_idx + 1}: {len(rep_cids)} representatives, best score = {best:.3f} kcal/mol")

    return query_mol_copy, rep_cids, aligned_coords, scores


def run_pipeline(
    protein_pdb: str,
    ref_ligand: str,
    query_ligand: str,
    output_dir: str = "output_predictions",
    # Conformer generation
    num_confs: int = 1000,
    rmsd_threshold: float = 1.0,
    # MCS options
    mcs_mode: Literal["auto", "single", "multi", "cross"] = "auto",
    min_fragment_size: int = 5,
    max_fragments: int = 3,
    # Force field
    mmff_optimize: bool = True,
    # Optimization
    optimize: bool = False,
    optimizer: Literal["adam", "adamw", "lbfgs"] = "adam",
    opt_steps: int = 100,
    opt_lr: float = 0.05,
    opt_batch_size: int = 128,
    freeze_mcs: bool = True,
    # Scoring
    weight_preset: Literal["vina", "vina_lp", "vinardo"] = "vina",
    torsion_penalty: bool = True,
    # Device
    device: Optional[str] = None,
    verbose: bool = True
) -> dict:
    """
    Run the complete LigAlign pipeline with all options.

    For multi/cross MCS modes, runs conformer generation and scoring
    across ALL MCS positions and merges the results to find the best
    poses from the combined pool.

    Args:
        protein_pdb: Path to protein pocket PDB file
        ref_ligand: Path to reference ligand SDF file
        query_ligand: SMILES string or path to SDF file
        output_dir: Directory to save results (default: "output_predictions")
        num_confs: Number of conformers to generate per MCS position (default: 1000)
        rmsd_threshold: RMSD threshold for clustering in Angstrom (default: 1.0)
        mcs_mode: MCS alignment mode (default: "auto")
        min_fragment_size: Min atoms per fragment for cross mode (default: 5)
        max_fragments: Max fragments to find for cross mode (default: 3)
        mmff_optimize: Apply MMFF94 relaxation after alignment (default: True)
        optimize: Enable gradient-based torsion optimization (default: False)
        optimizer: Optimizer type: "adam", "adamw", "lbfgs" (default: "adam")
        opt_steps: Optimization steps (default: 100)
        opt_lr: Learning rate (default: 0.05)
        opt_batch_size: Batch size for optimization (default: 128)
        freeze_mcs: Keep MCS atoms fixed during optimization (default: True)
        weight_preset: Vina scoring weights (default: "vina")
        torsion_penalty: Apply torsional entropy penalty (default: True)
        device: Device to use: "cuda", "cpu", or None for auto (default: None)
        verbose: Print progress messages (default: True)

    Returns:
        dict with keys: output_file, num_poses, best_score, runtime,
        num_conformers, num_representatives, mcs_size, mcs_positions,
        canonical_smiles, device
    """
    t0 = time.time()

    if not verbose:
        RDLogger.DisableLog('rdApp.warning')

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    aligner = LigandAligner(device=device)

    # Load inputs
    if verbose:
        print(f"Loading reference ligand from {ref_ligand}...")
    ref_suppl = Chem.SDMolSupplier(ref_ligand)
    ref_mol = ref_suppl[0]
    if ref_mol is None:
        raise ValueError(f"Failed to load reference ligand from {ref_ligand}")

    if verbose:
        print(f"Loading protein pocket from {protein_pdb}...")
    pocket_bundle = load_pocket_bundle(protein_pdb, device, aligner.compute_vina_features)

    query_mol, canonical_smiles = process_query_ligand(query_ligand)
    if verbose:
        print(f"\nProcessing Query Ligand: {canonical_smiles}")

    # 1. MCS Search - get ALL mappings
    requested_mcs_mode = mcs_mode
    resolved_mode, all_mappings = _resolve_mcs_mappings(
        mcs_mode, ref_mol, query_mol, aligner,
        min_fragment_size, max_fragments, verbose,
    )
    num_mcs_positions = len(all_mappings)

    # Compute rotatable bonds once
    num_rotatable_bonds = None
    if torsion_penalty:
        from rdkit.Chem import rdMolDescriptors
        # Need a mol with Hs removed for consistent counting
        query_no_h = Chem.RemoveHs(Chem.AddHs(query_mol))
        num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(query_no_h)

    pocket_coords = pocket_bundle.coords
    pocket_features = pocket_bundle.features

    # 2-6. Generate, relax, and score for each MCS position
    if verbose and num_mcs_positions > 1:
        print(f"\nRunning ensemble over {num_mcs_positions} MCS positions...")

    all_query_mols = []
    all_rep_cids = []
    all_aligned_coords = []
    all_scores = []
    all_position_mappings = []

    for pos_idx, mapping in enumerate(all_mappings):
        query_mol_pos, rep_cids, aligned_coords, scores = _generate_and_score_for_mapping(
            mapping=mapping,
            position_idx=pos_idx,
            query_mol=query_mol,
            ref_mol=ref_mol,
            aligner=aligner,
            pocket_coords=pocket_coords,
            pocket_features=pocket_features,
            num_confs=num_confs,
            rmsd_threshold=rmsd_threshold,
            mmff_optimize=mmff_optimize,
            weight_preset=weight_preset,
            num_rotatable_bonds=num_rotatable_bonds,
            verbose=verbose,
        )
        if len(rep_cids) > 0:
            all_query_mols.append(query_mol_pos)
            all_rep_cids.append(rep_cids)
            all_aligned_coords.append(aligned_coords)
            all_scores.append(scores)
            all_position_mappings.append(mapping)

    if not all_scores:
        raise RuntimeError("Failed to generate conformers for any MCS position")

    # Save initial scores before optimization
    all_initial_scores = [s.clone() for s in all_scores]

    # 7. Optional Gradient Optimization (per-position with correct MCS mapping)
    if optimize:
        total_poses = sum(len(c) for c in all_rep_cids)
        if verbose:
            print(f"\n--- Running Gradient-Based Torsion Optimization on {total_poses} Cluster Representatives ---")

        for i in range(len(all_query_mols)):
            mapping = all_position_mappings[i]
            query_indices = [m[1] for m in mapping]
            query_features = aligner.compute_vina_features(all_query_mols[i])

            if verbose and num_mcs_positions > 1:
                print(f"  Position {i + 1}: optimizing {len(all_rep_cids[i])} poses...")

            all_aligned_coords[i] = aligner.step6_refine_pose(
                mol=all_query_mols[i],
                ref_indices=query_indices,
                init_coords=all_aligned_coords[i],
                pocket_coords=pocket_coords,
                query_features=query_features,
                pocket_features=pocket_features,
                num_steps=opt_steps,
                lr=opt_lr,
                freeze_mcs=freeze_mcs,
                num_rotatable_bonds=num_rotatable_bonds,
                weight_preset=weight_preset,
                batch_size=opt_batch_size,
                optimizer=optimizer,
            )

            intra_mask = compute_intramolecular_mask(all_query_mols[i], aligner.device)
            precomputed = precompute_interaction_matrices(query_features, pocket_features, aligner.device)
            all_scores[i] = aligner.step4_vina_scoring(
                all_aligned_coords[i], pocket_coords, query_features, pocket_features,
                num_rotatable_bonds, weight_preset,
                intramolecular_mask=intra_mask, precomputed_matrices=precomputed,
            )

        if verbose:
            all_init = torch.cat(all_initial_scores)
            all_opt = torch.cat(all_scores)
            diffs = all_opt - all_init
            best_idx = torch.argmin(all_opt).item()
            print(f"Optimization complete!")
            print(f"  Best score: {all_opt[best_idx]:.3f} kcal/mol (delta = {diffs[best_idx]:.3f})")
            print(f"  Average improvement: {diffs.mean():.3f} kcal/mol")

    # Merge all positions into a single pool
    merged_coords = torch.cat(all_aligned_coords)
    merged_scores = torch.cat(all_scores)
    merged_initial_scores = torch.cat(all_initial_scores)
    merged_rep_cids = []
    merged_position_labels = []
    for i, cids in enumerate(all_rep_cids):
        merged_rep_cids.extend(cids)
        merged_position_labels.extend([i + 1] * len(cids))

    total_representatives = len(merged_rep_cids)

    if verbose and num_mcs_positions > 1:
        per_pos = [f"pos{i+1}={len(c)}" for i, c in enumerate(all_rep_cids)]
        print(f"\nMerged {total_representatives} poses from {num_mcs_positions} positions ({', '.join(per_pos)})")

    # Use best position's mapping for molecule-level SDF properties
    best_pos_idx = min(range(len(all_scores)), key=lambda i: all_scores[i].min().item())
    best_mapping = all_position_mappings[best_pos_idx]
    query_mol_out = all_query_mols[best_pos_idx]

    num_mcs_atoms = len(best_mapping)
    ref_heavy = ref_mol.GetNumHeavyAtoms()
    query_heavy = query_mol_out.GetNumHeavyAtoms()
    ref_cov = (num_mcs_atoms / ref_heavy * 100) if ref_heavy else 0
    query_cov = (num_mcs_atoms / query_heavy * 100) if query_heavy else 0

    query_mol_out.SetProp("MCS_Num_Atoms", str(num_mcs_atoms))
    query_mol_out.SetProp("MCS_Ref_Coverage", f"{ref_cov:.1f}%")
    query_mol_out.SetProp("MCS_Query_Coverage", f"{query_cov:.1f}%")
    query_mol_out.SetProp("LigAlign_MCS_Mode", resolved_mode)
    query_mol_out.SetProp("LigAlign_MCS_Mode_Requested", requested_mcs_mode)
    query_mol_out.SetProp("LigAlign_Num_Confs_Generated", str(num_confs))
    query_mol_out.SetProp("LigAlign_MMFF_Requested", str(mmff_optimize))
    query_mol_out.SetProp("LigAlign_MMFF_Optimized", str(mmff_optimize))
    if num_mcs_positions > 1:
        query_mol_out.SetProp("LigAlign_MCS_Positions_Tried", str(num_mcs_positions))
    if optimize:
        query_mol_out.SetProp("LigAlign_Gradient_Optimized", "True")
        query_mol_out.SetProp("LigAlign_Optimized_Poses", str(total_representatives))

    # 8. Output — save all poses sorted by energy
    os.makedirs(output_dir, exist_ok=True)
    out_sdf = os.path.join(output_dir, "predicted_poses.sdf")
    aligner.step5_final_selection(
        query_mol_out, merged_rep_cids, merged_coords, merged_scores,
        initial_scores=merged_initial_scores,
        position_labels=merged_position_labels if num_mcs_positions > 1 else None,
        output_path=out_sdf,
    )
    num_saved = total_representatives

    runtime = time.time() - t0
    best_score = float(torch.min(merged_scores).item())

    if verbose:
        print(f"\n-> Prediction Completed successfully in {runtime:.2f}s!")
        print(f"-> Results saved to: {out_sdf}")

    return {
        "output_file": out_sdf,
        "num_poses": num_saved,
        "best_score": best_score,
        "runtime": runtime,
        "num_conformers": num_confs,
        "num_representatives": total_representatives,
        "mcs_size": num_mcs_atoms,
        "mcs_positions": num_mcs_positions,
        "best_position": best_pos_idx + 1,
        "canonical_smiles": canonical_smiles,
        "device": str(device),
    }


def run_batch(
    protein_pdb: str,
    ref_ligand: str,
    query_ligands: List[str],
    output_dir: str = "output_predictions",
    verbose: bool = True,
    **kwargs,
) -> List[dict]:
    """
    Run the pipeline on multiple query ligands, sharing pocket loading
    and aligner initialization across all queries.

    Args:
        protein_pdb: Path to protein pocket PDB file
        ref_ligand: Path to reference ligand SDF file
        query_ligands: List of SMILES strings or SDF file paths
        output_dir: Base directory for results (each query gets a subdirectory)
        verbose: Print progress messages (default: True)
        **kwargs: All other arguments passed to run_pipeline()
            (num_confs, rmsd_threshold, mcs_mode, optimize, etc.)

    Returns:
        List of result dicts (same format as run_pipeline), one per query.
        Failed queries return a dict with "error" key instead.

    Example:
        >>> from lig_align import run_batch
        >>> results = run_batch(
        ...     protein_pdb="pocket.pdb",
        ...     ref_ligand="reference.sdf",
        ...     query_ligands=["CCO", "c1ccccc1", "CC(=O)O"],
        ...     optimize=True,
        ... )
        >>> for r in results:
        ...     if "error" not in r:
        ...         print(f"{r['canonical_smiles']}: {r['best_score']:.3f} kcal/mol")
    """
    t0 = time.time()

    if not verbose:
        RDLogger.DisableLog('rdApp.warning')

    # Pre-warm pocket cache with a dummy call to load_pocket_bundle
    device_str = kwargs.get("device", None)
    if device_str is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    aligner = LigandAligner(device=device)

    if verbose:
        print(f"Pre-loading pocket from {protein_pdb}...")
    load_pocket_bundle(protein_pdb, device, aligner.compute_vina_features)

    if verbose:
        print(f"Processing {len(query_ligands)} queries...\n")

    results = []
    for i, query in enumerate(query_ligands):
        query_dir = os.path.join(output_dir, f"query_{i:04d}")
        if verbose:
            print(f"[{i + 1}/{len(query_ligands)}] {query}")

        try:
            result = run_pipeline(
                protein_pdb=protein_pdb,
                ref_ligand=ref_ligand,
                query_ligand=query,
                output_dir=query_dir,
                verbose=verbose,
                **kwargs,
            )
            results.append(result)
        except Exception as e:
            if verbose:
                print(f"  FAILED: {e}")
            results.append({"query": query, "error": str(e)})

    total_time = time.time() - t0
    successful = sum(1 for r in results if "error" not in r)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Batch complete: {successful}/{len(query_ligands)} succeeded in {total_time:.2f}s")
        if successful > 0:
            avg_time = total_time / len(query_ligands)
            scores = [r["best_score"] for r in results if "error" not in r]
            print(f"  Avg time per query: {avg_time:.2f}s")
            print(f"  Best score range: [{min(scores):.3f}, {max(scores):.3f}] kcal/mol")

    return results
