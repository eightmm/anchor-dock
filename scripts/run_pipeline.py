import argparse

from anchor_dock.reference import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="LigAlign: High-speed 3D Ligand Pose Prediction Pipeline")
    parser.add_argument("-p", "--protein", required=True, help="Path to the protein pocket PDB file")
    parser.add_argument("-r", "--ref_ligand", required=True, help="Path to the reference ligand SDF file")
    parser.add_argument("-q", "--query_ligand", required=True, help="SMILES string or path to an SDF file of the query ligand")
    parser.add_argument("-o", "--out_dir", default="output_predictions", help="Directory to save the resulting SDF file (default: output_predictions)")
    parser.add_argument("-n", "--num_confs", type=int, default=1000, help="Number of conformers to generate (default: 1000)")
    parser.add_argument("--rmsd_threshold", type=float, default=1.0, help="RMSD threshold (Angstrom) for clustering (default: 1.0)")
    parser.add_argument("--no_mmff", action="store_true", help="Disable MMFF94 force field optimization for query appendages")
    parser.add_argument("--optimize", action="store_true", help="Enable Gradient-based Torsion Optimization on ALL cluster representatives")
    parser.add_argument("--opt_batch_size", type=int, default=128, help="Batch size for optimization (default: 128)")
    parser.add_argument("--opt_steps", type=int, default=100, help="Number of optimization steps (default: 100)")
    parser.add_argument("--opt_lr", type=float, default=0.05, help="Learning rate for optimization (default: 0.05)")
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw", "lbfgs"], default="adam", help="Optimizer for torsion optimization (default: adam)")
    parser.add_argument("--free_mcs", action="store_true", help="During optimization, let the MCS also optimize instead of acting as a rigid anchor")
    parser.set_defaults(torsion_penalty=True)
    torsion_group = parser.add_mutually_exclusive_group()
    torsion_group.add_argument("--torsion_penalty", dest="torsion_penalty", action="store_true",
                               help="Include the standard AutoDock Vina torsional entropy penalty (default)")
    torsion_group.add_argument("--no_torsion_penalty", dest="torsion_penalty", action="store_false",
                               help="Disable the torsional entropy penalty and report interaction-only scores")
    parser.add_argument("--weight_preset", type=str, choices=["vina", "vina_lp", "vinardo"], default="vina", help="Preset dictionary for Vina functional weights")

    # MCS Mode Options
    parser.add_argument("--mcs_mode", type=str, choices=["auto", "single", "multi", "cross"], default="auto",
                        help="MCS alignment mode (default: auto)")
    parser.add_argument("--min_fragment_size", type=int, default=5,
                        help="Minimum atoms per fragment for cross-matching mode (default: 5)")
    parser.add_argument("--max_fragments", type=int, default=3,
                        help="Maximum fragments to find for cross-matching mode (default: 3)")

    args = parser.parse_args()

    run_pipeline(
        protein_pdb=args.protein,
        ref_ligand=args.ref_ligand,
        query_ligand=args.query_ligand,
        output_dir=args.out_dir,
        num_confs=args.num_confs,
        rmsd_threshold=args.rmsd_threshold,
        mcs_mode=args.mcs_mode,
        min_fragment_size=args.min_fragment_size,
        max_fragments=args.max_fragments,
        mmff_optimize=not args.no_mmff,
        optimize=args.optimize,
        optimizer=args.optimizer,
        opt_steps=args.opt_steps,
        opt_lr=args.opt_lr,
        opt_batch_size=args.opt_batch_size,
        freeze_mcs=not args.free_mcs,
        weight_preset=args.weight_preset,
        torsion_penalty=args.torsion_penalty,
    )


if __name__ == "__main__":
    main()
