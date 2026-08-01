"""Command-line entry point for AnchorDock."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .covalent import dock_covalent
from .reference import dock_reference


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-p", "--protein", required=True, help="Protein pocket/full PDB")
    parser.add_argument("-q", "--query", required=True, help="Query SMILES or SDF")
    parser.add_argument("-o", "--output", default="output_predictions", help="Output directory")
    parser.add_argument("-n", "--num-confs", type=int, default=1000)
    parser.add_argument("--rmsd-threshold", type=float, default=1.0)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--optimizer", choices=("adam", "adamw", "lbfgs"), default="adam")
    parser.add_argument("--opt-steps", type=int, default=100)
    parser.add_argument("--opt-lr", type=float, default=0.05)
    parser.add_argument("--opt-batch-size", type=int, default=128)
    parser.add_argument("--weight-preset", choices=("vina", "vina_lp", "vinardo"), default="vina")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--quiet", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anchor-dock")
    subcommands = parser.add_subparsers(dest="mode", required=True)

    reference = subcommands.add_parser("reference", help="Reference-ligand MCS anchoring")
    _common(reference)
    reference.add_argument("-r", "--reference", required=True, help="Reference ligand SDF")
    reference.add_argument("--mcs-mode", choices=("auto", "single", "multi", "cross"), default="auto")
    reference.add_argument("--min-fragment-size", type=int, default=5)
    reference.add_argument("--max-fragments", type=int, default=3)
    reference.add_argument("--no-mmff", action="store_true")
    reference.add_argument("--free-mcs", action="store_true")

    covalent = subcommands.add_parser("covalent", help="Reactive residue-warhead anchoring")
    _common(covalent)
    covalent.add_argument("-r", "--reactive-residue", default=None, help="For example CYS145:A")
    covalent.add_argument("--pocket-cutoff", type=float, default=12.0)
    covalent.add_argument("--rotation-scan-step", type=int, default=30)
    covalent.add_argument("--rotation-top-k", type=int, default=50)
    covalent.add_argument("--warhead-index", type=int, default=0)
    covalent.add_argument("--strict-compatibility", action="store_true")
    covalent.add_argument("--top-k", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shared = dict(
        protein_pdb=args.protein,
        query_ligand=args.query,
        output_dir=args.output,
        num_confs=args.num_confs,
        rmsd_threshold=args.rmsd_threshold,
        optimize=args.optimize,
        optimizer=args.optimizer,
        opt_steps=args.opt_steps,
        opt_lr=args.opt_lr,
        opt_batch_size=args.opt_batch_size,
        weight_preset=args.weight_preset,
        device=args.device,
        verbose=not args.quiet,
    )
    if args.mode == "reference":
        result = dock_reference(
            ref_ligand=args.reference,
            mcs_mode=args.mcs_mode,
            min_fragment_size=args.min_fragment_size,
            max_fragments=args.max_fragments,
            mmff_optimize=not args.no_mmff,
            freeze_mcs=not args.free_mcs,
            **shared,
        )
    else:
        result = dock_covalent(
            reactive_residue=args.reactive_residue,
            pocket_cutoff=args.pocket_cutoff,
            rotation_scan_step=args.rotation_scan_step,
            rotation_top_k=args.rotation_top_k,
            warhead_index=args.warhead_index,
            strict_compatibility=args.strict_compatibility,
            top_k=args.top_k,
            **shared,
        )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
