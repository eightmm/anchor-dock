"""Command-line interface for AnchorDock."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

from .batch import dock_batch
from .covalent import dock_covalent
from .free import dock_free
from .reference import dock_reference


def _optimization_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--optimizer", choices=("adam", "adamw", "lbfgs"), default="adam")
    parser.add_argument("--opt-steps", type=int, default=100)
    parser.add_argument("--opt-lr", type=float, default=0.05)
    parser.add_argument("--opt-batch-size", type=int, default=128)
    parser.add_argument(
        "--scorer",
        "--weight-preset",
        dest="scorer",
        choices=("vina", "vinardo", "softdock", "vina_lp"),
        default="vina",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anchor-dock")
    subcommands = parser.add_subparsers(dest="command", required=True)

    reference = subcommands.add_parser("reference", help="reference-ligand MCS docking")
    reference.add_argument("-p", "--protein", required=True)
    reference.add_argument("-r", "--reference", required=True)
    reference.add_argument("-q", "--query", required=True)
    reference.add_argument("-o", "--output", default="output_predictions")
    reference.add_argument("-n", "--num-confs", type=int, default=1000)
    reference.add_argument("--rmsd-threshold", type=float, default=1.0)
    reference.add_argument("--mcs-mode", choices=("auto", "single", "multi", "cross"), default="auto")
    reference.add_argument("--min-fragment-size", type=int, default=5)
    reference.add_argument("--max-fragments", type=int, default=3)
    reference.add_argument("--max-mappings", type=int, default=64)
    reference.add_argument("--match-chirality", action="store_true")
    reference.add_argument("--no-relax", "--no-mmff", dest="no_relax", action="store_true")
    reference.add_argument("--optimize", action="store_true")
    reference.add_argument("--free-anchor", "--free-mcs", dest="free_anchor", action="store_true")
    reference.add_argument("--top-k", type=int, default=None)
    _optimization_arguments(reference)

    covalent = subcommands.add_parser("covalent", help="residue-warhead covalent docking")
    covalent.add_argument("-p", "--protein", required=True)
    covalent.add_argument("-q", "--query", required=True)
    covalent.add_argument("-r", "--reactive-residue", default=None)
    covalent.add_argument("-o", "--output", default="output_predictions")
    covalent.add_argument("-n", "--num-confs", type=int, default=1000)
    covalent.add_argument("--rmsd-threshold", type=float, default=1.0)
    covalent.add_argument("--pocket-cutoff", type=float, default=12.0)
    covalent.add_argument("--rotation-scan-step", type=int, default=30)
    covalent.add_argument("--rotation-top-k", type=int, default=50)
    covalent.add_argument("--warhead-index", type=int, default=0)
    covalent.add_argument("--strict-compatibility", action="store_true")
    covalent_optimization = covalent.add_mutually_exclusive_group()
    covalent_optimization.add_argument("--optimize", dest="optimize", action="store_true")
    covalent_optimization.add_argument("--no-optimize", dest="optimize", action="store_false")
    covalent.set_defaults(optimize=None)
    covalent.add_argument("--top-k", type=int, default=None)
    _optimization_arguments(covalent)

    free = subcommands.add_parser("free", help="unanchored multistart local docking")
    free.add_argument("-p", "--protein", required=True)
    free.add_argument("-q", "--query", required=True)
    free.add_argument("-o", "--output", default="anchor_dock_free")
    free.add_argument("--center", type=float, nargs=3, default=None)
    free.add_argument("--box-size", type=float, nargs=3, default=(20.0, 20.0, 20.0))
    free.add_argument("-n", "--num-confs", type=int, default=64)
    free.add_argument("--num-starts", type=int, default=128)
    free.add_argument("--rmsd-threshold", type=float, default=1.0)
    free.add_argument("--top-k", type=int, default=20)
    _optimization_arguments(free)
    free.set_defaults(scorer="softdock", opt_steps=150)

    batch = subcommands.add_parser("batch", help="homogeneous or mixed-mode batch execution")
    batch.add_argument("input")
    batch.add_argument("-m", "--mode", choices=("reference", "covalent", "free"), default=None)
    batch.add_argument("-p", "--protein", default=None)
    batch.add_argument("-r", "--reference", default=None)
    batch.add_argument("--reactive-residue", default=None)
    batch.add_argument("-o", "--output", default="anchor_dock_batch")
    batch.add_argument("--on-error", choices=("record", "raise", "skip"), default="record")
    batch.add_argument("--resume", action="store_true")
    batch.add_argument(
        "--scorer",
        "--weight-preset",
        dest="scorer",
        choices=("vina", "vinardo", "softdock", "vina_lp"),
        default=None,
    )
    batch.add_argument("--device", choices=("cpu", "cuda"), default=None)
    batch.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    legacy_flags = sorted({flag for flag in ("--weight-preset", "--no-mmff", "--free-mcs") if flag in raw_argv})
    if legacy_flags:
        warnings.warn(
            f"deprecated 0.2 CLI option(s): {', '.join(legacy_flags)}",
            FutureWarning,
            stacklevel=2,
        )
    if getattr(args, "scorer", None) == "vina_lp":
        parser.error(
            "the unvalidated 0.2 'vina_lp' preset was removed; use vina, "
            "vinardo, or an explicitly configured custom scorer"
        )
    exit_code = 0
    if args.command == "reference":
        result = dock_reference(
            args.protein,
            args.reference,
            args.query,
            args.output,
            num_confs=args.num_confs,
            rmsd_threshold=args.rmsd_threshold,
            mcs_mode=args.mcs_mode,
            min_fragment_size=args.min_fragment_size,
            max_fragments=args.max_fragments,
            max_mappings=args.max_mappings,
            match_chirality=args.match_chirality,
            relax=not args.no_relax,
            optimize=args.optimize,
            optimizer=args.optimizer,
            opt_steps=args.opt_steps,
            opt_lr=args.opt_lr,
            opt_batch_size=args.opt_batch_size,
            freeze_anchor=not args.free_anchor,
            scorer=args.scorer,
            top_k=args.top_k,
            random_seed=args.seed,
            device=args.device,
            verbose=not args.quiet,
        )
    elif args.command == "covalent":
        optimize = args.optimize
        if optimize is None:
            optimize = False
            warnings.warn(
                "omitting covalent --optimize currently preserves the 0.2 default False; "
                "pass --optimize or --no-optimize explicitly",
                FutureWarning,
                stacklevel=2,
            )
        result = dock_covalent(
            args.protein,
            args.query,
            args.reactive_residue,
            args.output,
            pocket_cutoff=args.pocket_cutoff,
            num_confs=args.num_confs,
            rmsd_threshold=args.rmsd_threshold,
            rotation_scan_step=args.rotation_scan_step,
            rotation_top_k=args.rotation_top_k,
            optimize=optimize,
            optimizer=args.optimizer,
            opt_steps=args.opt_steps,
            opt_lr=args.opt_lr,
            opt_batch_size=args.opt_batch_size,
            scorer=args.scorer,
            top_k=args.top_k,
            warhead_index=args.warhead_index,
            strict_compatibility=args.strict_compatibility,
            random_seed=args.seed,
            device=args.device,
            verbose=not args.quiet,
        )
    elif args.command == "free":
        result = dock_free(
            args.protein,
            args.query,
            args.output,
            center=args.center,
            box_size=args.box_size,
            num_confs=args.num_confs,
            num_starts=args.num_starts,
            rmsd_threshold=args.rmsd_threshold,
            optimizer=args.optimizer,
            opt_steps=args.opt_steps,
            opt_lr=args.opt_lr,
            opt_batch_size=args.opt_batch_size,
            scorer=args.scorer,
            top_k=args.top_k,
            random_seed=args.seed,
            device=args.device,
            verbose=not args.quiet,
        )
    else:
        defaults = {"verbose": not args.quiet}
        if args.scorer is not None:
            defaults["scorer"] = args.scorer
        if args.device is not None:
            defaults["device"] = args.device
        result = dock_batch(
            args.input,
            mode=args.mode,
            protein_pdb=args.protein,
            reference_ligand=args.reference,
            reactive_residue=args.reactive_residue,
            output_dir=args.output,
            on_error=args.on_error,
            resume=args.resume,
            **defaults,
        )
        if any(item.get("success") is not True for item in result):
            exit_code = 1
        try:
            summary = json.loads((Path(args.output) / "summary.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            summary = None
        summary_errors = summary.get("errors", 0) if isinstance(summary, dict) else 0
        if isinstance(summary_errors, int | float) and summary_errors > 0:
            exit_code = 1
    print(json.dumps(result, indent=2, default=str))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
