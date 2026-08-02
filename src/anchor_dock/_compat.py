"""One-release adapters from the 0.2 call surface to the 0.3 engine.

These adapters preserve invocation syntax only. They never route to the 0.2
scorer, atom typing, or output schema.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

_REFERENCE_POSITIONAL_PARAMETERS = (
    "protein_pdb",
    "ref_ligand",
    "query_ligand",
    "output_dir",
    "num_confs",
    "rmsd_threshold",
    "mcs_mode",
    "min_fragment_size",
    "max_fragments",
    "mmff_optimize",
    "optimize",
    "optimizer",
    "opt_steps",
    "opt_lr",
    "opt_batch_size",
    "freeze_mcs",
    "weight_preset",
    "torsion_penalty",
    "device",
    "verbose",
)
_COVALENT_POSITIONAL_PARAMETERS = (
    "protein_pdb",
    "query_ligand",
    "reactive_residue",
    "output_dir",
    "pocket_cutoff",
    "_cached_pocket",
    "num_confs",
    "rmsd_threshold",
    "rotation_scan_step",
    "rotation_top_k",
    "optimize",
    "optimizer",
    "opt_steps",
    "opt_lr",
    "opt_batch_size",
    "weight_preset",
    "torsion_penalty",
    "save_all_poses",
    "top_k",
    "device",
    "verbose",
    "warhead_index",
    "strict_compatibility",
)


def _bind_legacy_positional_options(
    args: tuple[Any, ...],
    raw_options: dict[str, Any],
    parameter_names: tuple[str, ...],
    changes: list[str],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if len(args) > len(parameter_names):
        raise TypeError(f"expected at most {len(parameter_names)} positional arguments, got {len(args)}")
    options = dict(raw_options)
    for index in range(4, len(args)):
        name = parameter_names[index]
        if name in options:
            raise TypeError(f"got multiple values for argument {name!r}")
        options[name] = args[index]
    if len(args) > 4:
        changes.append("positional options after output_dir are deprecated; pass them by keyword")
    return args[:4], options


def _warn(changes: list[str]) -> None:
    if changes:
        warnings.warn(
            "AnchorDock 0.2 compatibility adapter: " + "; ".join(changes),
            FutureWarning,
            stacklevel=3,
        )


def _rename(options: dict[str, Any], old: str, new: str, changes: list[str]) -> None:
    if old not in options:
        return
    if new in options:
        raise ValueError(f"cannot pass both {old!r} and {new!r}")
    options[new] = options.pop(old)
    changes.append(f"{old} is deprecated; use {new}")


def _reject_vina_lp(options: dict[str, Any]) -> None:
    if options.get("scorer") == "vina_lp":
        raise ValueError(
            "the unvalidated 0.2 'vina_lp' preset was removed; choose 'vina' or "
            "'vinardo', or provide an explicitly named custom ScoringConfig"
        )


def _reference_options(raw: dict[str, Any], changes: list[str]) -> dict[str, Any]:
    options = dict(raw)
    _rename(options, "ref_ligand", "reference_ligand", changes)
    _rename(options, "mmff_optimize", "relax", changes)
    _rename(options, "freeze_mcs", "freeze_anchor", changes)
    _rename(options, "weight_preset", "scorer", changes)
    _reject_vina_lp(options)
    if options.get("optimizer") == "adamw":
        changes.append("optimizer='adamw' is retained for 0.2 compatibility")
    return options


def _covalent_options(raw: dict[str, Any], changes: list[str]) -> dict[str, Any]:
    options = dict(raw)
    _rename(options, "weight_preset", "scorer", changes)
    if "_cached_pocket" in options:
        cached_pocket = options.pop("_cached_pocket")
        if cached_pocket is not None:
            raise ValueError(
                "_cached_pocket was an internal 0.2 hook; 0.3 caches receptor contexts "
                "automatically and dock_batch should be used for batches"
            )
        changes.append("_cached_pocket=None is ignored; receptor contexts are cached automatically")
    if "save_all_poses" in options:
        save_all = options.pop("save_all_poses")
        if save_all is False and options.get("top_k") is None:
            options["top_k"] = 3
        changes.append("save_all_poses is deprecated; use top_k")
    _reject_vina_lp(options)
    if options.get("optimizer") == "adamw":
        changes.append("optimizer='adamw' is retained for 0.2 compatibility")
    return options


def dock_reference(*args: Any, **kwargs: Any) -> dict[str, object]:
    """Call the 0.3 reference engine while translating unambiguous 0.2 names."""
    changes: list[str] = []
    canonical_args, raw_options = _bind_legacy_positional_options(
        args, kwargs, _REFERENCE_POSITIONAL_PARAMETERS, changes
    )
    options = _reference_options(raw_options, changes)
    _warn(changes)
    from .reference.pipeline import dock_reference as canonical

    return canonical(*canonical_args, **options)


def dock_covalent(*args: Any, **kwargs: Any) -> dict[str, object]:
    """Call the 0.3 covalent engine with the safe 0.2 optimization default."""
    changes: list[str] = []
    canonical_args, raw_options = _bind_legacy_positional_options(
        args, kwargs, _COVALENT_POSITIONAL_PARAMETERS, changes
    )
    options = _covalent_options(raw_options, changes)
    if "optimize" not in options:
        options["optimize"] = False
        changes.append("omitted optimize currently means False; pass it explicitly")
    _warn(changes)
    from .covalent.pipeline import dock_covalent as canonical

    return canonical(*canonical_args, **options)


def run_reference_pipeline(*args: Any, **kwargs: Any) -> dict[str, object]:
    changes = ["run_reference_pipeline/run_pipeline is deprecated; use dock_reference"]
    canonical_args, raw_options = _bind_legacy_positional_options(
        args, kwargs, _REFERENCE_POSITIONAL_PARAMETERS, changes
    )
    options = _reference_options(raw_options, changes)
    _warn(changes)
    from .reference.pipeline import dock_reference as canonical

    return canonical(*canonical_args, **options)


def run_covalent_pipeline(*args: Any, **kwargs: Any) -> dict[str, object]:
    changes = ["run_covalent_pipeline is deprecated; use dock_covalent"]
    canonical_args, raw_options = _bind_legacy_positional_options(
        args, kwargs, _COVALENT_POSITIONAL_PARAMETERS, changes
    )
    options = _covalent_options(raw_options, changes)
    if "optimize" not in options:
        options["optimize"] = False
    _warn(changes)
    from .covalent.pipeline import dock_covalent as canonical

    return canonical(*canonical_args, **options)


def dock_reference_batch(
    protein_pdb: str,
    ref_ligand: str,
    query_ligands: object,
    output_dir: str = "output_predictions",
    verbose: bool = True,
    **kwargs: Any,
) -> list[dict[str, object]]:
    changes = ["dock_reference_batch/run_batch is deprecated; use dock_batch"]
    options = _reference_options(kwargs, changes)
    if "reference_ligand" in options:
        raise ValueError("reference_ligand conflicts with the ref_ligand argument")
    on_error = options.pop("on_error", "record")
    resume = bool(options.pop("resume", False))
    _warn(changes)
    from .batch import dock_batch

    return dock_batch(
        query_ligands,
        mode="reference",
        protein_pdb=protein_pdb,
        reference_ligand=ref_ligand,
        output_dir=output_dir,
        on_error=on_error,
        resume=resume,
        verbose=verbose,
        **options,
    )


def dock_covalent_batch(
    protein_pdb: str,
    ligands: str | Iterable[str],
    reactive_residue: str | None = None,
    output_dir: str = "results",
    *,
    pocket_cutoff: float = 12.0,
    device: object = None,
    verbose: bool = True,
    **kwargs: Any,
) -> list[dict[str, object]]:
    changes = ["dock_covalent_batch/run_batch_docking is deprecated; use dock_batch"]
    options = _covalent_options(kwargs, changes)
    if "optimize" not in options:
        options["optimize"] = False
    on_error = options.pop("on_error", "record")
    resume = bool(options.pop("resume", False))
    _warn(changes)
    from .batch import dock_batch

    return dock_batch(
        ligands,
        mode="covalent",
        protein_pdb=protein_pdb,
        reactive_residue=reactive_residue,
        output_dir=output_dir,
        on_error=on_error,
        resume=resume,
        pocket_cutoff=pocket_cutoff,
        device=device,
        verbose=verbose,
        **options,
    )
