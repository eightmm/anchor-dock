"""Refine an existing pose independently under named scoring functions."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import platform
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from numbers import Integral, Real
from pathlib import Path

import torch
from rdkit import Chem, rdBase

from ._version import __version__
from .core.engine import DockingEngine
from .core.io import ReceptorContext, file_content_fingerprint, load_receptor_context
from .core.optimization import OPTIMIZATION_CONTRACT, SE3PoseModel, optimize_pose_module
from .core.prepared import PreparedLigand
from .core.registry import create_scoring_function, scoring_function_info
from .core.scoring import resolve_scorer
from .core.topology import find_rotatable_bonds
from .preparation import (
    LigandPreparation,
    ReceptorPreparation,
    load_ligand_for_refinement,
    prepare_ligand,
    prepare_receptor,
)


@dataclass
class PoseRefinementResult:
    scorer: str
    mol: Chem.Mol
    coordinates: torch.Tensor
    scores: dict[str, dict]
    initial_scores: dict[str, dict]
    optimized_objective: str
    metadata: dict

    def as_dict(self, *, include_coordinates: bool = True) -> dict:
        value = dict(schema="anchor-dock-sf-refinement-v1", success=True, scorer=self.scorer, scores=self.scores, initial_scores=self.initial_scores,
                     optimized_objective=self.optimized_objective, metadata=self.metadata)
        if include_coordinates:
            value["coordinates_angstrom"] = self.coordinates.detach().cpu().tolist()
        return value


def _identity(value) -> str:
    if isinstance(value, (str, Path)):
        try:
            if Path(value).is_file():
                return file_content_fingerprint(value)
        except OSError:
            pass  # A long SMILES is not a filesystem path.
    if isinstance(value, LigandPreparation):
        return value.fingerprint
    if isinstance(value, PreparedLigand):
        return value.fingerprint
    if isinstance(value, Chem.Mol):
        value = Chem.MolToMolBlock(value)
    return "sha256:" + hashlib.sha256(str(value).encode()).hexdigest()


def _metric(value, units, *, role="reported_score", direction="lower") -> dict:
    number = float(torch.as_tensor(value).detach().cpu())
    if not math.isfinite(number):
        raise FloatingPointError("scorer returned a nonfinite metric")
    return dict(value=number, units=units, direction=direction, role=role)


def _configuration(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _configuration(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _configuration(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_configuration(v) for v in value]
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                    sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest(), external_state_required=True)
    return dict(type=f"{type(value).__module__}.{type(value).__qualname__}", external_state_required=True)


def _mol_with_coordinates(mol, coordinates):
    result = Chem.Mol(mol)
    result.RemoveAllConformers()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, point in enumerate(coordinates.detach().cpu().tolist()):
        conformer.SetAtomPosition(index, point)
    result.AddConformer(conformer)
    return result


def _write_result(result: PoseRefinementResult, output_dir):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    sdf, report = directory / "refined.sdf", directory / "result.json"
    if sdf.exists() or report.exists():
        raise FileExistsError(f"refinement output already exists in {directory}")
    molecule = _mol_with_coordinates(result.mol, result.coordinates)
    molecule.SetProp("AnchorDock_Version", __version__)
    molecule.SetProp("AnchorDock_Result_Schema", "anchor-dock-sf-refinement-v1")
    molecule.SetProp("AnchorDock_Scorer", result.scorer)
    molecule.SetProp("AnchorDock_Optimization_Objective", result.optimized_objective)
    molecule.SetProp("AnchorDock_Scorer_Fingerprint", str(result.metadata.get("scorer_fingerprint", "")))
    for name, metric in result.scores.items():
        molecule.SetProp("AnchorDock_" + name, str(metric["value"]))
        molecule.SetProp("AnchorDock_" + name + "_Units", metric["units"])
        molecule.SetProp("AnchorDock_" + name + "_Direction", metric["direction"])
    writer = Chem.SDWriter(str(sdf))
    try:
        writer.write(molecule)
    finally:
        writer.close()
    report.write_text(json.dumps(result.as_dict(), indent=2, allow_nan=False) + "\n")


def refine_pose(
    protein, ligand, *, scorer="vina", scorer_options: Mapping[str, object] | None = None,
    preparation: str = "auto", preparation_options: Mapping[str, object] | None = None,
    optimizer: str | None = None, num_steps: int = 100, learning_rate: float | None = None,
    optimizer_options: Mapping[str, object] | None = None, seed: int = 42,
    device: str | torch.device = "cpu", output_dir: str | Path | None = None,
) -> PoseRefinementResult:
    """Optimize one supplied pose; retain its graph/order and existing 3D pose.

    Torch-compatible models share SE(3)+torsion optimization; table-reference
    objectives require explicit pattern search. No raw scores are combined.
    AMBER topology contains the receptor, so ``protein=None`` is permitted.
    """
    if preparation not in {"auto", "inferred", "meeko"}:
        raise ValueError("preparation must be auto, inferred or meeko")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2**31)")
    if isinstance(num_steps, bool) or not isinstance(num_steps, Integral) or num_steps < 0:
        raise ValueError("num_steps must be a nonnegative integer")
    options, prep_options = dict(scorer_options or {}), dict(preparation_options or {})
    if set(prep_options) - {"ligand", "receptor"}:
        raise ValueError("preparation_options accepts ligand and receptor dictionaries")
    ligand_options, receptor_options = dict(prep_options.get("ligand", {})), dict(prep_options.get("receptor", {}))
    if "seed" in ligand_options:
        raise ValueError("set the refinement seed once, not inside ligand preparation options")
    if ligand_options.get("generate_coordinates_if_missing", False):
        raise ValueError("local refinement requires an existing positioned 3D pose; use guided docking to place a new conformer")
    if isinstance(ligand, (LigandPreparation, PreparedLigand)) and ligand_options:
        raise ValueError("ligand preparation options cannot alter an already prepared ligand")
    if isinstance(protein, (ReceptorPreparation, ReceptorContext)) and receptor_options:
        raise ValueError("receptor preparation options cannot alter an already prepared receptor")
    info = scoring_function_info(scorer) if isinstance(scorer, str) else None
    if info:
        if info.name in {"amber", "ad4", "adgpu"}:
            options.setdefault("device", str(device))
        model_factory = create_scoring_function(scorer, **options)
        selected_name, profile = info.name, info.preparation
    else:
        if options:
            raise ValueError("scorer_options require a registered scorer name")
        model_factory = resolve_scorer(scorer)
        selected_name = model_factory.name
        profile = getattr(model_factory, "preparation_requirement", "inferred")
    try:
        metadata = dict(version=__version__, input_ligand_sha256=_identity(ligand), seed=int(seed),
                        runtime=dict(python=platform.python_version(), torch=torch.__version__, rdkit=rdBase.rdkitVersion,
                                     requested_device=str(device), torch_threads=torch.get_num_threads(),
                                     deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
                        independent_local_refinement=True, registry_info=info.as_dict() if info else None)
        target_device = torch.device(device)
        amber = profile == "amber-topology"
        prepared_ligand = None
        if amber:
            if preparation != "auto" or prep_options:
                raise ValueError("AMBER uses its explicit topology/coordinates, not Meeko preparation")
            mol = load_ligand_for_refinement(ligand, seed=seed, add_hydrogens=False, generate_coordinates_if_missing=False)
            model_factory.validate_ligand(mol)
            expected_receptor = model_factory.receptor_coordinates(device=target_device, dtype=torch.float64)
            if protein is not None:
                supplied = protein if isinstance(protein, ReceptorContext) else load_receptor_context(protein, target_device)
                if supplied.coords.shape != expected_receptor.shape or not torch.allclose(
                    supplied.coords.to(expected_receptor), expected_receptor, atol=.002, rtol=0,
                ):
                    raise ValueError("protein coordinates/order disagree with AMBER topology; use its matching receptor or protein=None")
            coords = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float64, device=target_device)[None]
            prepared = model_factory.prepare({"active": torch.ones(mol.GetNumAtoms(), dtype=torch.bool)}, expected_receptor, {})
            metadata.update(amber=model_factory.metadata, receptor_source="explicit AMBER System",
                            component_semantics=prepared.component_semantics)
        else:
            use_prepared = (profile == "prepared-ad4" or preparation == "meeko"
                            or isinstance(ligand, (PreparedLigand, LigandPreparation))
                            or isinstance(protein, ReceptorPreparation))
            if profile == "prepared-ad4" and preparation == "inferred":
                raise ValueError("this SF requires prepared AD4 types and charges")
            if use_prepared:
                ligand_options.setdefault("generate_coordinates_if_missing", False)
                lp = ligand if isinstance(ligand, (LigandPreparation, PreparedLigand)) else prepare_ligand(ligand, seed=seed, **ligand_options)
                prepared_ligand = lp if isinstance(lp, PreparedLigand) else lp.prepared
                mol = prepared_ligand.mol
                coords = prepared_ligand.coordinates(target_device)[None]
                if isinstance(protein, ReceptorContext):
                    receptor = protein
                else:
                    rp = protein if isinstance(protein, ReceptorPreparation) else prepare_receptor(protein, **receptor_options)
                    receptor = rp.context(target_device)
                    metadata.update(receptor_preparation=rp.metadata, receptor_preparation_fingerprint=rp.fingerprint)
                metadata.update(ligand_preparation=getattr(lp, "metadata", {"contract": "supplied PreparedLigand"}),
                                ligand_preparation_fingerprint=prepared_ligand.fingerprint,
                                output_atom_to_input_atom=getattr(lp, "metadata", {}).get(
                                    "output_atom_to_input_atom", list(prepared_ligand.original_atom_indices)),
                                pdbqt_serials=list(prepared_ligand.pdbqt_serials))
            else:
                if prep_options:
                    raise ValueError("preparation_options require preparation=meeko or a prepared SF")
                mol = load_ligand_for_refinement(ligand, seed=seed, add_hydrogens=False, generate_coordinates_if_missing=False)
                coords = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32, device=target_device)[None]
                receptor = protein if isinstance(protein, ReceptorContext) else load_receptor_context(protein, target_device)
            needs_pattern = (getattr(model_factory, "supports_autograd", True) is False
                             or (info is not None and "autograd" not in info.optimization))
            engine = DockingEngine(model_factory, device=target_device, optimizer="pattern" if needs_pattern else "adam")
            problem = engine.prepare(mol, receptor, prepared_ligand=prepared_ligand,
                                     num_rotatable_bonds=None if prepared_ligand else len(find_rotatable_bonds(mol)))
            mol, prepared = problem.mol, problem.scorer
            metadata.update(receptor_fingerprint=receptor.structure_fingerprint,
                            receptor_source_fingerprint=receptor.source_fingerprint)
        needs_pattern = (getattr(model_factory, "supports_autograd", True) is False
                         or (info is not None and "autograd" not in info.optimization))
        route = optimizer or ("pattern" if needs_pattern else "adam")
        required_route = "pattern" if route == "pattern" else "autograd"
        if info is not None and required_route not in info.optimization:
            raise ValueError(f"{info.name} does not support the {required_route} optimization route")
        if getattr(model_factory, "supports_autograd", True) is False and route != "pattern":
            raise ValueError("nondifferentiable SF requires optimizer=pattern")
        rate = learning_rate if learning_rate is not None else {"adam": .05, "adamw": .05, "lbfgs": .5, "cartesian": 1., "pattern": .1}.get(route, .05)
        opt_options = dict(optimizer_options or {})
        if set(opt_options) & {"optimizer", "learning_rate", "num_steps"}:
            raise ValueError("set optimizer, learning_rate and num_steps through their explicit arguments")
        pivot = next((atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1), None)
        if pivot is None:
            raise ValueError("a ligand must contain at least one heavy atom")
        pose = SE3PoseModel(mol, coords, pivot, coords[:, pivot], torch.zeros(1, 3, device=target_device), target_device)
        if amber:
            pose = pose.double()
        initial = pose().detach()
        final, stats = optimize_pose_module(pose, prepared.search_energy, optimizer=route, num_steps=int(num_steps),
                                            learning_rate=rate, **opt_options)
        if final.ndim == 2:
            final = final[None]
        with torch.no_grad():
            initial_raw, final_raw = prepared.raw_components(initial), prepared.raw_components(final)
            reference = final_raw.intramolecular[0]
            before, after = prepared.report(initial_raw, reference), prepared.report(final_raw, reference)
        @torch.no_grad()
        def metrics(raw, report, xyz):
            direction = info.direction if info else getattr(model_factory, "direction", "lower")
            values = {"score": _metric(report.score[0], prepared.units, direction=direction),
                      "search_energy": _metric(raw.search_energy[0], prepared.units, role="optimization_objective")}
            if getattr(prepared, "component_semantics", "pairwise") == "undecomposed-full-complex-potential":
                for key, value in prepared.components(xyz).items():
                    values[key] = _metric(value[0], prepared.units, role="physical_potential")
            else:
                values.update(intermolecular=_metric(raw.intermolecular[0], prepared.units, role="component"),
                              intramolecular=_metric(raw.intramolecular[0], prepared.units, role="component"))
            return values
        metadata.update(scorer_fingerprint=prepared.fingerprint, optimization_route=route,
                        optimization_contract=OPTIMIZATION_CONTRACT, optimization=stats.as_dict(),
                        effective_tensor_device=str(initial.device), coordinate_dtype=str(initial.dtype),
                        learning_rate=float(rate), initial_coordinates_sha256="sha256:" + hashlib.sha256(
                            initial.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
        effective = inspect.signature(optimize_pose_module).bind_partial(
            optimizer=route, num_steps=int(num_steps), learning_rate=rate, **opt_options)
        effective.apply_defaults()
        metadata["optimization_parameters"] = _configuration({key: value for key, value in effective.arguments.items()
                                                               if key not in {"model", "energy_fn"}})
        metadata.setdefault("output_atom_to_input_atom", list(range(mol.GetNumAtoms())))
        result = PoseRefinementResult(selected_name, _mol_with_coordinates(mol, final[0]), final[0].detach().cpu(),
                                      metrics(final_raw, after, final), metrics(initial_raw, before, initial),
                                      getattr(model_factory, "objective", "scorer-defined lower-is-better search energy"), metadata)
        result.metadata["output_atomic_numbers"] = [atom.GetAtomicNum() for atom in result.mol.GetAtoms()]
        result.metadata["output_isomeric_smiles"] = Chem.MolToSmiles(result.mol, canonical=False, isomericSmiles=True)
        result.metadata["scorer_options"] = _configuration(options)
        if output_dir is not None:
            _write_result(result, output_dir)
        return result
    finally:
        if info is not None and callable(getattr(model_factory, "close", None)):
            model_factory.close()


def refine_with_scorers(protein, ligand, scorers: Sequence[str], *, scorer_options: Mapping[str, Mapping] | None = None,
                        on_error: str = "record", output_dir: str | Path | None = None, **options) -> dict:
    """Run every requested SF from the same input, with no cross-SF raw ranking."""
    if (isinstance(scorers, str) or not scorers or any(not isinstance(name, str) for name in scorers)
            or len(set(scorers)) != len(scorers)):
        raise ValueError("scorers must be a nonempty sequence of distinct registered names")
    if on_error not in {"record", "raise"}:
        raise ValueError("on_error must be record or raise")
    choices = dict(scorer_options or {})
    if set(choices) - set(scorers):
        raise ValueError("scorer_options contains an unrequested SF")
    results, errors = {}, {}
    for index, name in enumerate(scorers):
        destination = None
        if output_dir is not None:
            destination = Path(output_dir) / (f"{index:02d}-" + name.replace(":", "-"))
        try:
            result = refine_pose(protein, ligand, scorer=name, scorer_options=choices.get(name), output_dir=destination, **options)
            results[name] = result.as_dict()
        except Exception as error:
            if on_error == "raise":
                raise
            errors[name] = dict(type=type(error).__name__, message=str(error))
    value = dict(schema="anchor-dock-sf-comparison-v1", success=not errors, independent_runs=True, raw_scores_comparable=False,
                 common_input_sha256=_identity(ligand), results=results, errors=errors)
    if output_dir is not None:
        report = Path(output_dir) / "comparison.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        if report.exists():
            raise FileExistsError(report)
        report.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return value
