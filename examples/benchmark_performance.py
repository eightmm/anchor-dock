"""Compare exact-semantics kernels with an immutable Git baseline.

Example (from a clone containing the upstream commit)::

    uv run --frozen python examples/benchmark_performance.py \
        --baseline-ref 125a4f82ec94cc0ffeb71d44d92912f58ff9ba79

This measures preparation, scoring and a synthetic optimizer loop. It does not
measure end-to-end docking success, enrichment, or GPU performance on a CPU host.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import random
import statistics
import subprocess
import sys
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem

from anchor_dock.core import features, optimization, scoring

ROOT = Path(__file__).resolve().parents[1]


def _baseline_module(ref: str, stem: str) -> types.ModuleType:
    path = f"src/anchor_dock/core/{stem}.py"
    source = subprocess.run(
        ["git", "show", f"{ref}:{path}"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    name = f"anchor_dock.core._benchmark_baseline_{stem}"
    module = types.ModuleType(name)
    module.__file__ = f"git:{ref}:{path}"
    module.__package__ = "anchor_dock.core"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timings(
    before: Callable[[], Any], after: Callable[[], Any], *, device: torch.device, warmup: int, repeats: int
) -> dict[str, Any]:
    for _ in range(warmup):
        before()
        after()
    _sync(device)
    samples: dict[str, list[float]] = {"baseline": [], "current": []}
    # Alternate order to avoid always charging one implementation for drift.
    for repeat in range(repeats):
        ordered = (("baseline", before), ("current", after))
        if repeat % 2:
            ordered = tuple(reversed(ordered))
        for name, fn in ordered:
            _sync(device)
            start = time.perf_counter()
            fn()
            _sync(device)
            samples[name].append(1000.0 * (time.perf_counter() - start))
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {
        "median_ms": medians,
        "speedup": medians["baseline"] / medians["current"],
        "samples_ms": samples,
    }


def _memory(fn: Callable[[], Any], device: torch.device) -> dict[str, int | None]:
    # Logical saved tensor bytes count repeated views/references separately.
    # It is a reproducible autograd workload measure, NOT measured peak memory.
    logical_bytes = 0
    tensor_count = 0

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal logical_bytes, tensor_count
        logical_bytes += tensor.numel() * tensor.element_size()
        tensor_count += 1
        return tensor

    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        initial = torch.cuda.memory_allocated(device)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        fn()
    _sync(device)
    peak = torch.cuda.max_memory_allocated(device) - initial if device.type == "cuda" else None
    return {"saved_tensor_logical_bytes": logical_bytes, "saved_tensor_count": tensor_count,
            "cuda_peak_allocated_increment_bytes": peak}


def _max_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).abs().max().cpu()) if a.numel() else 0.0


def _feature_fixture(n: int, generator: torch.Generator) -> dict[str, torch.Tensor]:
    # Controlled synthetic atom attributes, not a claim of sampled chemistry.
    return {
        "radius_vina": 1.5 + torch.rand(n, generator=generator) * 0.6,
        "radius_vinardo": 1.5 + torch.rand(n, generator=generator) * 0.6,
        "hydrophobic": (torch.rand(n, generator=generator) < 0.4).float(),
        "donor": (torch.rand(n, generator=generator) < 0.25).float(),
        "acceptor": (torch.rand(n, generator=generator) < 0.35).float(),
        "active": torch.rand(n, generator=generator) < 0.92,
    }


def _score_bench(args: argparse.Namespace, baseline: types.ModuleType, device: torch.device) -> list[dict[str, Any]]:
    generator = torch.Generator().manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    ligand = {k: v.to(device) for k, v in _feature_fixture(args.ligand_atoms, generator).items()}
    receptor = {k: v.to(device) for k, v in _feature_fixture(args.receptor_atoms, generator).items()}
    receptor_xyz = ((torch.rand(args.receptor_atoms, 3, generator=generator) - 0.5) * 24).to(device, dtype)
    coordinates = ((torch.rand(args.batch_size, args.ligand_atoms, 3, generator=generator) - 0.5) * 8).to(device, dtype)
    triangular = torch.triu(torch.rand(args.ligand_atoms, args.ligand_atoms, generator=generator) < 0.25, diagonal=1)
    intra = (triangular | triangular.T).to(device, dtype)
    exclusion = (torch.rand(args.ligand_atoms, args.receptor_atoms, generator=generator) < 0.025).to(device)
    result = []
    for name in args.scorers:
        old = baseline.PairwiseScorer(name).prepare(
            ligand, receptor_xyz, receptor, num_rotatable_bonds=6,
            intramolecular_mask=intra, intermolecular_exclusion_mask=exclusion,
        )
        new = scoring.PairwiseScorer(name).prepare(
            ligand, receptor_xyz, receptor, num_rotatable_bonds=6,
            intramolecular_mask=intra, intermolecular_exclusion_mask=exclusion,
        )
        x_old = coordinates.clone().requires_grad_()
        x_new = coordinates.clone().requires_grad_()
        old_raw = old.raw_components(x_old)
        new_raw = new.raw_components(x_new)
        errors = {}
        for component in ("intermolecular", "intramolecular", "search_energy"):
            a, b = getattr(old_raw, component), getattr(new_raw, component)
            torch.testing.assert_close(a, b, rtol=args.rtol, atol=args.atol)
            errors[component] = _max_error(a, b)
        old_grad = torch.autograd.grad(old_raw.search_energy.sum(), x_old)[0]
        new_grad = torch.autograd.grad(new_raw.search_energy.sum(), x_new)[0]
        torch.testing.assert_close(old_grad, new_grad, rtol=args.rtol, atol=args.atol)
        errors["coordinate_gradient"] = _max_error(old_grad, new_grad)
        del old_raw, new_raw, old_grad, new_grad

        def forward(prepared):
            with torch.no_grad():
                return prepared.search_energy(coordinates)

        def backward(prepared, coords):
            return torch.autograd.grad(prepared.search_energy(coords).sum(), coords)[0]

        result.append({
            "scorer": name, "max_absolute_error": errors,
            "forward": _timings(lambda: forward(old), lambda: forward(new), device=device,
                                warmup=args.warmup, repeats=args.repeats),
            "forward_and_gradient": _timings(lambda: backward(old, x_old), lambda: backward(new, x_new),
                                             device=device, warmup=args.warmup, repeats=args.repeats),
            "autograd_memory": {"baseline": _memory(lambda: backward(old, x_old), device),
                                "current": _memory(lambda: backward(new, x_new), device)},
        })
    return result


def _assert_features(before: dict[str, Any], after: dict[str, Any]) -> None:
    if set(before) != set(after):
        raise AssertionError("feature names differ")
    for name, value in before.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, after[name], rtol=0, atol=0)
        elif value != after[name]:
            raise AssertionError(f"feature {name} differs")


def _preparation_bench(args: argparse.Namespace, baseline: types.ModuleType, device: torch.device) -> list[dict[str, Any]]:
    if baseline.ATOM_TYPING_VERSION != features.ATOM_TYPING_VERSION:
        return [{"skipped": "atom-typing contracts differ; preparation timings would compare different semantics",
                 "baseline_contract": baseline.ATOM_TYPING_VERSION, "current_contract": features.ATOM_TYPING_VERSION}]
    ligands = Chem.MolFromSmiles("CC(=O)Nc1ccc(O)c(S(=O)(=O)NCCN)c1")
    if ligands is None:
        raise RuntimeError("benchmark ligand could not be constructed")
    fixtures = [("ligand", ligands)]
    pocket_path = ROOT / "examples/10gs/10gs_pocket.pdb"
    if pocket_path.is_file():
        protein = Chem.MolFromPDBFile(str(pocket_path), sanitize=False, removeHs=False)
        if protein is None:
            raise RuntimeError("benchmark protein could not be read")
        fixtures.append(("unsanitized_pdb_pocket", protein))
    result = []
    for name, mol in fixtures:
        old_mol, new_mol = Chem.Mol(mol), Chem.Mol(mol)
        _assert_features(baseline.compute_atom_features(old_mol, device), features.compute_atom_features(new_mol, device))
        result.append({"fixture": name, "atoms": mol.GetNumAtoms(), "exact_feature_parity": True,
                       **_timings(lambda: baseline.compute_atom_features(old_mol, device),
                                  lambda: features.compute_atom_features(new_mol, device), device=device,
                                  warmup=args.warmup, repeats=args.repeats)})
    return result


def _optimization_bench(args: argparse.Namespace, baseline: types.ModuleType, device: torch.device) -> dict[str, Any]:
    before_contract = getattr(baseline, "OPTIMIZATION_CONTRACT", "legacy-v1")
    if before_contract != optimization.OPTIMIZATION_CONTRACT:
        return {"skipped": "optimizer contracts differ; pose trajectories are intentionally not parity-comparable",
                "baseline_contract": before_contract, "current_contract": optimization.OPTIMIZATION_CONTRACT}
    class Pose(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.positions = torch.nn.Parameter(values.clone())

        def forward(self):
            return self.positions[:, None, :]

    values = torch.linspace(0.0, 2.0, args.batch_size * 3, device=device).reshape(args.batch_size, 3)

    def run(module):
        model = Pose(values)
        return module.optimize_pose_module(
            model, lambda xyz: xyz.square().sum(dim=(1, 2)), num_steps=args.optimizer_steps,
            early_stopping=False, optimizer="adam",
        )

    old_coords, old_stats = run(baseline)
    new_coords, new_stats = run(optimization)
    torch.testing.assert_close(old_coords, new_coords, rtol=args.rtol, atol=args.atol)
    for name, value in old_stats.as_dict().items():
        other = new_stats.as_dict()[name]
        if isinstance(value, int):
            if value != other:
                raise AssertionError(f"optimizer stat {name} differs")
        elif abs(value - other) > args.atol + args.rtol * abs(value):
            raise AssertionError(f"optimizer stat {name} differs")
    return {"fixture": "synthetic_quadratic_adam_loop", "steps": args.optimizer_steps,
            "early_stopping": False, "max_coordinate_error": _max_error(old_coords, new_coords),
            **_timings(lambda: run(baseline), lambda: run(optimization), device=device,
                       warmup=args.warmup, repeats=args.repeats)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True, help="Immutable Git commit to compare with")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ligand-atoms", type=int, default=32)
    parser.add_argument("--receptor-atoms", type=int, default=512)
    parser.add_argument("--scorers", nargs="+", choices=("vina", "vinardo"), default=["vina", "vinardo"])
    parser.add_argument("--skip-preparation", action="store_true", help="Skip potentially expensive RDKit preparation")
    parser.add_argument("--skip-optimization", action="store_true", help="Skip the synthetic optimizer loop")
    parser.add_argument("--optimizer-steps", type=int, default=75)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    if any(getattr(args, x) < 1 for x in ("threads", "batch_size", "ligand_atoms", "receptor_atoms", "repeats")):
        parser.error("threads, dimensions and repeats must be positive")
    if args.warmup < 0 or args.optimizer_steps < 0 or args.atol < 0 or args.rtol < 0:
        parser.error("warmup, steps and tolerances must be nonnegative")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        parser.error("only an available CPU or CUDA device is supported")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    baseline_ref = subprocess.run(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{args.baseline_ref}^{{commit}}"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    baseline_scoring = _baseline_module(baseline_ref, "scoring")
    baseline_features = _baseline_module(baseline_ref, "features")
    baseline_optimization = _baseline_module(baseline_ref, "optimization")
    report = {
        "scope": "kernel/preparation/synthetic-loop benchmark; not end-to-end docking",
        "baseline_commit": baseline_ref,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "rdkit": importlib.metadata.version("rdkit"),
                        "device": str(device), "cuda_available": torch.cuda.is_available(),
                        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
                        "threads": torch.get_num_threads()},
        "parameters": vars(args),
        "scoring": _score_bench(args, baseline_scoring, device),
        "preparation": [] if args.skip_preparation else _preparation_bench(args, baseline_features, device),
        "optimization": None if args.skip_optimization else _optimization_bench(args, baseline_optimization, device),
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
