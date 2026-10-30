"""Reproduce force/energy parity and batched Torch-vs-OpenMM timing.

The bundled 22-atom fixture is a numerical oracle, not a docking benchmark.
Both scorers import the same vacuum NoCutoff AMBER topology. Imports and
fixed-receptor preparation are excluded from forward/backward timings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from anchor_dock import OpenMMScorer, TorchAmberScorer, __version__


def evaluate(scorer, coordinates):
    candidate = coordinates.detach().clone().requires_grad_()
    values = scorer.energy(candidate)
    gradient, = torch.autograd.grad(values.sum(), candidate)
    return values.detach(), gradient.detach()


def measure(models, coordinates, repeats, warmup):
    def synchronize():
        if coordinates.is_cuda:
            torch.cuda.synchronize(coordinates.device)

    for _ in range(warmup):
        for model in models.values():
            evaluate(model, coordinates)
    samples = {name: [] for name in models}
    names = list(models)
    for iteration in range(repeats):
        # Rotate measurement order so each backend occupies each time slot.
        order = names[iteration % len(names):] + names[:iteration % len(names)]
        for name in order:
            synchronize()
            start = time.perf_counter()
            evaluate(models[name], coordinates)
            synchronize()
            samples[name].append(1000 * (time.perf_counter() - start))
    return {name: dict(median_ms=statistics.median(values), q25_ms=float(np.quantile(values, .25)),
                       q75_ms=float(np.quantile(values, .75)), samples_ms=values)
            for name, values in samples.items()}


def main():
    fixture = Path(__file__).resolve().parents[1] / "tests/fixtures/amber"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prmtop", type=Path, default=fixture / "alanine-dipeptide-implicit.prmtop")
    parser.add_argument("--inpcrd", type=Path, default=fixture / "alanine-dipeptide-implicit.inpcrd")
    parser.add_argument("--indices", type=json.loads, default=list(range(6, 16)))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=713)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch_sizes) < 1 or args.repeats < 1 or args.warmup < 0 or args.threads < 1:
        parser.error("batch sizes, repeats and threads must be positive; warmup must be nonnegative")
    torch.set_num_threads(args.threads)
    generator = torch.Generator().manual_seed(args.seed)
    options = dict(prmtop=args.prmtop, inpcrd=args.inpcrd, ligand_atom_indices=args.indices)
    scorer = TorchAmberScorer.from_amber(**options, device=args.device)
    uncached = TorchAmberScorer.from_amber(**options, device=args.device, pair_cache_max_pairs=0)
    oracle = OpenMMScorer.from_amber(**options, platform="Reference")
    try:
        rows = []
        base = scorer.ligand_coordinates(device="cpu", dtype=torch.float64)
        for batch in args.batch_sizes:
            coordinates = (base[None] + .01 * torch.randn(batch, *base.shape, generator=generator,
                                                         dtype=torch.float64)).to(args.device)
            actual_e, actual_g = evaluate(scorer, coordinates)
            oracle_e, oracle_g = evaluate(oracle, coordinates)
            uncached_e, uncached_g = evaluate(uncached, coordinates)
            errors = dict(energy_kcal_mol=float((actual_e - oracle_e).abs().max()),
                          gradient_kcal_mol_angstrom=float((actual_g - oracle_g).abs().max()),
                          cache_energy_kcal_mol=float((actual_e - uncached_e).abs().max()),
                          cache_gradient_kcal_mol_angstrom=float((actual_g - uncached_g).abs().max()))
            if max(errors.values()) > 1e-9:
                raise AssertionError(f"physical energy/gradient parity failed: {errors}")
            timings = measure({"torch": scorer, "torch_uncached": uncached, "openmm_bridge": oracle},
                              coordinates, args.repeats, args.warmup)
            rows.append(dict(batch=batch, errors=errors, timings=timings,
                             cache_speedup=timings["torch_uncached"]["median_ms"] / timings["torch"]["median_ms"],
                             speedup=timings["openmm_bridge"]["median_ms"] / timings["torch"]["median_ms"]))
        value = dict(scope="22-atom force/energy microbenchmark; not docking accuracy or general throughput",
                     version=__version__, seed=args.seed, dtype="float64", device=args.device,
                     torch_threads=torch.get_num_threads(), python=platform.python_version(),
                     torch=torch.__version__, openmm=oracle.metadata["openmm_version"],
                     oracle_platform="Reference", cuda_available=torch.cuda.is_available(),
                     prmtop_sha256=hashlib.sha256(args.prmtop.read_bytes()).hexdigest(),
                     inpcrd_sha256=hashlib.sha256(args.inpcrd.read_bytes()).hexdigest(),
                     script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     torch_scorer=scorer.metadata, rows=rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        print(json.dumps([{k: row[k] for k in ("batch", "errors", "cache_speedup", "speedup")} for row in rows]))
    finally:
        oracle.close()


if __name__ == "__main__":
    main()
