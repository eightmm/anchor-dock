# AnchorDock performance check — 2026-10-30

> **Historical snapshot.** For the current implementation and verified CPU status, see [CPU validation](cpu-validation-2026-10-30.md). The results and commands below belong to the recorded source version; their original evidence is retained.

The patch reuses one RDKit donor/acceptor extraction during preparation and removes repeated CUDA host reads from optimizer row helpers. Atomic typing, score equations, restraint/release behavior, and public APIs retain their existing semantics.

This historical report describes performance-only commit `beefe6a79df0b0c4b86e81ca0833ddd7c3627ab8`, before the typing/Jacobian corrections. See [the scientific audit](scoring-audit-2026-10-30.md) for current behavior. Check out that commit before reproducing these same-semantics timings.

Base: `125a4f82ec94cc0ffeb71d44d92912f58ff9ba79`. The validation environment held an immutable local snapshot (`372bd86`) of that fetched source; the benchmark JSON retains that local provenance. Use the upstream commit when reproducing from a full clone.

## Measured CPU preparation and optimizer overhead

Python 3.12.14, Torch 2.10.0+cu128, RDKit 2025.9.5, CPU, one Torch thread. Same-process baseline/current executions use fixed inputs, one warmup and five alternating-order repetitions; times below are medians. Raw samples and parity checks are in [performance-2026-10-30.json](performance-2026-10-30.json).

| Workload | Baseline | Patched | Ratio |
|---|---:|---:|---:|
| Ligand feature preparation, 18 atoms | 3.893 ms | 2.062 ms | 1.89× |
| Unsanitized bundled 10gs pocket, 1,392 atoms | 5,796.126 ms | 2,844.012 ms | 2.04× |
| Synthetic quadratic Adam loop, batch 16, 75 steps | 44.010 ms | 40.831 ms | 1.08× |

These are preparation and optimizer-overhead measurements, not complete docking speedups. Scorer timing samples are included for transparency but do not establish a general scorer-throughput improvement. A streamed pair-term experiment showed inconsistent threaded-CPU backward behavior and was not adopted; the accepted weighted scoring path retains the existing dense equations. The score-ramp change only removes temporary scalar tensors while preserving endpoint derivatives. Autograd saved-tensor logical bytes were unchanged; CUDA peak memory was not measured.

## Changes

- `features.py`: compute RDKit feature sets/PDB fallbacks once and share them with XS typing; resolve H/S/metal overrides before constructing final donor/acceptor tensors. No atom-by-atom GPU writes.
- `optimization.py`: restore/update pose rows with broadcast `where`, clear Adam moments with `masked_fill_`, and update gradient masks/step counts without dynamic boolean gathers. CPU empty-mask shortcuts remain. Required finite-loss and convergence checks remain; this does not remove every synchronization from optimization.
- `scoring.py`: use scalar `where` branches instead of constructing small device tensors for every piecewise-ramp evaluation. The original zero-gradient endpoint convention remains.

## Validation

- Baseline suite: 347 passed.
- Final suite: **445 passed, 13 CUDA-only skipped**.
- Ruff, `uv lock --check`, wheel build, installed-wheel import/version/API and all four CLI help commands passed.
- Independent legacy scoring equations cover raw scores, ligand/receptor coordinate gradients, float32/float64, rank-2 inputs, 2D/broadcast/per-row masks, strict cutoff, inactive atoms, custom weight order, zero weights and piecewise endpoint derivatives.
- Row tests cover partial convergence, inactive parameters/moments, noncontiguous layouts, unselected NaN snapshots, and repeated guide/release calls. Adam/AdamW/LBFGS were also compared against the baseline module across dtype and early-stopping settings.
- Six separate-process pipeline fixtures matched the baseline's exported coordinates, full best scores, per-pose scores and ranks exactly: interaction with Vina/Vinardo/SoftDock, multi-interaction, reference and covalent. The interaction fixtures use 2 guided + 1 release steps; reference/covalent use 20 optimization steps. These are regression fixtures, not scientific accuracy benchmarks.
- A read-only independent code review found no blocking correctness issue.

## Reproduce

```bash
uv sync --frozen --group dev
uv run --frozen pytest -q
uv run --frozen ruff check src tests examples
uv run --frozen python examples/benchmark_performance.py \
  --baseline-ref 125a4f82ec94cc0ffeb71d44d92912f58ff9ba79 \
  --threads 1 --warmup 1 --repeats 5
```

The benchmark also supports `--device cuda`, `--dtype float64`, dimensions, repeated timings, and stage skips. Run on an otherwise idle machine; small kernel medians vary with thread count and host load. CUDA row-helper profiler/semantic tests are included but were skipped on this host. GPU throughput, GPU peak memory, and a large target/ligand benchmark remain unmeasured.
