# Torch physical scoring — 2026-10-30

> **Historical snapshot.** For the current implementation and verified CPU status, see [CPU validation](cpu-validation-2026-10-30.md). The results and commands below belong to the recorded source version; their original evidence is retained.

AnchorDock 0.6.1 focuses on Torch energy functions and shared SE(3)+torsion optimization. Public Smina/GNINA/native-Vina optimizer adapters and the three DKOES presets are removed. They are not replaced silently. Vina remains the interaction default; Vina and Vinardo retain their empirical meaning.

The catalog contains five continuous objectives (`vina`, `vinardo`, `ad4`, `adgpu`, `amber`) and two separate nondifferentiable table references (`vina-final`, `vinardo-final`). These seven entries are not seven independent physical models. Native Vina remains an optional numerical oracle, while `OpenMMScorer` remains an explicit reference/advanced backend.

## Physical implementation and correctness

The named `amber` function now imports a fully parameterized AMBER topology once and evaluates harmonic bonds/angles, periodic torsions, Coulomb/Lennard-Jones and explicit exceptions in Torch. OpenMM is not called during energy/gradient evaluation. Input atom indices, elements and available topology connectivity are checked; bonded H must remain with their parent. Unsupported forces, periodic electrostatics, implicit solvent and parameter offsets fail explicitly.

The implementation uses FP64, caches fixed–fixed energy once, and optimizes the moving contribution relative to the reference pose. Up to 200,000 candidate moving pairs have their indices and exception-adjusted coefficients cached during preparation. Larger systems retain tiled evaluation; cached evaluation is tiled too. Both routes evaluate the same declared potential. This is full-complex potential energy, not binding affinity or MM/GBSA subtraction.

The official 22-atom AMBER fixture is evaluated as vacuum NoCutoff despite its source filename containing `implicit`. The benchmark's perturbed batches agree with an independent OpenMM Reference Context to maximum energy error **1.28e-13 kcal/mol** and gradient error **1.29e-12 kcal/mol/angstrom**. Cached/fallback errors are at most **2.14e-14** in energy and **7.11e-15** in gradient. Tests also cover nonzero torsion phases, reordered maps, 1–4 exceptions, batch independence, fixed receptor state, unsupported systems, gradcheck and gradgradcheck. Runtime tests explicitly prohibit construction of an OpenMM Context.

Cartesian pullback now rejects physical-domain trial errors through backtracking. Invalid initial geometries still fail. The physical potential is not softened to hide singular geometry.

## Measured optimizations

All timings below use CPU, one Torch thread and forward plus backward. They are kernel microbenchmarks with retained raw samples, not docking-throughput or docking-accuracy measurements. CUDA was unavailable.

AD4 now shares cell geometry and eight-corner interpolation weights across affinity, electrostatic and desolvation samples. Its ordinary sampler remains unchanged. Against the previous repeat-three sampling path:

| Batch / ligand atoms | FP32 speedup | FP64 speedup |
|---|---:|---:|
| 1 / 40 | 1.064x | 1.052x |
| 64 / 40 | 1.210x | 1.131x |
| 256 / 64 | 1.346x | 1.221x |

The positive-curl FP64 case improves 1.200x. All measured forward values are bitwise identical. Shared backward reduction changes rounding: maximum absolute differences are 2.38e-6 (FP32) and 3.55e-15 (FP64), with no gradient-sign changes in these cases. Near-zero values can have large ULP distances; the JSON records absolute errors and ULPs by magnitude. Actual Vina AD4 map/intra/report oracle tests pass.

Torch AMBER static pair caching, using the bundled fixture and interleaved measurement order:

| Batch | Cached vs uncached Torch | Cached Torch vs OpenMM Reference bridge |
|---|---:|---:|
| 1 | 1.532x | 0.315x |
| 32 | 1.309x | 0.834x |
| 128 | 1.174x | 1.175x |

OpenMM is faster for small CPU batches. The Torch implementation supplies the shared differentiable objective; these numbers do not establish a universal speed advantage. Reproduce with `python examples/benchmark_torch_physical.py --output benchmark.json` in an environment with the `amber` extra. Inputs, seed, versions, timing samples and source hashes are recorded.

`PairwiseScorer(..., intra_mode="sparse")` is opt-in. It evaluates only contributing intra pairs. On prepared 10gs, isolated intra work improves 1.805x, but full-scorer timing is 0.899x because intermolecular work dominates and allocation/noise matter. Synthetic sparse masks show larger isolated gains, without proving general end-to-end gains. FP32 direct pair norms and shorter reductions differ from dense `cdist`; the default dense numerical path remains intact.

## Validation and provenance

The frozen all-extras suite passes **970 tests**, with **18 CUDA-only skips**. Ruff, lock validation and a fresh wheel build pass. Installed-wheel validation confirms all 41 Python source hashes, exact catalog contents, absence of removed modules, independent SF execution, Context-free AMBER energy/gradient/Cartesian refinement, prepared ADGPU map reuse and six CLI commands. The wheel is built after clearing generated build intermediates so deleted adapter modules cannot leak into the package. CI separately covers core-only and optional physical/reference dependencies.

[Machine-readable evidence](torch-physical-2026-10-30.json) contains source-file hashes, catalog contracts, all benchmark samples and numeric errors. The immutable public pre-change baseline is commit `3ef03f25d2143abdde0b5c9ffa22ce5ca00be970`; local benchmark commit labels identify the reconstructed checkout, while source hashes identify the exact files used. [Fixture provenance](../tests/fixtures/amber/README.md) pins its upstream source and hashes.

[Usage and supported boundaries](../docs/SCORING_FUNCTIONS.md). Batch resume epoch is 7, ranked docking schema remains 4. Broad multi-target pose recovery, GPU speed and binding-affinity calibration remain unmeasured.
