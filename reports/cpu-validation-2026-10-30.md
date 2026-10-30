# Current CPU validation — 0.6.2, 2026-10-30

This is the current implementation and documentation checkpoint. Earlier dated reports retain historical observations at their recorded source versions. The package is 0.6.2, batch resume epoch is 8, ranked SDF schema remains 4, and the independent SF result schemas remain version 1.

The [machine-readable manifest](cpu-validation-2026-10-30.json) identifies all 41 Python source files and hashes each raw evidence artifact. Numerical checks, public workflows, timing experiments and documentation checks are separate scopes. None establishes global docking completeness, affinity accuracy or GPU performance.

## Corrections found by CPU inspection

Vina/Vinardo factories previously shared mutable config weights with caller/global objects, while factory fingerprints were captured only once. Prepared scorers also retained caller-owned coordinates/features/masks. A later mutation could change a calculation without matching its recorded identity. Factories now snapshot incoming configs and compute their fingerprint from current configuration; preparation snapshots its own config, receptor coordinates, features and masks. Tensor clones retain the existing autograd connection. Factory changes affect subsequent preparation, not already prepared objectives.

The actual weight evaluation order is included in factory, prepared and standalone config fingerprints. Sorted JSON values alone missed this order: a finite custom cancellation example produced energies 0 and 1 with the same old fingerprint. The fix changes identity, not the evaluation order. Sixteen ordinary Vina/Vinardo × FP32/FP64 × dense/sparse × curl cases retain bitwise-identical energies and coordinate gradients; the four runtime energy methods retain their original source bodies. [Pairwise evidence](cpu-validation-2026-10-30/pairwise-audit.json).

Torch AMBER previously computed unused LJ powers for epsilon-zero Coulomb-only pairs. Finite but extreme unused sigma values could overflow and yield `0 * infinity`. Inactive LJ parameters are now canonicalized before powers in cached, tiled and fixed-energy evaluation. Analytical Coulomb energy, first/second derivatives and exception cases are covered. Ordinary force-field parameters and physical formulas are unchanged. The Torch AMBER contract is `torch-fixed-charge-vacuum-nocutoff-fp64-v3`. [AMBER evidence](cpu-validation-2026-10-30/amber-audit.json).

## Numerical oracles

| Check | Executed scope | Observed result |
|---|---|---|
| Atom typing | 24 cases against Meeko 0.8.0 | All compared XS assignments match |
| Native Vina/Vinardo table reference | 48 same-preparation comparisons with Vina 1.2.7 | Largest component error 4.10e-6; largest zero-reference total error 2.92e-6 |
| AD4 maps and report | Flexible ligands in synthetic affine maps, native Vina AD4 | Intermolecular/report differences below 3.56e-15 |
| AD4 analytic intra vs native radial tables | Same explicit contributors | Maximum difference 9.12e-4; the distinct table probe agrees to 1.12e-16 |
| Torch AMBER | 12 random full-FF Systems, 72 map/tile/cache configurations, 216 poses against OpenMM Reference | Maximum energy error 3.18e-12 kcal/mol; gradient error 1.82e-11 kcal/mol/angstrom |

These limits are implementation-specific. The continuous Vina-family search objective is not identical to the native FAST lookup/curl objective: in this controlled ligand set, uncapped analytic and native inter energy differ by up to 0.585, and matching positive caps still leaves up to 0.166. Native-table final references are exposed separately. AD4 analytic intra terms likewise are not native radial-table evaluation. These known convention differences are not hidden by a single universal parity tolerance.

Raw evidence: [typing](cpu-validation-2026-10-30/atom-typing.json), [Vina-family references](cpu-validation-2026-10-30/native-vina-parity.json), [AD4](cpu-validation-2026-10-30/ad4-parity.json), [AMBER](cpu-validation-2026-10-30/amber-audit.json).

## Public CPU workflows

Eight workflow groups pass against the frozen source: six CLI commands and removed-option rejection; repeatable 10gs reference docking; independent Vina/Vinardo pose refinement; single interaction; two-interaction AND guidance; covalent docking; three-mode batch/resume; and prepared AD4/ADGPU map reuse. Input structures remain unchanged, and SDF results are read back and checked. [Workflow evidence](cpu-validation-2026-10-30/workflows.json).

- Reference MCS anchor readback error is 0 angstrom and repeated coordinates/scores match.
- The 20-step independent 10gs refinement search energies change from -15.98453 to -17.14097 for Vina and -14.82740 to -15.69870 for Vinardo. These are separate objectives, not cross-SF rankings.
- Guided outputs are checked against an independently reconstructed scorer with no guide penalty; maximum search-energy discrepancy is 4.14e-9.
- Covalent SDF bond distances are 1.819971 and 1.819954 angstrom in the tested fixture.
- Three successful batch jobs are reused on resume with matching signatures and artifacts.
- Map workflow checks use synthetic constant maps. This does not validate actual AutoGrid generation or a receptor-specific map preparation pipeline.

## Small local pose-recovery check

The current source was also run on 10gs, 1iep and 5x72, using fixed-frame symmetry-aware heavy-atom RMSD, Vina, seed 713, one CPU thread, 40 steps and two poses per mild/moderate perturbation level. Each complex also has an unperturbed crystal control. Counts below exclude that control.

| Complex | Initial within 2 angstrom | Adam final | LBFGS final | Cartesian final |
|---|---:|---:|---:|---:|
| 10gs | 2/4 | 3/4 | 4/4 | 2/4 |
| 1iep | 2/4 | 3/4 | 3/4 | 3/4 |
| 5x72 | 3/4 | 4/4 | 4/4 | 4/4 |
| Total | 7/12 | 10/12 | 11/12 | 9/12 |

All recorded search energies are nonincreasing. No initially within-2-angstrom pose is lost in these runs. The Cartesian metric and energy minimization do not guarantee better RMSD: 10gs Cartesian has no newly recovered poses in this short budget. Twelve near-native perturbations per optimizer are too few to establish a general winner. Default optimizer and iteration budgets are not changed based on these counts.

Inputs retain the archived [preparation policy and pinned public sources](redocking-2026-10-30.md#input-provenance): complete residues within 12 angstrom for the external pockets and explicit A-altloc selection for 5x72. Current input and starting-pose hashes are in [10gs](cpu-validation-2026-10-30/recovery-10gs.json), [1iep](cpu-validation-2026-10-30/recovery-1iep.json) and [5x72](cpu-validation-2026-10-30/recovery-5x72.json). Package and driver sources remain unchanged during each run. This is local refinement of known-pocket perturbations, not de novo docking.

## CPU timings and limits

Timings use one Torch thread and retain raw samples. The ordinary Vina/Vinardo forward/backward comparisons with the equivalent 0.6.1 baseline have zero energy and gradient deltas. Median old/new speed ratios are 0.948 and 0.966 in this small kernel run; no new speedup is claimed from the correctness fixes. Timing differences must be read with their sample dispersion. [Kernel/preparation/synthetic-loop data](cpu-validation-2026-10-30/kernel-benchmark.json).

The public equivalent baseline is commit `2e9042beef6b4b7cb9c3bace3c3bf6bdeba89de3`. The local reconstructed checkout uses commit label `4e11680`; its relevant source bytes match that public snapshot. The raw benchmark retains the actual local label instead of inventing provider history.

| AMBER batch | Cached/uncached Torch speed ratio | Cached Torch/OpenMM Reference bridge speed ratio |
|---|---:|---:|
| 1 | 1.486 | 0.278 |
| 32 | 1.234 | 0.758 |
| 128 | 1.130 | 1.161 |

The AMBER microbenchmark uses the bundled 22-atom fixture as vacuum NoCutoff, perturbed coordinates and seven interleaved samples after two warmups. OpenMM is faster for small CPU batches. A ratio above one in a single larger batch does not establish general superiority. This also does not measure GPU performance or protein-sized throughput. [Physical benchmark](cpu-validation-2026-10-30/physical-benchmark.json).

## Documentation and package verification

The current README, user guide, API signatures, scoring contracts, preparation/map guide, batch guide, architecture, examples inventory, migration history and changelog are checked against the implementation. Duplicated architecture text is reduced to a canonical guide plus a compatibility entry. Historical report contents are retained behind explicit status notices.

The frozen all-extras suite passes **985 tests**, with **18 CUDA-only skips**. Ruff, frozen-lock validation, whitespace validation and a fresh wheel build pass. Installed-wheel verification checks source identity, module origins, all CLI commands, independent SF execution and the newly fixed mutation/zero-LJ cases. The machine-readable manifest records the final documentation and wheel results.

Current setup and reproducible numerical commands:

```bash
uv sync --frozen --all-extras --group dev
uv run --frozen --all-extras pytest -q
uv run --frozen ruff check src tests examples
uv lock --check
uv run --frozen --all-extras python examples/audit_atom_typing.py --output cpu-typing.json
uv run --frozen --all-extras python examples/audit_native_scoring.py --output cpu-native.json
uv run --frozen --all-extras python examples/audit_ad4.py --output cpu-ad4.json
uv run --frozen --all-extras python examples/benchmark_torch_physical.py \
  --device cpu --threads 1 --repeats 7 --warmup 2 --output cpu-physical.json
uv run --frozen python examples/benchmark_redocking.py \
  --device cpu --threads 1 --scorer vina --optimizers adam lbfgs cartesian \
  --steps 40 --poses-per-level 2 --seed 713 --output cpu-recovery-10gs.json
```

For 1iep/5x72, supply the matching prepared `--ligand`, `--receptor` and `--complex-id`; those external inputs are not bundled. The audit scripts and tests state their optional dependencies. [User guide](../docs/USAGE.md), [API](../docs/API_REFERENCE.md), [scoring functions](../docs/SCORING_FUNCTIONS.md) and [migration](../docs/MIGRATION.md) describe supported production inputs and boundaries.
