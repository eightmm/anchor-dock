# Examples and CPU checks

Run these scripts from a source checkout with its locked environment. The molecular test fixtures referenced by numerical audits are repository data, not files installed by the package wheel. Commands below write new reports to a caller-chosen directory; existing example predictions are historical artifacts.

```bash
uv sync --frozen --all-extras --group dev
```

## Guided docking examples

| Script/input | Purpose | Required inputs |
|---|---|---|
| `reference.py` | Reference-MCS docking of an ibuprofen query | Bundled `10gs` pocket/reference; writes `examples/output/reference` |
| `covalent.py` | Supported residue–warhead product docking | Protein, exact residue selector and ligand arguments |
| `interaction.py` | One explicit atom-pair distance guide | Protein, residue/atom, ligand, mapped SMARTS and distance window arguments |
| `interaction_multi.py` | Two simultaneous `AND` guides | Replace the illustrative receptor/ligand paths and hypotheses |
| `batch.py` | Mixed-job Python batch | Replace illustrative paths/jobs with valid inputs |
| `custom_scorer.py` | Custom differentiable Torch objective | Replace `pocket.pdb` and illustrative interaction hypothesis |
| `batch/jobs.jsonl`, `batch/ligands.smi`, `batch/ligands.csv` | Manifest/input templates | Supply matching protein/reference and required mode options |

The reference example generates 128 conformers and optimizes them, so it is a workflow demonstration rather than a minimal CPU smoke test. Interaction and covalent selectors are scientific inputs, not automatically inferred labels. See [usage](../docs/USAGE.md) and [batch contracts](../docs/BATCH.md).

`batch/ligands.smi` is a homogeneous ligand list. Supply the protein and either all five single-interaction fields or a canonical `interactions` list. JSON/JSONL supports nested multi-interaction lists; CSV is intended for its documented flat job fields.

## Numerical audits

| Script | Check | Dependencies and data |
|---|---|---|
| `audit_atom_typing.py` | Inferred/prepared atom-role comparison | `prepare` extra; generates declared ligand cases |
| `audit_native_scoring.py` | Continuous and FAST-table reference versus native Vina/Vinardo | `native-vina` extra; generates seeded cases; fixture regeneration is explicit |
| `audit_ad4.py` | AutoGrid map, intra and report conventions versus native Vina AD4 | `native-vina` extra; synthetic maps, not a receptor-map generation benchmark |
| `audit_amber.py` | Explicit OpenMM force bridge versus independent Context | `amber` extra; caller-supplied prmtop/inpcrd and all-atom particle indices |

The optional Vina binding is an oracle for these checks. These scripts do not restore removed native docking optimizer adapters. `audit_amber.py` evaluates the explicit `OpenMMScorer` bridge; use the physical benchmark below for the named pure Torch `amber` implementation.

For example, compare the bridge on the bundled topology under its declared vacuum model:

```bash
uv run --frozen --all-extras python examples/audit_amber.py \
  --prmtop tests/fixtures/amber/alanine-dipeptide-implicit.prmtop \
  --inpcrd tests/fixtures/amber/alanine-dipeptide-implicit.inpcrd \
  --indices '[6,7,8,9,10,11,12,13,14,15]' \
  --implicit-solvent none --output audit-results/openmm-reference.json
```

The source fixture filename contains `implicit`; selecting `none` above still constructs a vacuum NoCutoff System. Its provenance is recorded [with the fixture](../tests/fixtures/amber/README.md).

## Benchmarks

| Script | Scope | Important boundary |
|---|---|---|
| `benchmark_torch_physical.py` | Pure Torch AMBER cached/tiled energy and gradient parity, plus timing against the OpenMM bridge | Uses the bundled 22-atom fixture by default; imports/setup are outside timings |
| `benchmark_redocking.py` | Seeded rigid/torsional perturbation recovery in a known pocket | Bundled 10gs defaults; fixed receptor-frame symmetry RMSD without alignment; not global docking |
| `benchmark_performance.py` | Preparation, scorer and synthetic optimizer kernels versus an immutable Git baseline | Requires an available compatible `--baseline-ref`; no docking-accuracy conclusion |

A short reproducible CPU physical check:

```bash
uv run --frozen --all-extras python examples/benchmark_torch_physical.py \
  --device cpu --threads 1 --batch-sizes 1 8 \
  --warmup 1 --repeats 3 --output audit-results/torch-physical.json
```

For a full recorded timing run, retain the script's warmup/repeat defaults. Run timing experiments serially on an otherwise idle host, compare the same hardware/thread/dtype settings, and retain raw samples and source hashes. Faster isolated intra or small-fixture timings do not guarantee faster full docking or GPU execution. [Latest CPU validation](../reports/cpu-validation-2026-10-30.md) and the [report index](../reports/README.md) distinguish current verification from dated experiments.
