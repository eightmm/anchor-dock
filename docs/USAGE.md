# Usage

## Install

```bash
uv sync --frozen --group dev
```

Python 3.12 or newer is required.

## CLI

Reference-guided docking:

```bash
uv run anchor-dock reference \
  --protein examples/10gs/10gs_pocket.pdb \
  --reference examples/10gs/10gs_ligand.sdf \
  --query "CC(=O)Nc1ccc(O)cc1" \
  --mcs-mode auto \
  --output output_predictions
```

Covalent docking:

```bash
uv run anchor-dock covalent \
  --protein protein.pdb \
  --query "C=CC(=O)NCC" \
  --reactive-residue CYS145:A \
  --optimize
```

Free local search:

```bash
uv run anchor-dock free \
  --protein pocket.pdb \
  --query "CCO" \
  --center 12 -3 8 \
  --box-size 20 20 20
```

Batch execution:

```bash
uv run anchor-dock batch examples/batch/jobs.jsonl \
  --output screen \
  --resume
```

`free` defaults to `--optimize`; pass `--no-optimize` to skip local refinement. Single-molecule inputs (`reference`, `covalent`, `free`) require an SDF file with exactly one record; multi-ligand SDF files must go through `batch`.

Each command prints JSON. A batch command exits nonzero when any recorded result failed. Run `anchor-dock <mode> --help` for all options.

The legacy flags `--weight-preset`, `--no-mmff`, and `--free-mcs` warn and translate for one release. `vina_lp` fails instead of being silently remapped.

## MCS modes

- `auto`: choose cross-fragment anchors only when they cover more atoms; otherwise preserve all distinct contiguous placements when needed.
- `single`: one deterministic largest contiguous placement.
- `multi`: bounded distinct contiguous occurrences and symmetry correspondences.
- `cross`: bounded non-overlapping fragment combinations, including alternative linker-cut packings.

MCS timeouts and hard node-budget exhaustion discard partial results and fail. Configured candidate/decomposition caps instead return deterministic bounded candidates with false proof flags. Mapping failures retain mapping pairs, original selection index, and seed. `MCS_Candidate_Complete`, `MCS_Max_Size_Proven`, and `MCS_Candidate_Limit` distinguish exhaustive candidate sets from deterministic capped subsets. Because fragment decomposition is bounded, any nonempty cross search—including `auto` that ultimately resolves to `single` or `multi`—records the overall proof flags as false even when all placements of its chosen patterns were enumerated. `max_mappings` is limited to 4096.

## Reading output

Use `AnchorDock_Score` for sorted reported scores and `AnchorDock_Search_Energy` for the objective. Keep scorer/receptor/source fingerprints, units, semantics, intramolecular reference, package/schema version, and `Search_Parameters`. Requested/applied torsion and optimization fields are separate. Covalent output records atom-pair bond-target source/bounds, 1–3 geometry, the reactant receptor fingerprint, and the versioned product-state nucleophile typing change.

Free mode samples seeded Haar-uniform SO(3) starting orientations. Ligands and references must be single connected components; salts and mixtures fail explicitly.

## Verification

```bash
uv lock --check
uv run --frozen ruff check src tests examples
uv run --frozen pytest -q
uv build --wheel
```
