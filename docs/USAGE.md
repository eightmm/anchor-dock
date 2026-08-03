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

Interaction-guided local search:

```bash
uv run anchor-dock interaction \
  --protein pocket.pdb \
  --query "CCO" \
  --receptor-residue ASP189:A \
  --receptor-atom OD1 \
  --ligand-smarts "[O:1]" \
  --target-distance 3.0 \
  --distance-tolerance 0.5
```

Batch execution:

```bash
uv run anchor-dock batch examples/batch/jobs.jsonl \
  --output screen \
  --resume
```

`interaction` defaults to a bounded search: 32 conformer attempts, 128 candidates, 16 preselected poses, 50 guided optimization steps, 25 restraint-free release steps, and at most 16 distinct ligand anchor atoms. Pass `--no-optimize` to retain only the seeded/preselected poses. Query ligands may still be passed as SMILES, InChI, or supported molecule files. When a ligand or reference input is an SDF, it must contain exactly one record; multi-ligand SDF files must go through `batch`.

The receptor residue selector uses forms such as `ASP189:A` or `ASP189A:A` (insertion code before the colon). Omitting the chain is allowed only when the remaining selector is unique; an explicit blank chain is written as `ASP189:`. The receptor atom name must resolve to exactly one non-hydrogen standard `ATOM` record. Alternate locations and ambiguous records are rejected.

The ligand SMARTS must contain exactly one mapped query atom and that map number must be `:1`; for example, `[O:1]` selects oxygen atoms while `[CX3](=[O:1])` restricts the mapped oxygen to a carbonyl environment. Every distinct matching ligand atom is explored automatically. If there are no matches, or the full match set would exceed `--max-matches`, the run fails instead of choosing or truncating silently.

Interaction guidance is a generic atom-pair distance hypothesis. It does not discover the receptor/ligand atoms, assign an interaction type, or infer protonation, tautomers, or chemical compatibility. The flat-bottom restraint guides the first optimization phase, is removed for release, and is never added to reported scorer values. Only released poses inside the specified distance window are exported.

Each command prints JSON. A batch command exits nonzero when any recorded result failed. Run `anchor-dock <mode> --help` for all options.

The legacy flags `--weight-preset`, `--no-mmff`, and `--free-mcs` warn and translate for one release. `vina_lp` fails instead of being silently remapped.

## MCS modes

- `auto`: choose cross-fragment anchors only when they cover more atoms; otherwise preserve all distinct contiguous placements when needed.
- `single`: one deterministic largest contiguous placement.
- `multi`: bounded distinct contiguous occurrences and symmetry correspondences.
- `cross`: bounded non-overlapping fragment combinations, including alternative linker-cut packings.

MCS timeouts and hard node-budget exhaustion discard partial results and fail. Configured candidate/decomposition caps instead return deterministic bounded candidates with false proof flags. Mapping failures retain mapping pairs, original selection index, and seed. `MCS_Candidate_Complete`, `MCS_Max_Size_Proven`, and `MCS_Candidate_Limit` distinguish exhaustive candidate sets from deterministic capped subsets. Because fragment decomposition is bounded, any nonempty cross search—including `auto` that ultimately resolves to `single` or `multi`—records the overall proof flags as false even when all placements of its chosen patterns were enumerated. `max_mappings` is limited to 4096.

## Reading output

Use `AnchorDock_Score` for sorted reported scores and `AnchorDock_Search_Energy` for the scorer's objective. Keep scorer/receptor/source fingerprints, units, semantics, intramolecular reference, package/schema version, and `Search_Parameters`. Requested/applied torsion and optimization fields are separate. Covalent output records atom-pair bond-target source/bounds, 1–3 geometry, the reactant receptor fingerprint, and the versioned product-state nucleophile typing change. Interaction output separately records the resolved receptor atom, all ligand matches, target window, initial/guided/final distances, restraint values, and guide/release statistics.

Interaction mode samples seeded Haar-uniform SO(3) starting orientations around the selected receptor atom. Ligands and references must be single connected components; salts and mixtures fail explicitly.

## Verification

```bash
uv lock --check
uv run --frozen ruff check src tests examples
uv run --frozen pytest -q
uv build --wheel
```
