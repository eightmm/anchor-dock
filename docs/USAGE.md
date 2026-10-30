# Usage

## Install

```bash
uv sync --frozen --group dev
```

Python 3.12 or newer is required. These commands run from a repository checkout; `uv run --frozen` uses its managed environment without shell activation. Examples use CPU explicitly. Input paths such as `protein.pdb` and `pocket.pdb` are placeholders for a correctly prepared user structure.

For Meeko preparation or AMBER topology import, install the relevant extras:

```bash
uv sync --frozen --extra prepare --extra amber --group dev
```

See [the SF guide](SCORING_FUNCTIONS.md) for preparation requirements. No Smina/GNINA/native-optimizer executable is used by the catalog. Optional `native-vina` supplies a numerical scoring oracle.

## CLI

Reference-guided docking:

```bash
uv run --frozen anchor-dock reference \
  --protein examples/10gs/10gs_pocket.pdb \
  --reference examples/10gs/10gs_ligand.sdf \
  --query "CC(=O)Nc1ccc(O)cc1" \
  --mcs-mode single --num-confs 16 --max-mappings 8 \
  --device cpu --seed 713 --quiet --output output_predictions
```

Covalent docking:

```bash
uv run --frozen anchor-dock covalent \
  --protein protein.pdb \
  --query "C=CC(=O)NCC" \
  --reactive-residue CYS145:A \
  --optimize --device cpu
```

Interaction-guided local search:

```bash
uv run --frozen anchor-dock interaction \
  --protein pocket.pdb \
  --query "CCO" \
  --receptor-residue ASP189:A \
  --receptor-atom OD1 \
  --ligand-smarts "[O:1]" \
  --target-distance 3.0 \
  --distance-tolerance 0.5 --device cpu
```

Multiple simultaneous constraints use one JSON array. Shell quoting matters, so wrap the JSON in single quotes:

```bash
uv run --frozen anchor-dock interaction \
  --protein pocket.pdb \
  --query "NCCO" --device cpu \
  --interactions-json '[{"receptor_residue":"ASP189:A","receptor_atom":"OD1","ligand_smarts":"[N:1]","target_distance":3.0,"distance_tolerance":0.5},{"receptor_residue":"SER190:A","receptor_atom":"OG","ligand_smarts":"[O:1]","target_distance":2.9,"distance_tolerance":0.4,"restraint_weight":12.0}]'
```

`--interactions-json` cannot be mixed with any of the five single-selector flags. Its ordered non-empty array accepts at most eight items. Every item supplies `receptor_residue`, `receptor_atom`, `ligand_smarts`, `target_distance`, and `distance_tolerance`, with optional `restraint_weight`.

Batch execution:

```bash
uv run --frozen anchor-dock batch examples/batch/jobs.jsonl \
  --output screen \
  --resume --device cpu
```

Existing-pose scoring and refinement:

```bash
uv run --frozen anchor-dock scorers
uv run --frozen anchor-dock refine \
  --protein examples/10gs/10gs_pocket.pdb \
  --ligand examples/10gs/10gs_ligand.sdf \
  --scorers vina vinardo --steps 20 --seed 713 --device cpu \
  --output sf_refinement
```

`refine` requires one existing 3D ligand in the receptor coordinate frame. It runs every requested function from the same source pose; `--steps 0` evaluates the prepared starting pose without optimizer steps. Preparation can add/merge H and round PDBQT coordinates, and those changes are recorded. SMILES/2D embedding by itself does not position a ligand near the receptor.

The complete seven-entry catalog is for existing-pose refinement and the prepared engine. Guided high-level docking commands expose inferred Vina/Vinardo scoring and compatible custom scorers; selecting AMBER or AD4 does not supply their missing topology/maps or transfer a covalent/reference constraint automatically. See [SF selection, constructor options and result schemas](SCORING_FUNCTIONS.md). `refine_with_scorers` sweeps functions for one pose; it is not a multi-ligand batch mode.

`interaction` defaults to a bounded search: 32 conformer attempts, 128 candidates, 16 preselected poses, 50 guided optimization steps, 25 restraint-free release steps, at most 16 distinct ligand anchors per selector, and at most 64 Cartesian joint assignments. Pass `--no-optimize` to retain only the seeded/preselected poses. Query ligands may still be passed as SMILES, InChI, or supported molecule files. When a ligand or reference input is an SDF, it must contain exactly one record; multi-ligand SDF files must go through `batch`.

The receptor residue selector uses forms such as `ASP189:A` or `ASP189A:A` (insertion code before the colon). Omitting the chain is allowed only when the remaining selector is unique; an explicit blank chain is written as `ASP189:`. The receptor atom name must resolve to exactly one non-hydrogen standard `ATOM` record. Alternate locations and ambiguous records are rejected.

Each ligand SMARTS must contain exactly one mapped query atom and that map number must be `:1`; for example, `[O:1]` selects oxygen atoms while `[CX3](=[O:1])` restricts the mapped oxygen to a carbonyl environment. Every distinct matching ligand atom is explored automatically. If there are no matches, any selector exceeds `--max-matches`, or the Cartesian product exceeds `--max-joint-matches`, the run fails instead of choosing or truncating silently. Exact duplicate interaction items are rejected, but the same resolved ligand atom may legitimately satisfy more than one receptor constraint.

Multiple list items mean `ALL`/`AND`; `OR`, `ANY`, and k-of-n alternatives must be separate jobs. The pocket is the union around all selected receptor residues. The selector with the fewest ligand matches supplies primary placement, conservative pairwise shell checks remove provably infeasible conformer/assignment pairs, and all secondary violations affect preselection. Guidance averages the per-item weighted flat-bottom penalties, then removes every restraint for release. Only poses inside every requested window are exported and ranked by the unmodified scorer.

Interaction guidance is a generic atom-pair distance hypothesis. It automatically enumerates ligand atoms selected by the supplied SMARTS, but does not discover receptor atoms, assign an interaction type, or infer protonation, tautomers, or chemical compatibility.

The six commands are `reference`, `covalent`, `interaction`, `batch`, `scorers` and `refine`. Each command prints JSON. A batch command exits nonzero when any recorded result failed. Run `uv run --frozen anchor-dock <command> --help` for all options.

The legacy flags `--weight-preset`, `--no-mmff`, and `--free-mcs` warn and translate for one release. `vina_lp` fails instead of being silently remapped.

## MCS modes

- `auto`: choose cross-fragment anchors only when they cover more atoms; otherwise preserve all distinct contiguous placements when needed.
- `single`: one deterministic largest contiguous placement.
- `multi`: bounded distinct contiguous occurrences and symmetry correspondences.
- `cross`: bounded non-overlapping fragment combinations, including alternative linker-cut packings.

MCS timeouts and hard node-budget exhaustion discard partial results and fail. Configured candidate/decomposition caps instead return deterministic bounded candidates with false proof flags. Mapping failures retain mapping pairs, original selection index, and seed. `MCS_Candidate_Complete`, `MCS_Max_Size_Proven`, and `MCS_Candidate_Limit` distinguish exhaustive candidate sets from deterministic capped subsets. Because fragment decomposition is bounded, any nonempty cross search—including `auto` that ultimately resolves to `single` or `multi`—records the overall proof flags as false even when all placements of its chosen patterns were enumerated. `max_mappings` is limited to 4096.

## Reading output

For ranked docking SDFs, use `AnchorDock_Score` for sorted reported scores and `AnchorDock_Search_Energy` for the scorer's objective. Keep scorer/receptor/source fingerprints, units, semantics, intramolecular reference, package/schema version, and `Search_Parameters`. Requested/applied torsion and optimization fields are separate. Covalent output records atom-pair bond-target source/bounds, 1–3 geometry, the reactant receptor fingerprint, and the versioned product-state nucleophile typing change. Interaction output separately records the resolved receptor atom, all ligand matches, target window, initial/guided/final distances, restraint values, and guide/release statistics.

Standalone refinement writes `refined.sdf` and `result.json` for each successful function, plus `comparison.json` for a sweep. Those files use their own versioned SF schemas and per-metric units/directions; they are not ranked docking schema `4`. Existing result files are not overwritten.

Interaction mode samples seeded Haar-uniform SO(3) starting orientations around the selected receptor atom for a single run, or around the deterministic primary receptor atom for a multi run. Ligands and references must be single connected components; salts and mixtures fail explicitly.

## Verification

The [current CPU validation report](../reports/cpu-validation-2026-10-30.md) records executed CLI/API workflows and numerical checks. Reproduce the required checks from a clean checkout:

```bash
uv lock --check
uv run --frozen ruff check src tests examples
uv run --frozen pytest -q
uv run --frozen --all-extras pytest -q
uv build --wheel
```
