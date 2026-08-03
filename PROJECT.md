# PROJECT.md

Project-specific decisions and verification contract.

## Status

- State: confirmed
- Open decisions: none

## Project

- Type: general
- Goal: Provide bounded, reproducible single- and multi-interaction-guided local docking in place of the former unconstrained `free` search.
- Scope:
  - Public names are `dock_interaction`, CLI/batch mode `interaction`, and `DockingJob.interaction`.
  - Require an explicit receptor residue, receptor atom, atom-mapped ligand SMARTS, target distance, and tolerance.
  - Accept a canonical non-empty `interactions` list for simultaneous `ALL`/`AND` constraints while preserving the existing five single-interaction keyword arguments as a compatibility form. Supplying both forms is an error.
  - Each list item independently supplies the same five fields and may override `restraint_weight`; exact duplicate items are rejected, while one ligand atom may legitimately satisfy more than one receptor constraint.
  - Enumerate distinct ligand `:1` atom matches automatically; reject zero matches and fail when the configured cap would truncate them.
  - Accept at most eight simultaneous interaction items, form deterministic Cartesian joint-match hypotheses, fail above `max_joint_matches`, and retain only conformer/hypothesis pairs that pass conservative pairwise shell-distance feasibility checks.
  - Choose the interaction with the fewest ligand matches as the deterministic primary placement anchor. Secondary constraints must affect coarse preselection through their flat-bottom violations; quotas remain fair across viable joint hypotheses and conformers.
  - Coarse-score bounded seeded candidates, select fairly across matches/conformers, run guided SE(3)+torsion optimization, then a restraint-free release phase.
  - Average the per-interaction weighted flat-bottom penalties during guidance, preserving the existing one-interaction objective exactly. Filter final poses by every requested distance window and rank/report only the scorer's unmodified score.
  - Score the union pocket around every selected receptor residue and key its bounded cache by the complete ordered receptor selector set.
  - Remove the former free API, CLI, batch factory/enum, module, examples, and current documentation; retain historical changelog text.
  - Publish multi-interaction support as 0.5.0 with SDF output schema 4 and batch resume epoch 5; 0.4 single-interaction calls remain source-compatible.
- Non-goals:
  - No global docking completeness claim.
  - No automatic residue or receptor-atom choice.
  - No `OR`, `ANY`, or k-of-n constraint semantics; alternatives are separate batch jobs.
  - No exact analytical multi-sphere intersection solver or global geometric completeness claim.
  - No claim that a distance-only guide is a hydrogen bond, salt bridge, metal interaction, or pi interaction.
  - No protonation, tautomer, or chemical-compatibility inference.

## Commands

- Setup: `uv sync --frozen --group dev`
- Test: `uv run --frozen pytest -q`
- Run: `uv run --frozen anchor-dock interaction --help`
- Lint/typecheck: `uv run --frozen ruff check src tests examples`

## Paths

- Data: user-provided PDB receptor and SMILES/InChI/single-molecule ligand files; bundled examples under `examples/`.
- Config: `pyproject.toml`; interaction options are explicit API/CLI/batch fields and recorded search parameters.
- Outputs/logs: per-run SDF/JSON outputs below the requested output directory; batch manifests are `results.jsonl` and `summary.json`.
- Checkpoints: n/a; docking has no learned-state checkpoint.

## Verification

- Success criteria:
  - `free`, `dock_free`, and `DockingJob.free` are absent from current public code/docs/examples/CI; the unrelated reference aliases `--free-anchor` and `--free-mcs` remain.
  - Receptor and ligand selectors fail closed on invalid, absent, ambiguous, hydrogen, or over-cap inputs.
  - Initial candidates are deterministic; single runs place their selected ligand atom at the target distance, while multi runs place the primary atom there and include all secondary violations in preselection.
  - Only preselected candidates enter optimization; guided restraint energy is never included in `AnchorDock_Score` or `AnchorDock_Search_Energy`.
  - Exported poses satisfy the requested distance window after the restraint-free release phase; an all-invalid run errors explicitly.
  - Batch signatures include the canonical ordered interaction list, its per-item weights, and every joint-search bound, and cannot resume artifacts from the prior epoch.
  - Package metadata, runtime version, lockfile, wheel, CLI, docs, and CI agree on 0.5.0 and the single/multi interaction API.
- Required checks:
  - Focused interaction, CLI, batch, output, and compatibility tests.
  - Full `uv run --frozen pytest -q`.
  - `uv run --frozen ruff check src tests examples`.
  - `uv lock --check` and `uv build --wheel`.
  - Installed-wheel import/CLI smoke test and GitHub Actions after push.

## Notes

- Gotchas (non-obvious decisions an agent would get wrong):
  - The ligand SMARTS must contain exactly one mapped query atom and its map number must be `:1`; matching is against the canonical heavy-atom ligand.
  - Multiple distinct mapped ligand atoms are hypotheses to enumerate, not an ambiguity to resolve by choosing the first.
  - Defaults are bounded for local search: 32 conformer attempts, 128 candidates, 16 preselected poses, 50 guided steps, 25 release steps, at most 16 distinct ligand anchors per selector, at most 64 joint hypotheses, and at most eight simultaneous interactions.
  - Reuse one live pose module for guide and release phases so rigid-rotation parameters and their gradients are preserved.
  - Score the union of residue-centered pockets (12 A default cutoff), not an unrestricted full-receptor/random-box search.
  - A guidance weight is in scorer-energy units per squared angstrom and is recorded separately from scorer provenance.
- Do not touch:
  - The reference-mode `--free-anchor` / `--free-mcs` compatibility flag or historical 0.3 changelog entries.
  - User inputs, credentials, generated result directories, or unrelated repository work.
- Risks:
  - Distance-only constraints can admit chemically implausible angles; output must call them generic atom-pair distance hypotheses.
  - Broad SMARTS can create many hypotheses; bounded enumeration must error rather than silently truncate.
  - Match counts multiply across interaction items; `max_matches` applies per item and `max_joint_matches` applies to their Cartesian product before candidate allocation.
  - Multi-interaction feasibility pruning is conservative and pairwise: passing it does not claim the full constraint set is realizable, while failing it must never discard a realizable rigid conformer.
  - Current single-selector flags remain available. Multi-interaction CLI input is one JSON array via `--interactions-json`; mixing it with any single-selector flag fails closed.
  - For multi runs, SDF and result provenance include the ordered interaction specifications, resolved receptor atoms, per-selector ligand matches, selected joint hypothesis, and per-interaction initial/guided/final distances.
  - PDB alternate locations and ambiguous chain/insertion identifiers must be rejected rather than guessed.
  - Restraint-dominated objectives can conceal poor physical poses; the release phase, hard final filter, and unmodified score reporting are mandatory.
