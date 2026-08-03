# PROJECT.md

Project-specific decisions and verification contract.

## Status

- State: confirmed
- Open decisions: none

## Project

- Type: general
- Goal: Replace the public unconstrained `free` search with a bounded, reproducible interaction-guided local docking mode.
- Scope:
  - Public names are `dock_interaction`, CLI/batch mode `interaction`, and `DockingJob.interaction`.
  - Require an explicit receptor residue, receptor atom, atom-mapped ligand SMARTS, target distance, and tolerance.
  - Enumerate distinct ligand `:1` atom matches automatically; reject zero matches and fail when the configured cap would truncate them.
  - Coarse-score bounded seeded candidates, select fairly across matches/conformers, run guided SE(3)+torsion optimization, then a restraint-free release phase.
  - Filter final poses by the requested distance window and rank/report only the scorer's unmodified score.
  - Remove the former free API, CLI, batch factory/enum, module, examples, and current documentation; retain historical changelog text.
  - Publish this breaking contract as 0.4.0 with SDF output schema 3 and batch resume epoch 4.
- Non-goals:
  - No global docking completeness claim.
  - No automatic residue or receptor-atom choice.
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
  - Initial candidates are deterministic and place each selected ligand atom at the requested target distance.
  - Only preselected candidates enter optimization; guided restraint energy is never included in `AnchorDock_Score` or `AnchorDock_Search_Energy`.
  - Exported poses satisfy the requested distance window after the restraint-free release phase; an all-invalid run errors explicitly.
  - Batch signatures include every interaction selector/geometry field and cannot resume artifacts from the prior epoch.
  - Package metadata, runtime version, lockfile, wheel, CLI, docs, and CI agree on 0.4.0 and the interaction API.
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
  - Defaults are bounded for local search: 32 conformer attempts, 128 candidates, 16 preselected poses, 50 guided steps, 25 release steps, and at most 16 distinct ligand anchors.
  - Reuse one live pose module for guide and release phases so rigid-rotation parameters and their gradients are preserved.
  - Score the residue-centered pocket (12 A default cutoff), not an unrestricted full-receptor/random-box search.
  - A guidance weight is in scorer-energy units per squared angstrom and is recorded separately from scorer provenance.
- Do not touch:
  - The reference-mode `--free-anchor` / `--free-mcs` compatibility flag or historical 0.3 changelog entries.
  - User inputs, credentials, generated result directories, or unrelated repository work.
- Risks:
  - Distance-only constraints can admit chemically implausible angles; output must call them generic atom-pair distance hypotheses.
  - Broad SMARTS can create many hypotheses; bounded enumeration must error rather than silently truncate.
  - PDB alternate locations and ambiguous chain/insertion identifiers must be rejected rather than guessed.
  - Restraint-dominated objectives can conceal poor physical poses; the release phase, hard final filter, and unmodified score reporting are mandatory.
