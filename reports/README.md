# Reports

This directory contains versioned validation evidence and historical progress notes. Each report states its input scope and numerical contract; historical timings and scores should not be mixed across contracts.

## Current contracts

- [Torch scoring-function guide](../docs/SCORING_FUNCTIONS.md): current 0.6.1 catalog, AMBER Torch/OpenMM distinction, map requirements and removed adapters.
- [Torch physical scoring](torch-physical-2026-10-30.md): pure Torch AMBER reference checks, sparse intramolecular evaluation, shared AD4 grid sampling and bounded performance evidence.

## Recorded validation

- [Historical 0.6 scoring-function integration](scoring-functions-2026-10-30.md): the earlier catalog/refinement and optional native/OpenMM execution. Its DKOES and native optimizer adapters were removed in 0.6.1; its test counts describe that recorded snapshot.

- [Implementation and validation](completion-2026-10-30.md): prepared input, scoring and optimizer completion, checks and limits.
- [Native Vina/Vinardo scoring](native-scoring-parity-2026-10-30.md): 48 same-preparation full-ligand comparisons.
- [AD4 scoring](ad4-native-scoring-2026-10-30.md): synthetic map and prepared intra checks against native Vina AD4.
- [Local pose recovery](redocking-2026-10-30.md): fixed-frame symmetry RMSD on three real complexes.
- [Initial scientific audit](scoring-audit-2026-10-30.md): atom-typing and pose-gradient corrections.
- [Historical performance-only check](performance-2026-10-30.md): predates scientific corrections.

## Archived summary

- [progress.md](progress.md): captured capabilities, results, and runtime notes

## Useful Cross-Links

- [Architecture](../docs/ARCHITECTURE.md)
- [Reference strategy](../docs/ARCHITECTURE.md#reference-strategy)

## Available Visual Assets

Primary GIFs already in the repository:

- `examples/10gs/visualizations/test_run.gif`
- `examples/10gs/visualizations/test_ref_run.gif`
- `examples/10gs/visualizations/mcs_q1_fixed.gif`
- `examples/10gs/visualizations/mcs_q1_free.gif`
- `examples/10gs/visualizations/coverage/`
- `examples/10gs/predictions/`

