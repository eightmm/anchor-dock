# Validation reports

## Current implementation: 0.6.2

- [CPU validation and documentation audit](cpu-validation-2026-10-30.md): current executable checks, scorer fixes, numerical oracles, workflows, package verification and measured CPU limits.
- [Machine-readable CPU evidence](cpu-validation-2026-10-30.json): source/input hashes, commands, versions and result summaries.
- [Scoring-function guide](../docs/SCORING_FUNCTIONS.md): current seven-entry Torch catalog, preparation and optimization contracts.

The current catalog uses Vina/Vinardo empirical objectives, AD4/ADGPU map formulations, a pure Torch AMBER vacuum potential and two separate table references. Native Smina/GNINA/Vina optimizer adapters and DKOES presets were removed. An explicit OpenMM reference/advanced object remains available. Batch epoch is 8; ranked SDF schema is 4.

## Historical evidence

These records retain their original inputs, source identities and numeric results. They do not certify newer releases, and scores/timings from different contracts must not be combined.

- [0.6.1 Torch physical scoring](torch-physical-2026-10-30.md) and [raw evidence](torch-physical-2026-10-30.json): pure Torch AMBER, static pair caching, shared AD4 map geometry and opt-in sparse intra evaluation.
- [0.6.0 scoring-function integration](scoring-functions-2026-10-30.md): includes native adapters and DKOES presets subsequently removed in 0.6.1.
- [Implementation completion](completion-2026-10-30.md): prepared inputs, scoring and geometry-aware optimization at the recorded commit.
- [Native Vina/Vinardo parity](native-scoring-parity-2026-10-30.md): same-preparation ligand score comparisons.
- [AD4 native reference](ad4-native-scoring-2026-10-30.md): map and prepared intra comparisons with native Vina AD4.
- [Pose-recovery experiments](redocking-2026-10-30.md): local fixed-frame symmetry RMSD on three complexes, not a broad docking benchmark.
- [Scientific scoring audit](scoring-audit-2026-10-30.md): atom typing and pose-gradient corrections.
- [Early performance experiment](performance-2026-10-30.md): predates later scientific scoring corrections.
- [Progress snapshot](progress.md): archived project notes.

## Related documentation

- [Architecture](../docs/ARCHITECTURE.md)
- [User guide](../docs/USAGE.md)
- [API reference](../docs/API_REFERENCE.md)
- [Migration history](../docs/MIGRATION.md)
- [Examples and validation tools](../examples/README.md)
