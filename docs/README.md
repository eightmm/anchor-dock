# Documentation

The current implementation provides bounded guided docking and independent existing-pose refinement with Torch physical/empirical scoring functions.

| Guide | Canonical content |
|---|---|
| [USAGE.md](USAGE.md) | Installation, CLI workflows and output interpretation |
| [API_REFERENCE.md](API_REFERENCE.md) | Supported public Python interfaces and result fields |
| [SCORING_FUNCTIONS.md](SCORING_FUNCTIONS.md) | Named Torch catalog, preparation, map/topology requirements and plugin registration |
| [SCORING.md](SCORING.md) | Energy equations, atom typing, reporting and derivative contracts |
| [PREPARED.md](PREPARED.md) | Graph/PDBQT atom identity, authoritative branches and AutoGrid conventions |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Module boundaries, strategy invariants and independent-refinement architecture |
| [BATCH.md](BATCH.md) | Manifest inputs, mixed jobs, ownership, errors and resume |
| [MIGRATION.md](MIGRATION.md) | Current compatibility changes and earlier removed APIs |

[Examples](../examples/README.md) distinguish bundled runnable checks from user-input templates. [Reports](../reports/README.md) separate current validation from historical measurements. [ANCHOR_DOCK_ARCHITECTURE.md](ANCHOR_DOCK_ARCHITECTURE.md) is a compatibility link to the canonical architecture guide.

The seven named entries are `vina`, `vinardo`, `ad4`, `adgpu`, `amber`, `vina-final` and `vinardo-final`. The last two are nondifferentiable table references; `adgpu` does not launch AutoDock-GPU. The named `amber` implementation evaluates a supported vacuum potential in Torch after parameter import. The explicit OpenMM object remains a distinct reference/advanced backend. Historical Smina/GNINA optimizer results do not describe current functionality.

[Latest CPU validation](../reports/cpu-validation-2026-10-30.md) records the current release checks; older reports retain their dated measurements.
