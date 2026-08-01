# Migration to AnchorDock

## Repository and package identity

`lig-mcs-align` is now AnchorDock. The distribution name is `anchor-dock` and the primary import namespace is `anchor_dock`.

## API mapping

| Previous API | Preferred AnchorDock API |
|---|---|
| `lig_align.run_pipeline(...)` | `anchor_dock.dock_reference(...)` |
| `lig_align.run_batch(...)` | `anchor_dock.dock_reference_batch(...)` |
| `cov_vina.run_covalent_pipeline(...)` | `anchor_dock.dock_covalent(...)` |
| `cov_vina.run_batch_docking(...)` | `anchor_dock.dock_covalent_batch(...)` |

Both old namespaces have been removed; `anchor_dock` is the only import path.

`cov_vina` was pure re-export, so the mappings above replace it exactly. `lig_align` held the real reference-mode implementation, which moved rather than disappeared:

| Previous module | Now |
|---|---|
| `lig_align.pipeline` | `anchor_dock.reference.pipeline` |
| `lig_align.aligner` | `anchor_dock.reference.aligner` |
| `lig_align.molecular.mcs` | `anchor_dock.reference.mcs` |
| `lig_align.molecular.conformer` | `anchor_dock.reference.conformers` |
| `lig_align.molecular.relax` | `anchor_dock.reference.relax` |
| `lig_align.io.input`, `lig_align.io.pocket` | `anchor_dock.reference.io` |
| `lig_align.selection.final_selection` | `anchor_dock.reference.output` |
| `lig_align.io.visualization` | `anchor_dock.reference.visualization` |
| `lig_align.scoring`, `.alignment`, `.optimization`, `.molecular.features` | `anchor_dock.core` (these were already re-exports) |

The move was verified to leave reference-mode output byte-identical across MCS modes, weight presets, and torsion optimization; `tests/test_reference_regression.py` keeps it that way.

SDF property names still carry the `LigAlign_` prefix (`LigAlign_MCS_Mode`, `LigAlign_MMFF_Optimized`, and so on). Renaming them would break anyone parsing previously written poses, so they are left alone.

## Behavior retained

Reference mode retains MCS `auto`, `single`, `multi`, and `cross` modes, optional MMFF relaxation, all-position pooling, differentiable Vina scoring, and torsion optimization.

Covalent mode retains automatic warhead detection, CYS/SER/THR/TYR/LYS/HIS anchors, adduct-first conformer generation, anchor-axis rotation scanning, pair exclusions near the formed bond, pocket caching, and batch execution.

## Intentional corrections

- One scoring implementation now serves both modes.
- Single- and multi-pose kinematics use one implementation.
- Pair masks accept either `[N,M]`, `[1,N,M]`, or `[B,N,M]` shapes consistently.
- PDB residue metadata is copied explicitly when extracting a pocket.
- Covalent scoring excludes pseudo protein atoms and the duplicated receptor nucleophile without deleting the reactive ligand atom's interactions with the rest of the pocket.
- Covalent outputs state that scores are non-bonded pose scores conditioned on an already-formed adduct.
