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

The old namespaces remain importable and delegate to the unified implementation.

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
