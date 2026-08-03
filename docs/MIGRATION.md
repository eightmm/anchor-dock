# Migration

## Migrating from 0.3 to 0.4

AnchorDock 0.4 removes unconstrained local search from the public Python API, CLI, batch enum/factory, examples, and current documentation. Replace it with the explicit interaction-guided interface only when the scientific input includes all five required fields:

| 0.4 field | Meaning |
|---|---|
| `receptor_residue` | exact residue selector, for example `ASP189:A` |
| `receptor_atom` | exact non-hydrogen standard-PDB atom name |
| `ligand_smarts` | SMARTS containing exactly one mapped `:1` query atom |
| `target_distance` | positive receptor/ligand atom distance in Å |
| `distance_tolerance` | positive half-width smaller than the target distance |

There is no automatic migration from a box/center search because choosing the residue, receptor atom, ligand chemical group, and distance hypothesis is a scientific decision. Public names are now `dock_interaction`, CLI/batch mode `interaction`, and `DockingJob.interaction`.

Distinct ligand atoms selected by `:1` are enumerated automatically. A zero-match SMARTS fails, and a match count beyond `max_matches` fails instead of silently selecting the first or truncating. The guide is a generic atom-pair distance restraint; it does not claim a hydrogen bond, salt bridge, metal interaction, pi interaction, protonation state, tautomer, or compatible chemistry.

The first optimization phase combines scorer search energy with a flat-bottom distance guide. The same live pose model then runs a restraint-free release phase. Exported poses must satisfy the hard final distance window, and `AnchorDock_Score` plus `AnchorDock_Search_Energy` contain only the unmodified scorer values. Output schema advances to `3`, and the batch resume epoch advances to `4`; older successful artifacts are not reusable under the new contract.

## Migrating from 0.2 to 0.3

AnchorDock 0.3 is one engine and one import namespace: `anchor_dock`. The old `lig_align` and `cov_vina` packages and their low-level façade modules were removed.

## One-release call adapters

The following unambiguous calls still execute through the 0.3 engine and emit one `FutureWarning` per call:

| 0.2 call | 0.3 destination |
|---|---|
| `run_reference_pipeline`, `reference.run_pipeline` | `dock_reference` |
| `run_covalent_pipeline` | `dock_covalent` |
| `dock_reference_batch`, `reference.run_batch` | `dock_batch(mode="reference")` |
| `dock_covalent_batch`, `run_batch_docking` | `dock_batch(mode="covalent")` |
| `ref_ligand` | `reference_ligand` |
| `mmff_optimize` | `relax` |
| `freeze_mcs` | `freeze_anchor` |
| `weight_preset="vina"` or `"vinardo"` | `scorer` |
| `save_all_poses=False` / `True` | `top_k=3` / `None` |

Passing both an old and new keyword fails with `ValueError`. The old reference/covalent default directory and filenames remain for this adapter release.
Legacy high-level positional options after `output_dir` are also bound using the 0.2 parameter order and warn; new code should pass every option by keyword.

## Intentional hard breaks

- 0.3 never invokes the 0.2 scorer, atom typer, optimizer, or output writer.
- `LigAlign_*`, `CovVina_*`, `Vina_Score`, and `Rank` tags are replaced by `AnchorDock_*` fields.
- Scores changed because atom typing, radii, cutoff, pair exclusions, intramolecular reference, and reporting changed. Do not compare 0.2 and 0.3 numbers as one scale.
- `vina_lp` had no validated provenance and now fails explicitly.
- `LigandAligner`, `final_selection`, manual pocket-cache hooks, and the old `scripts/` entry points were removed. Use the four high-level APIs or `DockingEngine`.
- Batch directories, signatures, resume manifests, and error payloads follow the 0.3 batch contract; aliases translate calls, not old on-disk layouts.
- Multi-component ligands/references now fail explicitly. Apply a documented desalting policy before calling AnchorDock if that is scientifically intended.

## Covalent default

For this transition release, omitting covalent `optimize` preserves the 0.2 value `False` and emits a warning. Pass `optimize=True` or `False` explicitly. The CLI equivalent is `--optimize` or `--no-optimize`.

## Output migration

Read scores with `AnchorDock_Score`, distinguish the objective with `AnchorDock_Search_Energy`, and persist scorer/receptor/source fingerprints, intramolecular reference, requested/applied torsion and optimization fields, `AnchorDock_Search_Parameters`, version, and output schema. Output schema `2` records the true RDKit conformer ID in `AnchorDock_Source_Conformer` and keeps the representative ordinal in `AnchorDock_Source_Representative_Index`. Covalent results also distinguish the reactant receptor fingerprint from the product-state scoring fingerprint and record the versioned receptor typing change.
