# Migration

## 0.6.2 CPU validation and prepared-state ownership

The scorer catalog and normal-input energy formulas are unchanged. Vina/Vinardo factories snapshot caller-supplied configs; preparation snapshots config, receptor coordinates, features and masks. Later mutations of those input objects do not change an already prepared scorer. Factory fingerprints reflect the current configuration and actual weight evaluation order; preparing again captures a new independent state.

Torch AMBER omits zero-strength Lennard-Jones terms before computing powers, preserving finite Coulomb-only energy and derivatives for finite supported parameters. This fixes cached and tiled evaluation without changing ordinary force-field parameters.

The package version is `0.6.2`, batch resume epoch is `8`, ranked docking SDF schema remains `4`, and standalone SF schemas remain `anchor-dock-sf-refinement-v1` / `anchor-dock-sf-comparison-v1`. Older batch records rerun. See the [current CPU validation report](../reports/cpu-validation-2026-10-30.md) and [current scoring guide](SCORING_FUNCTIONS.md).

## 0.6.1 Torch scoring focus

The current catalog focuses on Vina/Vinardo, explicit AD4/AutoDock-GPU map references and pure Torch AMBER. The named `amber` entry now constructs `TorchAmberScorer`: it imports an explicitly parameterized AMBER topology through OpenMM once, then evaluates energy and gradients in Torch. It supports nonperiodic vacuum NoCutoff systems and rejects GB, PBC and PME. There is no separate `amber-torch` alias.

Code that previously selected `scorer="amber"` for an OpenMM Context must now pass an explicit `OpenMMScorer` object to retain that behavior. `OpenMMScorer.from_amber(..., implicit_solvent="GBn2")` retains its supported advanced options; they are not supported by the named Torch `amber` entry.

Public Smina/GNINA/native Vina local-optimizer adapters and all three DKOES presets have been removed. Requests for those names fail explicitly. Remove them from scorer sweeps; there is no scientifically equivalent automatic replacement. The prepared native Vina/Vinardo FAST-table final-scoring reference remains separate from native optimizer execution. `NeuralScorerAdapter` and explicit scorer registration remain available for custom extensions.

`adgpu` selects the declared AutoDock-GPU map/pair reference formulation; it does not run or reproduce native AutoDock-GPU search. Both map routes require prepared ligand chemistry and existing receptor maps. Changing receptor charges requires regenerating those maps.

Batch resume epoch advances to `7`; prior successful artifacts are not reused. Ranked docking output schema remains `4`, and independent SF refinement keeps its own result schemas. Dated 0.6 reports document historical adapters and validation rather than the current catalog.

## 0.6 scoring functions (historical transition)

SoftDock is removed and requests for its name fail explicitly. Interaction docking defaults to Vina. Batch resume epoch advances to 6 so older default-scorer artifacts are not reused. Ranked docking output schema remains 4.

`refine_pose`, `refine_with_scorers`, `anchor-dock scorers` and `anchor-dock refine` provide independent SF refinement of an existing positioned 3D pose. Standalone results use their own versioned result schemas. Current preparation, AMBER dependencies and plugin contracts are documented in [SCORING_FUNCTIONS.md](SCORING_FUNCTIONS.md).

## Migrating from 0.4 to 0.5

AnchorDock 0.5 keeps every valid 0.4 single-interaction call source-compatible. The five keyword arguments normalize internally to a one-item constraint list. New simultaneous constraints use the canonical ordered form:

```python
interactions = [
    {
        "receptor_residue": "ASP189:A",
        "receptor_atom": "OD1",
        "ligand_smarts": "[N:1]",
        "target_distance": 3.0,
        "distance_tolerance": 0.5,
    },
    {
        "receptor_residue": "SER190:A",
        "receptor_atom": "OG",
        "ligand_smarts": "[O:1]",
        "target_distance": 2.9,
        "distance_tolerance": 0.4,
        "restraint_weight": 12.0,
    },
]
```

Supplying this list together with any single-interaction selector/geometry keyword is an error. The CLI equivalent is one JSON array passed to `--interactions-json`; do not combine it with the single-selector flags.

All items are simultaneous `ALL`/`AND` constraints. Use separate jobs for `OR`, `ANY`, or k-of-n alternatives. A list contains at most eight items. Each SMARTS still contains exactly one mapped `:1` atom and automatically enumerates at most `max_matches=16` distinct ligand atoms. The Cartesian joint assignments are bounded by `max_joint_matches=64`; neither cap silently truncates.

Multi runs score the union pocket around all selected residues. The interaction with the fewest ligand matches supplies primary placement, conservative pairwise geometry checks remove provably infeasible conformer/assignment pairs, and all secondary violations participate in preselection. Guidance averages per-item weighted penalties, release removes all penalties, final filtering requires every window, and ranking remains scorer-only. No receptor selection, interaction chemistry, protonation, or tautomer state is inferred.

Output schema advances to `4`, and the batch resume epoch advances to `5`. Multi-interaction provenance includes the ordered specifications, resolved receptor atoms, per-selector matches, selected joint assignment, and per-interaction initial/guided/final distances. Older successful batch artifacts are not reused.

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

