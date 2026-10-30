# Architecture

This is the canonical architecture guide for the current Torch scoring implementation. Public signatures and executable examples are maintained in [API_REFERENCE.md](API_REFERENCE.md) and [USAGE.md](USAGE.md); scoring contracts are in [SCORING.md](SCORING.md).

## Entry points and boundaries

| Entry point | Preparation and placement | Optimization and result |
|---|---|---|
| `dock_reference` | Reference MCS mappings and constrained conformers | Anchor-preserving torsion refinement; ranked docking output |
| `dock_covalent` | Explicit receptor nucleophile and supported warhead product graph | Product geometry and exclusions, then movable torsions; ranked docking output |
| `dock_interaction` | One to eight explicit atom-pair distance hypotheses and seeded local placement | Guided SE(3)+torsion search, restraint-free release, then hard distance filtering |
| `dock_batch` | Input/manifest normalization and bounded job dispatch | The selected strategy's result, with per-job provenance and resume validation |
| `refine_pose` / `refine_with_scorers` | Existing receptor-aligned 3D pose and scorer-specific preparation | Shared pose optimizer; separate per-SF results without cross-SF raw ranking |

The three docking strategies use `DockingEngine` for scorer-independent preparation, optimization and reporting. The existing-pose refinement workflow follows the scorer's declared preparation profile. Its AMBER path validates a complete all-atom topology and uses its fixed receptor directly; it does not run inferred XS preparation through `DockingEngine.prepare`.

`PreparedDockingProblem` contains the RDKit molecule, receptor context, prepared scorer, anchor indices, rotor count and optional prepared-input fingerprint. Initial coordinates are passed to optimization methods. Exclusion masks and derived pair data belong to the prepared scorer rather than separate problem fields.

## Module ownership

| Module | Responsibility |
|---|---|
| `core/engine.py` | Prepared problems, strategy-facing optimization and reporting |
| `core/kinematics.py`, `core/topology.py` | Rigid frames, SO(3) rotations, torsions and anchor restrictions |
| `core/optimization.py` | Shared Adam/AdamW/LBFGS, Cartesian pullback and pattern search |
| `core/scoring.py` | Vina/Vinardo equations, prepared pair data, score reporting and custom Torch adapters |
| `core/registry.py` | Named scorer factories, capability/preparation declarations and explicit plugin loading |
| `core/prepared.py`, `preparation.py` | Authoritative graph/PDBQT mapping and optional Meeko PDB/PQR/ligand preparation |
| `core/autogrid.py`, `core/grid.py`, `core/ad4.py` | AutoGrid ingestion, differentiable interpolation and declared AD4/ADGPU formulations |
| `core/torch_amber.py` | Parameter import and pure Torch fixed-charge vacuum potential |
| `core/amber.py` | Explicit optional OpenMM energy/force reference and advanced backend |
| `core/native_vina.py` | Nondifferentiable FAST-table reference scoring, without a native search process |
| `refinement.py` | Independent refinement, atom identity, metrics and provenance |
| `batch.py`, `core/output.py` | Job ownership/resume and strategy output contracts |

## Scoring and chemistry profiles

The catalog has seven entries: continuous `vina`, `vinardo`, `ad4`, `adgpu`, `amber`, plus pattern-only `vina-final` and `vinardo-final` table references. All named backends expose Torch objectives. `adgpu` describes the inspected map/pair formulation; it does not execute AutoDock-GPU. Public Smina/GNINA/native Vina optimizer adapters, DKOES presets and SoftDock are removed.

Three preparation profiles have different scientific contracts:

- **Inferred XS:** Vina/Vinardo can infer supported atom roles from the supplied graph/PDB state. Diagnostics and strict-mode rejection expose unresolved chemistry. This does not assign protonation or tautomers.
- **Prepared AD4:** An explicit chemical graph, atom mapping, types, charges and torsion tree are authoritative. Prepared Vina `num_tors` can be half-integral and differs from AD4 `TORSDOF`. AutoGrid maps are also authoritative; their association with a separately supplied receptor cannot be reconstructed from grid values alone.
- **AMBER topology:** The complete parameterized System, reference coordinates and explicit movable-particle indices define the potential. The named Torch backend supports nonperiodic vacuum NoCutoff bond/angle/periodic-torsion, Coulomb/LJ and exceptions. Unsupported force models fail explicitly. PQR charges alone are insufficient.

Native Vina bindings remain optional numerical oracles. `OpenMMScorer` is an explicit Python-object reference/advanced backend outside the named catalog. `NeuralScorerAdapter` remains a custom extension point; no built-in neural model or external optimizer is required by the physical catalog.

## Core invariants

1. Pose coordinates are `[B,N,3]` in angstroms; supported single-pose calls also accept `[N,3]`. Angles are radians.
2. Docking masks broadcast to the declared batch/atom axes. Vina-family intra masks exclude 1–2, 1–3, 1–4 and same-rigid-frame pairs. AMBER instead retains its topology's bonded terms and explicit nonbonded exclusions/scaled exceptions.
3. Inferred heavy-graph rotors are mapped back to the original atom order, so adding explicit H does not create terminal heavy-atom rotors. Prepared branches remain authoritative.
4. A torsion is disabled when fixed anchors occur on both graph sides. One live pose model is reused through interaction guide and release phases.
5. Shared optimization restores the best accepted state per pose. Early-stopped Adam rows cannot move through momentum. Cartesian and pattern trial-domain failures backtrack; an invalid initial potential still errors.
6. Search objectives are lower-is-better. Reported scores have their own units and direction. Vina-family reporting, AD4 unbound conventions and AMBER full-complex potential are not interchangeable.
7. Ranked strategy SDFs and independent-SF refinement outputs are separate contracts. Independent SF results never acquire a combined raw-score ranking.
8. Atom mappings, input/model hashes, seed and effective optimizer/runtime choices enter the appropriate result provenance. Prepared artifacts do not prove chemical correctness merely by being parseable.

## Strategy-specific contracts

### Reference

Reference mode evaluates bounded contiguous and alternative disjoint-fragment MCS candidates, quotients common pattern automorphisms, and selects `single`, `multi` or `cross` mappings deterministically. ETKDG coordinates use the absolute reference frame. Exact anchors and heavy-atom bond/1–3 geometry are checked before clustering; optional fixed-anchor relaxation and torsion refinement preserve those constraints. Pooled mappings share one final intramolecular reporting baseline. Candidate enumeration limits and global-maximum proof limits are reported separately.

### Covalent

Covalent mode resolves one receptor nucleophile and one supported warhead, applies a reaction-class topology transform, and introduces support/nucleophile atoms into the product topology. Validated formed-bond and 1–3 geometry, fixed protein support coordinates, product-state retyping and appropriate cross-pair exclusions constrain the remaining torsion search. Product-state features are copy-on-write so cached reactant contexts cannot be polluted. Generic distance restraints alone are not treated as covalent chemistry.

### Interaction

Interaction mode normalizes the five compatibility keywords or an ordered `interactions` list into one to eight simultaneous `ALL`/`AND` constraints. Mixed forms, exact duplicates, invalid/ambiguous receptor selectors and over-cap enumeration fail explicitly. Distinct SMARTS `:1` ligand atoms are capped by `max_matches`; their Cartesian joint assignments are capped by `max_joint_matches` rather than truncated.

For each conformer/assignment, conservative pairwise shell checks can reject an impossible rigid placement. Passing those checks is not a global feasibility proof. The constraint with the fewest ligand matches determines primary placement; seeded Haar-uniform orientations and all secondary violations affect bounded coarse preselection. Guidance adds the mean weighted flat-bottom penalties. Release removes every guide on the same live pose model. After output-coordinate quantization, every required distance window must pass before ranking by the unmodified scorer.

These are generic atom-pair distance hypotheses. They do not infer hydrogen bonds, salt bridges, metal/pi interactions, protonation, tautomers or chemical compatibility. `OR`, `ANY` and k-of-n alternatives are separate jobs.

## Receptor and parameter reuse

`ReceptorContext` stores coordinates, atom features, a scoring-structure fingerprint and the input-source fingerprint. Bounded context/pocket keys include content, device, typing version and the relevant ordered receptor selectors. Interaction pockets cover the union of selected complete residues; covalent product features receive a separate product-state identity. Pocket extraction preserves residue metadata and includes heteroatoms by default.

Prepared ligand/receptor objects and existing maps/topology can be reused. This is distinct from an automatic target-artifact generation/cache service. Meeko preparation, AutoGrid execution and AMBER parameterization have separate responsibilities: this package offers the documented preparation/import helpers and GPF generation, while map execution and force-field parameterization remain external.

AD4 shares interpolation cell/corner geometry among affinity/electrostatic/desolvation channels. Sparse Vina-family intra evaluation is opt-in because direct pair norms and shorter reductions can change FP32 rounding and total timing. Torch AMBER caches fixed–fixed energy once, removes that constant from search, and bounds cached moving-pair coefficients; larger systems use tiled evaluation. These optimizations do not establish global docking accuracy or GPU throughput.

## Output and compatibility

Ranked docking SDF output schema is `4`. Independent refinement uses `anchor-dock-sf-refinement-v1` and comparison uses `anchor-dock-sf-comparison-v1`. Batch resume epoch is `8`; content, search, typing and scorer contracts validate reuse. See [migration](MIGRATION.md) before reusing older artifacts.

[Latest CPU validation](../reports/cpu-validation-2026-10-30.md) records the current release checks; historical measurements are indexed separately in [reports](../reports/README.md). A successful unit or local-recovery check is not binding-affinity calibration or proof of global pose recovery.
