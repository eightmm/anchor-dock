# Changelog

## 0.5.0

- added canonical ordered `interactions=[...]` input for up to eight simultaneous atom-pair constraints while preserving the five single-interaction keywords;
- defined list semantics as `ALL`/`AND` only, rejected duplicate specifications and mixed single/list forms, and kept `OR`/`ANY` alternatives as separate jobs;
- enumerated each mapped SMARTS anchor automatically with `max_matches=16`, formed deterministic Cartesian assignments, and failed rather than truncating above `max_joint_matches=64`;
- added conservative per-conformer pairwise feasibility pruning, deterministic primary placement from the selector with the fewest matches, and all-constraint-aware coarse preselection;
- scored a union pocket around all selected receptor residues and keyed its bounded cache by the complete ordered selector set;
- averaged per-interaction weighted flat-bottom penalties during guidance, retained restraint-free release, required every final distance window, and ranked survivors only by the unmodified scorer;
- recorded ordered specifications, resolved receptor atoms, per-selector matches, joint assignments, and per-interaction distances without inferring interaction chemistry, protonation, or tautomers;
- added CLI `--interactions-json`, structured batch support, SDF output schema `4`, and batch resume epoch `5`.

## 0.4.0

- replaced the public unconstrained local search with `dock_interaction`, CLI/batch mode `interaction`, and `DockingJob.interaction`;
- required an exact receptor residue and atom, a ligand SMARTS with one mapped `:1` query atom, a target distance, and a tolerance;
- added fail-closed receptor selection and deterministic enumeration of every distinct ligand anchor match, with an error instead of cap truncation;
- added bounded seeded candidate generation, scorer-only coarse preselection across matches/conformers, guided SE(3)+torsion optimization, and restraint-free release on one live pose model;
- filtered released poses by the requested distance window and kept restraint values separate from the unmodified reported score and search energy;
- recorded selector, match, distance, guide/release, restraint, score, and protonation-limitation provenance without claiming an inferred chemical interaction;
- advanced the SDF output schema to `3` and batch resume epoch to `4` for the breaking contract.

## 0.3.0

- added one public namespace and four entry points: reference, covalent, free, and batch;
- retained one-release warning adapters for unambiguous 0.2 high-level calls while removing the old low-level namespaces and scripts;
- implemented inferred XS atom typing and modern AutoDock Vina/Vinardo pair functions, radii, coefficients, and 8 Å cutoff;
- separated search energy from reported score; recorded the intramolecular baseline, receptor/source identity, search parameters, and effective torsion semantics;
- fixed distributed-anchor kinematics and best-state optimizer restoration;
- removed duplicate reference MMFF relaxation and added truthful relaxation metadata;
- made reference MCS symmetry-aware and bounded with alternative fragment packing, explicit non-global cross completeness/max-size provenance, lazy placement search, fail-closed timeout/hard-node budgets, deterministic candidate caps, a consistent 4096 mapping cap, and structured mapping failures;
- added atom-pair-aware covalent bond targets, local noncarbon geometry preparation, rigid formed-bond/1–3 normalization, and end-to-end geometry gates;
- made covalent exclusions match formed-bond 1–2/1–3/1–4 graph distances and applied copy-on-write product-state typing to the receptor nucleophile;
- added scorer-independent `DockingEngine`, SoftDock, custom neural scorers, free local search, and heterogeneous manifests;
- standardized all SDF metadata under versioned `AnchorDock_*` fields and reject non-finite or dimensionally invalid output;
- added typed table options, content-hashed input/scorer/output resume integrity, runtime-aware batch signatures, callable/code/state-aware custom scorer fingerprints, atomic running/complete manifests, safe output-tree exclusion and all-input/output collision preflight, malformed-record reporting, water typing, and bounded receptor caches;
- added Haar-uniform free-search rotations, strict RDKit rotor topology, truthful optimization requested/applied/improved fields, and explicit salt/mixture rejection;
- rejected invalid direct batch modes, routed canonical batch jobs around compatibility adapters, and advanced the resume-signature epoch so pre-fix artifacts are recomputed;
- rejected occupied or incomplete covalent nucleophiles and invalid rotation scans before docking work begins;
- required exactly one bounded-read SDF record for single-molecule inputs, added explicit free `--no-optimize`, and advanced output schema `2` to distinguish true conformer IDs from representative ordinals;
- removed the unvalidated `vina_lp`, boronic-acid, and ambiguous alkynyl-amide claims instead of silently mapping them;
- changed scientific score semantics and output tags intentionally; 0.2 numeric results are not comparable to 0.3.
