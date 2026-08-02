# Changelog

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
- removed the unvalidated `vina_lp`, boronic-acid, and ambiguous alkynyl-amide claims instead of silently mapping them;
- changed scientific score semantics and output tags intentionally; 0.2 numeric results are not comparable to 0.3.
