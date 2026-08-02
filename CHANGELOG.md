# Changelog

## 0.3.0

- removed all remaining legacy `run_*`, `final_selection`, and batched-class aliases;
- added one public namespace and four entry points: reference, covalent, free, and batch;
- implemented inferred XS atom typing, official Vina/Vinardo pair functions, radii, coefficients, and 8 Å cutoff;
- separated search energy from reported score and added a common intramolecular baseline;
- fixed distributed-anchor kinematics and best-state optimizer restoration;
- removed duplicate reference MMFF relaxation and added truthful relaxation metadata;
- enforced covalent bond direction/length and narrowed covalent exclusions;
- added scorer-independent `DockingEngine`, SoftDock, custom neural scorers, free local search, and heterogeneous manifests;
- standardized all SDF metadata under `AnchorDock_*`;
- added typed table options, option-aware resume signatures, water typing, and covalent receptor-context caching.
