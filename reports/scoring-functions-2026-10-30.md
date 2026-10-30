# Scoring-function integration — 2026-10-30

> **Historical snapshot.** For the current implementation and verified CPU status, see [CPU validation](cpu-validation-2026-10-30.md). The results and commands below belong to the recorded source version; their original evidence is retained.

AnchorDock 0.6 removes the uncalibrated SoftDock baseline and makes Vina the interaction default. It adds a capability-aware SF catalog and independent existing-pose refinement, instead of treating all scores as interchangeable energies.

## Available implementations

- Continuous Torch: Vina, Vinardo and three pinned DKOES presets.
- AD4: prepared ordinary rigid-receptor map scoring with explicit formulation/report conventions.
- Native FAST-table Vina/Vinardo objectives: bounded derivative-free pattern search.
- AMBER/OpenMM: parameterized full-complex potential with an exact force-to-Torch first derivative.
- Native Vina Python: Vina, Vinardo and explicit-map AD4 local optimization.
- Installed Smina/GNINA CLI adapters: six empirical SF choices; GNINA also exposes native CNN refinement with distinct reported diagnostics.
- Explicit factory registration and installed-package entry points for additional scorers.

The catalog has 25 backend-qualified entries, not 25 different physical models. Proprietary scores are not cloned. Native engine optimizers do not inherit AnchorDock reference/covalent/interaction guide contracts.

[Usage, optional dependencies, preparation and plugin contracts](../docs/SCORING_FUNCTIONS.md).

## Validation

Frozen all-extras suite: **911 passed, 18 CUDA-only skipped**. Ruff, lock checks, wheel build and installed-wheel import/SF sweep/native Vina/OpenMM/six CLI smoke checks passed. CI separately exercises core-only and optional-backend environments.

Actual optional execution includes Meeko preparation, native Vina/Vinardo/AD4 Python calls and OpenMM Reference/CPU. Smina/GNINA binaries are unavailable: their tests use explicit CLI contract stubs plus pinned official source review. No actual CNN or GPU execution is claimed.

Independent review identified and corrected native local-box semantics, GPU visibility selection, hydrogen-only branch retention, consistent force caps, AD4-map fingerprints, zero-step semantics, reported score direction, registry capability routing and output/input atom identity.

Explicit H exposed an inferred-rotor SMARTS degree issue (for example CCO 0 inferred heavy rotors versus 2 with explicit H). Inferred rotors now match on a heavy graph and map back to original indices. Authoritative prepared branches remain separate: Meeko ethanol legitimately has a polar branch, Vina endpoint count 0.5 and TORSDOF 1.

## Executed optimization smoke

One supplied 10gs crystal pose, 25 Adam steps, seed 713, one CPU thread. All four SFs start from the same unmodified source pose. These decreases check optimization connectivity, not docking accuracy, binding affinity or superiority.

| SF | Initial search energy | Final search energy |
|---|---:|---:|
| vina | -15.984533 | -17.338755 |
| vinardo | -14.827404 | -15.821220 |
| dkoes_fast | -2.705498 | -5.666698 |
| dkoes_scoring_old | -5.701175 | -7.163066 |

Raw scores are not comparable across SF rows. [Input/source hashes, coordinates, metrics, parameters and runtime evidence](scoring-functions-2026-10-30.json).

The AMBER bridge was compared with an independent OpenMM Context using a pinned official 22-atom alanine topology and ten explicit movable particles, under vacuum and GBn2, two poses each. Maximum energy error was 0 and maximum gradient error 3.55e-15 kcal/mol/angstrom; both gradchecks passed. [AMBER evidence](amber-scoring-2026-10-30.json). This checks a force-field contract, not a protein-ligand binding free energy.

## Boundaries

Existing positioned 3D input is required for refinement. PDB/PQR preparation follows explicit residue/templates and preserves supplied charges; it does not guess missing termini, pKa or tautomers. Prepared paths can add/merge H and quantize PDBQT; source maps and rounding policies are recorded. AMBER requires a complete topology and all mapped ligand atoms.

Maps and parameters are not generated merely by selecting their scorer. AutoGrid GPF generation is available, while actual AutoGrid execution remains external. Native reconstructed bonds may differ from a supplied chemical graph. Native AutoDock-GPU search parity, real CNN execution, GPU throughput and broad multi-target docking accuracy remain unmeasured.

Version 0.6 / batch epoch 6 invalidate older default-SoftDock artifacts. Ranked docking SDF schema remains 4; independent SF refinement has its own explicit schemas.
