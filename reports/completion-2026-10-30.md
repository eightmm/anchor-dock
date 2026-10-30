# Prepared scoring and optimization completion — 2026-10-30

This extends scientific correction snapshot `957c75c`. Supported tasks remain bounded local docking/refinement with supplied chemistry; global search completeness and automatic protonation/tautomer selection are outside scope.

## Implemented

| Area | Result |
|---|---|
| Prepared input | Strict graph/PDBQT serial mapping, polar H, types/charges, tree and original indices. Prepared mobility drives intra contributors and kinematics. |
| Rotor semantics | Half-integral native Vina count preserved independently of AD4 TORSDOF. |
| Vina scoring | Continuous search retained; opt-in native curl and explicit nondifferentiable FAST-table final scoring. |
| AD4 grids | AutoGrid geometry/channel validation, inter maps, ordinary intra terms, torsion and explicit unbound reports; declared formulation/wrapper/bounds. |
| Geometry optimizer | Opt-in damped Cartesian metric, actual displacement bound, true-objective Armijo backtracking and per-pose rollback. |
| Reproducibility | Input/scorer/map fingerprints, finite option validation, explicit PDB H policy and versioned cache/resume contracts. No new required dependencies. |

[Usage and contracts](../docs/PREPARED.md). Adam remains default. The dense Cartesian reference uses bounded Jacobians and the existing rotation-vector chart. Initial atom-typing/HIS, SO(3), clipping, AdamW and LBFGS corrections remain included.

## Local verification

- Full frozen suite: **755 passed, 15 CUDA-only skipped**.
- Ruff, frozen lock, wheel build and whitespace checks passed.
- Installed wheel independently passed new API import, prepared AD4 refinement, native fixed-fixture scoring and all four CLI commands.
- Independent cross-review covered mapping, mobility/rotors, AD4 terms, grids/curl and optimizer Jacobians/state. Discovered zero-distance electrostatic and low-level invalid-option cases were corrected with regressions.
- Tested package digest: `3edf156991807ed3898ca102749464751efc908a2af9c6a74f415dd9e8bc876d`; the real-complex benchmark used the same digest.
- Remote CI is required before merging PR #6.

With installed RDKit 2025.9.5 the previous unsanitized PDB loader already retained H despite removeHs=True. Explicit preservation clarifies policy and prevents regressions; these experiments do not establish an old numerical H-loss bug.

## Executed external checks

**Vina/Vinardo:** eight prepared ligands × three placements × two scorers with a controlled rigid glycine receptor. Matched final-score conventions give max inter/intra/total errors **4.10e-6 / 3.25e-7 / 2.91e-6**. Continuous analytic energies intentionally differ from the discrete table. [Evidence](native-scoring-parity-2026-10-30.md).

**AD4:** four prepared flexible ligands with nonempty intra contributors and synthetic affine maps. Matched Vina AD4 inter/final error is **3.55e-15**. Analytic intra differs by up to 9.12e-4; native radial sampling explains that residual. [Evidence](ad4-native-scoring-2026-10-30.md).

**Actual complexes:** 10gs, 1iep and 5x72; 24 perturbed starts plus three crystal controls, 100 steps, one CPU thread, fixed-frame symmetry RMSD. Every tested optimizer finishes 24/24 perturbed starts within 2 Å and limited sanity bounds. Nineteen already passed initially: **five new recoveries, zero losses**. Adam/LBFGS recorded scalars exactly match the scientific baseline. Cartesian shows no consistent accuracy or speed benefit and remains opt-in. [Protocol, results and hashes](redocking-2026-10-30.md).

## Validation boundaries

CUDA correctness/throughput, native AutoDock-GPU end-to-end execution, actual receptor AutoGrid generation, broad redocking/cross-docking and screening enrichment remain unmeasured. Three selected complexes cannot establish general accuracy.

Maps are authoritative receptor fields; their values cannot prove association with a separate receptor graph. Supplied graphs may differ from native distance-based bond reconstruction. Unsupported extended/macrocycle types, flexible receptors and map-incompatible receptor exclusions fail explicitly.
