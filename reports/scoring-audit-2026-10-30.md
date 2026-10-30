# AnchorDock scientific scoring and Jacobian audit

The prior performance suite preserved existing behavior. This audit uses independent prepared types, native executables and analytic derivatives. It found substantive atom-typing and optimizer errors; correcting them intentionally changes scores and trajectories. Historical v2 scores do not use the v3 contract.

## Primary sources

- AutoDock-Vina develop [`bb64e5e26b520569a86360d69248069b5e665b3f`](https://github.com/forlilab/AutoDock-Vina/tree/bb64e5e26b520569a86360d69248069b5e665b3f): atom_constants.h, model.cpp:409, scoring_function.h, potentials.h, conf_independent.cpp, precalculate.h, grid.cpp and curl.h.
- Meeko develop [`5b3618a34d087d22f6ca3e6d6754c2aaba80020a`](https://github.com/forlilab/Meeko/tree/5b3618a34d087d22f6ca3e6d6754c2aaba80020a): meeko/data/params/ad4_types.json and meeko/atomtyper.py.
- AutoDock-GPU develop [`be06a135181178c8631653e1e32d9ece86b5b4d8`](https://github.com/forlilab/AutoDock-GPU/tree/be06a135181178c8631653e1e32d9ece86b5b4d8): cuda/calcenergy.cu, cuda/calcMergeEneGra.cu, host/src/processgrid.cpp, processresult.cpp and processligand.cpp.

The official repositories moved from ccsb-scripps to forlilab. No native kernel source is vendored in this patch.

## Confirmed findings and corrections

| Area | Finding | Correction/boundary |
|---|---|---|
| Pair terms | Ordinary weights/radii/Gaussians/ramps and 8 Å cutoff match | Retain analytic equations |
| Ligand typing | Pharmacophore labels misclassify amines, nitrile, nitro, anions | AD4-compatible graph predicates, inferred-xs-v3 |
| HIS | Both N unconditionally donor+acceptor | Declared tautomer/explicit-H handling, ambiguity/strict policy |
| Prepared types | Inference could replace prepared roles | Validated authoritative AD4-to-XS converter |
| Rotation | Identity rotation loses the Lie Jacobian | Stable SO(3) exponential |
| Clipping | One pose changes another pose's clipping scale | Per-pose norm across parameter blocks |
| AdamW | Hidden decay attracts coordinates to chart zero | No pose weight decay |
| LBFGS | Clipped gradient disagrees with returned loss | Unclipped objective derivative for strong Wolfe |
| Resume | Old semantics can be reused | Typing/optimizer contracts in signatures |

The rules target ordinary Meeko/Vina force-field conventions, not universally correct medicinal-chemistry HBA labels. Broad OA/sulfonamide rules are empirical. Inferred PDB templates remain approximate and missing state is not resolved by guessing protonation.

## Executed external checks

Python 3.12, RDKit 2025.9.5, Meeko 0.8.0 and native Vina 1.2.7, CPU:

- Actual Meeko MoleculePreparation on 24 genuine ETKDGv3 explicit-H conformers, seed713: old v2 differs on 10 molecules/12 heavy atoms. Corrected heavy-input, explicit-H and supplied-AD4 conversion match all24 fixtures.
- 48 one-pair Vina/Vinardo comparisons with manually aligned XS attributes: max raw difference0.0021623. Matching native positive-energy curl(E,1000) reduces max difference to0.00004905; residual radial/table behavior remains. This is not full ligand/rotor/native parity.
- Nine synthetic AutoGrid-format map cases, signed/zero charges and nonzero coordinates, compared with Vina AD4: raw ADGPU-style map contributions differ from Vina's wrapper by up to0.0032278. Vina curls each positive affinity/electrostatic/desolvation map sample separately. Explicitly matching that wrapper gives max error6.66e-16. These are synthetic maps, not real receptor AutoGrid output or native AutoDock-GPU execution.
- Grid FP64 gradcheck covers coordinates/charges, nonzero origin/anisotropic spacing, axis/channel order, cross terms, boundaries and invalid inputs.
- Rotation checks include zero/near-zero gradcheck/gradgradcheck, matrix-exponential parity, finite-rotation left-Jacobian and translation/torsion VJPs.

[Machine-readable evidence](scoring-audit-2026-10-30.json). Reproduce the Meeko check with `examples/audit_atom_typing.py` and its documented optional dependency. Meeko/Vina are audit tools, not new AnchorDock runtime dependencies.

Final project checks: **555 passed, 13 CUDA-only skipped**; Ruff, frozen lock, wheel build and installed-wheel API/CLI checks passed. Independent static review found no blocking correctness issue.

## Jacobian interpretation

Write x_i(q)=t+R(w)y_i(theta), g_i=dE/dx_i; physical force is -g_i.

```text
dE/dt = sum_i g_i
torque_gradient = sum_i (x_i - t) cross g_i
dE/dw = J_left(w).T @ torque_gradient
dE/dtheta_k = sum_{i in subtree_k} g_i dot [axis_k cross (x_i - origin_k)]
```

Autograd's VJP supplies J.T@g; no dense (3N)x(6+K) Jacobian is needed for first-order optimization. Plain torque is not the finite raw-rotation-vector derivative without the chart Jacobian. Angles are radians, positions angstroms, and grid derivatives include inverse spacing.

Concrete rotation failure: atom(1,0,0), zero rotation, d(y)/d(w_z) should be1. Old autograd returned0, central differences approximately0.99999993. Concrete clipping failure: the same quadratic pose ended0.5000008 alone versus0.6358875 with a high-gradient neighbor. Per-pose clipping produces0.5000008 in both cases.

A future damped pullback metric M=J.T W J+lambda D and dq=-eta M^-1 J.Tg can control Cartesian displacement across translation/rotation/torsion. This is a geometry metric, not the energy Hessian. Damping, redundant DOFs, SO(3) retraction and objective-decreasing step acceptance need qualification. No such preconditioner is claimed here.

## AD4/AutoDock-GPU direction

Grids do not prevent differentiation. Native AutoDock-GPU differentiates trilinear maps and maps Cartesian derivatives into translation/torque/torsion updates. A fixed receptor permits eight samples per atom/map instead of every receptor atom pair.

The prototype implements only sum_i A_type(x_i)+q_i E(x_i)+abs(q_i)D(x_i). Full AD4 needs prepared charges/types and polar H, correct intra contributors/exclusions, smoothed vdW/H-bond/electrostatic/desolvation choices, and torsion/unbound/report semantics. Macrocycles/covalent/flexible receptors require explicit state contracts.

Native GPU search updates differ from exact derivatives: the inspected default kernel quantizes/clips gradients, uses angular scaling and a Gaussian-shaped derivative for a rational desolvation approximation. A clean differentiable scorer should differentiate its own forward exactly; native update emulation is a separate mode. ADGPU and Vina AD4 also differ in boundaries and curl wrappers.

Next acceptance sequence: real AutoGrid reader with origin/node/axis checks; prepared type/charge mapping; native component parity; intra/torsion/unbound reporting; then fused interpolation/VJP and geometric preconditioning. GPU performance and real-map/native ADGPU end-to-end parity remain unmeasured.
