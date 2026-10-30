# Scoring

## Vina pair function

For atom distance `r` and type-radius sum `R`, define surface distance `d = r - R`. Pair terms are evaluated only for `r < 8 Å`.

```text
gauss1(d) = exp(-(d / 0.5)^2)
gauss2(d) = exp(-((d - 3.0) / 2.0)^2)
repulsion(d) = d^2 if d < 0 else 0
hydrophobic(d) = 1                    d <= 0.5
                 1.5 - d              0.5 < d < 1.5
                 0                    d >= 1.5
hbond(d) = 1                           d <= -0.7
           -d / 0.7                    -0.7 < d < 0
           0                           d >= 0
```

Default coefficients:

```text
gauss1      -0.035579
gauss2      -0.005156
repulsion    0.840245
hydrophobic -0.035069
hbond       -0.587439
rotor        0.05846
```

## Vinardo

Vinardo uses its own XS radii, one Gaussian of width `0.8`, repulsion weight `0.8`, hydrophobic interval `0–2.5`, hydrogen-bond interval `-0.6–0`, and weights `-0.045`, `-0.035`, and `-0.600`.

AnchorDock follows the modern [`ccsb-scripps/AutoDock-Vina`](https://github.com/ccsb-scripps/AutoDock-Vina) lineage: Vinardo uses the current Vina rotor coefficient (`0.05846`) and C_H/C_P Vinardo radius (`2.0`). This intentionally differs from some original smina/Vinardo defaults.

## Search energy and reported score

The differentiable search objective is

```text
E_search = E_inter + E_intra
```

For Vina-family reporting, all poses in one docking run share the intramolecular energy of the best final search pose:

```text
E_reported = (E_search - E_intra_reference) / (1 + 0.05846 * N_rot)
```

This keeps mapping/pose comparisons consistent while preserving a stable optimization objective. Outputs record `Intramolecular_Reference`, input rotor count, effective score rotor count, and whether the penalty was requested and actually applied. SoftDock and custom scorers do not use this denominator, so their effective count is zero even when the option was requested.

## Atom typing boundary

Official Vina derives XS roles from prepared AutoDock/PDBQT types and polar-H bonds. AnchorDock now uses `inferred-xs-v3`: versioned AD4-compatible N/O graph predicates instead of RDKit pharmacophore `BaseFeatures.fdef` labels. Donor N/O requires attached H evidence; acceptors follow the supported AD4 preset convention. This corrects tertiary amines, neutral amines, nitriles, nitro groups and anionic oxygen. These are force-field compatibility rules, not universal medicinal-chemistry acceptor labels: the ordinary Meeko preset assigns OA to all oxygen and NA to some sulfonamide nitrogen.

- `compute_atom_features(mol, device, strict=True)` rejects diagnosed ambiguity. Without strict mode, `typing_ambiguities` and `typing_source` accompany the tensors.
- PDB residue templates remain inferred. HID/HSD, HIE/HSE and HIP/HSP declare histidine states; explicit H is respected. Unspecified HIS without polar H becomes conservative N_P with a diagnostic rather than both nitrogens being donor+acceptor. Conflicting aliases/H are diagnosed.
- `compute_atom_features_from_ad4(mol, types, device)` converts authoritative atom-order-aligned prepared types. It validates element/order and requires polar-H graph atoms before heavy-atom selection. It does not assign protonation, charges or PDBQT connectivity.
- `DockingEngine(strict_atom_typing=True)` rejects ambiguous selected inputs. Permissive engine preparation emits `AtomTypingWarning` so unresolved chemistry is visible.
- Ordinary S/SA becomes XS S_P with no Vina H-bond role; this is official Vina behavior. AD4 SA has a different force-field role.
- Supported generic metals follow the official Vina inventory. Generic Met_D does not establish correct metal coordination chemistry.

Reference/interaction outputs record `AnchorDock_Atom_Typing=inferred-xs-v3`; covalent product typing records `inferred-xs-v3+covalent-product-v1`. The batch signature includes typing and optimizer contracts, invalidating old numerical semantics.

Pair weights/radii and analytic equations match the represented ordinary Vina/Vinardo terms. Exact native compatibility additionally requires the same prepared type/rotor/mobility model, intra-pair set, reference convention, radial/grid interpolation and positive-energy curl wrapper. RDKit strict rotors are not automatically native Vina num_tors or AD4 TORSDOF. Values remain labelled `kcal/mol-like`.

Unsupported ordinary XS elements (for example boron) and extended macrocycle/dummy types fail explicitly. The authoritative converter is low-level; top-level SMILES/PDB pipelines still perform inferred preparation and are not a complete PDBQT importer.

## SoftDock

`softdock` is an uncalibrated Torch baseline combining smooth steric overlap, contact attraction, hydrophobic contact, and donor–acceptor attraction. Units are arbitrary. It is useful for stable local search and as a template for learned scorers.

## Interaction guidance versus reported score

For `n` simultaneous interaction constraints, interaction mode adds the mean of the per-item weighted flat-bottom terms during its guide phase:

```text
penalty_i = weight_i * relu(abs(distance_i - target_i) - tolerance_i)^2
E_guide = E_search + (1 / n) * sum_i(penalty_i)
```

This is identical to the 0.4 objective when `n=1`. The release phase optimizes `E_search` alone on the same live pose model. Final poses must fall within every requested `target_i ± tolerance_i` window; survivors are ranked and written using only the scorer's normal `AnchorDock_Score` and `AnchorDock_Search_Energy`. Restraint energies, formulae, weights, and per-phase distances are stored separately as provenance and never mixed into those score fields.

A short distance does not by itself establish a particular chemical interaction. Each guide is therefore labelled a generic atom-pair distance hypothesis, without hydrogen-bond, salt-bridge, metal, pi-interaction, protonation, tautomer, or compatibility inference.

## Custom scorers

An `nn.Module` scorer receives:

```python
(ligand_coords, receptor_coords, ligand_features, receptor_features)
```

and returns one scalar per pose. The module must be differentiable with respect to `ligand_coords`.

Parameters and buffers must already be on the receptor device. AnchorDock evaluates the module without accumulating gradients into scorer weights and restores its prior training/evaluation state. Its automatic fingerprint includes class/forward/helper code identity, persistent and non-persistent parameters/buffers, nested tensor/array attributes, other serializable attributes, `functools.partial` arguments, inspectable callable-object and bound-method state, and `extra_repr()`. Modules with forward/backward hooks require `NeuralScorerAdapter(..., fingerprint="...")`, because hook identity and captured state cannot be inferred safely. Unregistered external modules used through a bound method, opaque callable state, and behavior controlled by imported/global dispatch, files, services, or other external state also require an explicit calibrated fingerprint. Units remain `arbitrary` unless an adapter states otherwise.


## Coordinate derivatives and optimizer contract

`rotation_vector_to_matrix` uses a stable SO(3) exponential, including the infinitesimal Jacobian at zero rotation. A collapsed bond axis still uses the separate axis-angle identity fallback. Autograd computes `J(q).T @ grad_x(E)` without a dense Jacobian. At a finite raw rotation vector, world torque needs the SO(3) left-Jacobian transpose; it is directly the parameter gradient only at identity or under a local retraction. Torsion derivatives use current axis-cross-lever actions in radians.

Adam/AdamW clipping is per pose across parameter blocks. This numerical norm still mixes translation/angular units; it is not a Cartesian metric. Pose AdamW has zero weight decay, avoiding origin-dependent forces. LBFGS strong-Wolfe closures use the true objective gradient without clipping. A damped `J.T @ W @ J + lambda * D` metric with displacement limits is a proposed experiment, not an implemented energy Hessian.

## Experimental AD4 map component

`core.grid.TrilinearGrid` samples fixed `[C,Z,Y,X]` maps at physical angstrom coordinates with explicit origin/spacing. Cell derivatives include inverse spacing and can jump across cell faces. Closed-box bounds reject outside points; native ADGPU instead has upper-exclusive bounds and search-specific penalties.

`AD4IntermolecularGrid` evaluates only `sum_i A_type_i(x_i) + q_i*E_map(x_i) + abs(q_i)*D_map(x_i)`. Maps are already weighted. Caller-supplied map/type/charge alignment is required. There is no AutoGrid parser, intra contributor list, torsion/unbound report or full ADGPU scorer registration. The validation-heavy sampler is a correctness reference and synchronizes CUDA; GPU throughput is unmeasured.

See [scientific audit](../reports/scoring-audit-2026-10-30.md) for primary sources, external checks and remaining parity boundaries.
