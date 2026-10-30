# Scoring

This document defines numerical semantics. [SCORING_FUNCTIONS.md](SCORING_FUNCTIONS.md) owns named selection, preparation and usage; [PREPARED.md](PREPARED.md) owns graph/PDBQT and map ingestion.

| Catalog objective | Meaning | Units / reporting direction |
|---|---|---|
| `vina`, `vinardo` | Continuous empirical docking energy and Vina-family report | `kcal/mol-like`, lower |
| `ad4`, `adgpu` | Prepared map/pair formulation with explicit unbound reporting | `kcal/mol-like`, lower |
| `amber` | Parameterized full-complex vacuum potential in Torch | `kcal/mol`, lower |
| `vina-final`, `vinardo-final` | Detached FAST-table reference objective/report | `kcal/mol-like`, lower; pattern only |

These functions share optimization infrastructure, not calibration. Raw scores are not averaged or ranked across functions. The table references do not invoke native search. `OpenMMScorer` is a separate explicit reference/advanced object; custom learned modules remain optional extensions.

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

## Vina-family search energy and reported score

The differentiable search objective is

```text
E_search = E_inter + E_intra
```

For Vina-family reporting, all poses in one docking run share the intramolecular energy of the best final search pose:

```text
E_reported = (E_search - E_intra_reference) / (1 + 0.05846 * N_rot)
```

This keeps mapping/pose comparisons consistent while preserving a stable optimization objective. Outputs record `Intramolecular_Reference`, input rotor count, effective score rotor count, and whether the penalty was requested and actually applied. Custom scorers do not use this denominator unless their own reporting contract declares it.

## Atom typing boundary

Official Vina derives XS roles from prepared AutoDock/PDBQT types and polar-H bonds. AnchorDock now uses `inferred-xs-v3`: versioned AD4-compatible N/O graph predicates instead of RDKit pharmacophore `BaseFeatures.fdef` labels. Donor N/O requires attached H evidence; acceptors follow the supported AD4 preset convention. This corrects tertiary amines, neutral amines, nitriles, nitro groups and anionic oxygen. These are force-field compatibility rules, not universal medicinal-chemistry acceptor labels: the ordinary Meeko preset assigns OA to all oxygen and NA to some sulfonamide nitrogen.

- `compute_atom_features(mol, device, strict=True)` rejects diagnosed ambiguity. Without strict mode, `typing_ambiguities` and `typing_source` accompany the tensors.
- PDB residue templates remain inferred. HID/HSD, HIE/HSE and HIP/HSP declare histidine states; explicit H is respected. Unspecified HIS without polar H becomes conservative N_P with a diagnostic rather than both nitrogens being donor+acceptor. Conflicting aliases/H are diagnosed.
- `compute_atom_features_from_ad4(mol, types, device)` converts authoritative atom-order-aligned prepared types. It validates element/order and requires polar-H graph atoms before heavy-atom selection. It does not assign protonation, charges or PDBQT connectivity.
- `DockingEngine(strict_atom_typing=True)` rejects ambiguous selected inputs. Permissive engine preparation emits `AtomTypingWarning` so unresolved chemistry is visible.
- Ordinary S/SA becomes XS S_P with no Vina H-bond role; this is official Vina behavior. AD4 SA has a different force-field role.
- Supported generic metals follow the official Vina inventory. Generic Met_D does not establish correct metal coordination chemistry.

Reference/interaction outputs record `AnchorDock_Atom_Typing=inferred-xs-v3`; covalent product typing records `inferred-xs-v3+covalent-product-v1`. The batch signature includes typing and optimizer contracts, invalidating old numerical semantics.

Pair weights/radii and analytic equations match the represented ordinary Vina/Vinardo terms. Exact native compatibility additionally requires the same prepared type/rotor/mobility model, intra-pair set, reference convention, radial/grid interpolation and positive-energy curl wrapper. `PreparedLigand` preserves a supplied graph/PDBQT mapping and tree, deriving half-integral Vina `num_tors` independently of AD4 `TORSDOF`. Inferred rotors match the supported strict SMARTS on a heavy-atom graph and map back to the original atom indices; explicit H does not create extra terminal heavy-atom rotors. Values remain labelled `kcal/mol-like`.

Unsupported ordinary XS elements (for example boron) and extended macrocycle/dummy types fail explicitly. Top-level SMILES/PDB pipelines retain inferred preparation; the engine's explicit prepared path imports ordinary ligand/rigid receptor PDBQT together with a caller-supplied chemical graph and serial mapping. It does not infer connectivity or protonation. See [prepared inputs](PREPARED.md).

`PairwiseScorer(..., positive_energy_cap=1000.0)` opts into native positive-energy curl: after the receptor sum for each ligand atom, and separately for each intra pair. The default retains uncapped analytic energies. The score-only `native_vina_score` additionally follows native FAST radial-table conventions and rejects coordinate gradients. The named `vina-final` / `vinardo-final` wrappers explicitly detach coordinates and may use bounded derivative-free pattern search; they cannot supply an autograd objective. The [full-ligand native audit](../reports/native-scoring-parity-2026-10-30.md) quantifies continuous and table-reference differences.

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


## Prepared state and scorer identity

`PairwiseScorer` copies the supplied `ScoringConfig` and its weights rather than sharing a mutable global or caller-owned weight dictionary. Factory identity reflects the current factory configuration, including the weight evaluation order that can affect floating-point reduction. Each prepared pairwise scorer snapshots its configuration, receptor coordinates, features and masks. Later mutations of the factory or original input objects do not silently change that prepared calculation or its identity. Tensor snapshots retain the supported autograd connections.

This is an ownership contract, not a cache for new chemistry. Prepare a new scorer when intentionally changing molecular state or model parameters. Directly editing a prepared object is a new state change, not protected input reuse; prepare a new object when changing its inputs. Custom modules and opaque external state retain their separate explicit-fingerprint requirements in the custom-scorer contract.

## Dense and sparse Vina-family intra evaluation

`PairwiseScorer(..., intra_mode="dense")` preserves dense `cdist` and its reduction order. Opt-in `intra_mode="sparse"` snapshots the mask, gathers contributing unordered pairs and evaluates their shared equations. Asymmetric mask entries retain the original half-sum semantics. Explicit diagonal mask entries retain their defined contribution; the ordinary prepared mobility masks exclude them.

Sparse evaluation reduces work when few intra pairs contribute, but the intermolecular tensor may dominate total time. Direct pair norms versus dense `cdist`, and the shorter reduction, can change FP32 rounding and an optimization trajectory. The mode enters scorer identity and does not change force-field coefficients. CPU measurements are scoped in the [validation reports](../reports/README.md).

## Coordinate derivatives and optimizer contract

`rotation_vector_to_matrix` uses a stable SO(3) exponential, including the infinitesimal Jacobian at zero rotation. A collapsed bond axis still uses the separate axis-angle identity fallback. Autograd computes `J(q).T @ grad_x(E)` without a dense Jacobian. At a finite raw rotation vector, world torque needs the SO(3) left-Jacobian transpose; it is directly the parameter gradient only at identity or under a local retraction. Torsion derivatives use current axis-cross-lever actions in radians.

Adam/AdamW clipping is per pose across parameter blocks. This numerical norm still mixes translation/angular units. Pose AdamW has zero weight decay, avoiding origin-dependent forces. LBFGS strong-Wolfe closures use the true objective gradient without clipping. Opt-in `cartesian` implements the dense reference metric `J.T @ J + lambda * D`, block-isotropic damping, actual atom-displacement limits and objective-based Armijo backtracking. It uses the existing rotation-vector chart and bounded per-pose Jacobians; it is not an energy Hessian or a fused GPU optimizer. Adam remains the default. Cartesian backtracking rejects map-domain and physical numerical-domain trial failures; invalid initial geometry still fails. Pattern search is a bounded per-pose coordinate poll with actual atom-displacement and evaluation limits, not an approximate native optimizer.

## AD4 and ADGPU map formulations

`core.grid.TrilinearGrid` samples fixed `[C,Z,Y,X]` maps at physical angstrom coordinates with explicit origin/spacing. Cell derivatives include inverse spacing and can jump across cell faces. Its ordinary sampler uses closed-box bounds. The AD4 raw-map formulation applies upper-exclusive bounds, while the explicit Vina map mode uses closed bounds and positive-energy curl. Out-of-grid inputs are rejected; this is not the native search penalty. `sample_channels` shares cell/corner geometry across multiple requested channels, and AD4 uses it for affinity/electrostatic/desolvation samples.

`AD4IntermolecularGrid` evaluates `sum_i A_type_i(x_i) + q_i*E_map(x_i) + abs(q_i)*D_map(x_i)`. Maps are already weighted. The companion AutoGrid loader validates physical geometry and channel order. `AD4Scorer` adds ordinary prepared intra contributors, torsion and explicit same-as-bound/fixed unbound reporting through the engine scorer interface. Standard analytic AD4 and inspected ADGPU forward approximations are explicit formulations; raw-map and Vina curl/boundary conventions are explicit modes. Receptor-atom exclusions and flexible receptors are unsupported.

Maps are the authoritative receptor field: contents are fingerprinted, but their physical association with a separately supplied receptor graph cannot be established from map values. The validation-heavy sampler synchronizes CUDA. Native radial-table differences, native AutoDock-GPU execution, real receptor map generation and GPU throughput remain separate validation boundaries. See [usage and contracts](PREPARED.md).

## AMBER fixed-charge vacuum potential

The named `amber` factory constructs `TorchAmberScorer`. OpenMM imports a fully parameterized System once; runtime energies and automatic derivatives use Torch, including FP64 computation for FP32 or FP64 coordinate inputs. This backend supports harmonic bonds/angles, periodic torsions, ordinary nonbonded Coulomb/Lennard-Jones and explicit exceptions under nonperiodic vacuum NoCutoff settings.

Its potential contains the supplied bonded terms and

$$
E_{ij}=\frac{Cq_iq_j}{r_{ij}}+4\epsilon_{ij}\left[\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12}-\left(\frac{\sigma_{ij}}{r_{ij}}\right)^6\right].
$$

Ordinary LJ uses Lorentz–Berthelot mixing. Explicit exception charge products, sigma and epsilon replace ordinary pair parameters, retaining topology-defined exclusions and 1–4 scaling. Bond/angle force constants and torsion phases follow the imported System; they are not Vina pair terms or its rotor denominator. Coordinates are angstroms, potential energy is kcal/mol and Cartesian derivatives are kcal/mol/angstrom.

Fixed–fixed contributions are computed once. Search minimizes the moving contribution minus its fixed reference value, avoiding loss of small improvements in a large receptor constant. Reported `score` and named `complex_potential` restore the full supplied complex potential. Protocol inter/intra placeholders are omitted from physical output metrics; no binding-energy subtraction or affinity calibration is implied.

A bounded static pair cache stores exception-adjusted moving-pair indices/coefficients when their candidate count fits `pair_cache_max_pairs` (default 200,000; zero disables it). Larger systems use tiled evaluation. Both routes retain the same potential and are checked against the independent OpenMM Reference backend. Atom mapping, source/parameter content and cache configuration are recorded. These caches do not parameterize new ligands or generate receptor maps.

Unsupported implicit solvent, periodic electrostatics, virtual sites, parameter offsets and force types fail explicitly. Zero-epsilon LJ terms canonicalize their unused sigma before high powers, so a Coulomb-only pair cannot acquire a `0 * inf` NaN from an irrelevant LJ value. Active zero-distance or singular bonded geometry is rejected rather than softened. The explicit `OpenMMScorer` object retains its supported advanced models and first-derivative force bridge; its settings are not silently transferred to the Torch vacuum function.

## Evidence and boundaries

[Latest CPU validation](../reports/cpu-validation-2026-10-30.md) and the [historical report index](../reports/README.md) describe source/input hashes, oracle comparisons, finite-difference checks and measured performance separately. The [initial scientific audit](../reports/scoring-audit-2026-10-30.md) retains primary-source provenance, but dated observations are not current test counts or a promise of unchanged implementation. Broad docking accuracy, affinity prediction and GPU throughput need their own evidence.
