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

Official Vina receives authoritative PDBQT atom types. AnchorDock infers XS-like types from RDKit chemistry and PDB residue metadata (including explicit water donor/acceptor fallbacks):

- hydrophobic or polar carbon;
- donor/acceptor nitrogen and oxygen;
- sulfur, phosphorus, halogens, silicon, astatine, and metals.

Reference and interaction outputs record `AnchorDock_Atom_Typing=inferred-xs-v2`. Covalent mode copy-on-write retypes the bonded receptor atom and records `inferred-xs-v2+covalent-product-v1`, the original reactant fingerprint, and a structured before/after change. The product rules use non-hydrogen-bonding CYS sulfur and bonded HIS nitrogen, acceptor-only SER/THR/TYR oxygen, and reaction-center-aware donor/acceptor states for LYS. These remain inferred XS-like types, not a protonation or quantum reaction model.

Vina/Vinardo values are therefore labelled `kcal/mol-like`; they are not claimed to be bitwise or affinity-identical to AutoDock Vina.

If an element cannot receive a validated XS-like type (for example boron), Vina and Vinardo fail explicitly. Use a calibrated custom scorer rather than accepting a silent generic type.

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
