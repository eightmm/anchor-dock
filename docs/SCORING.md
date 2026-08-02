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

Vinardo uses its own XS radii, one Gaussian of width `0.8`, repulsion weight `0.8`, hydrophobic interval `0–2.5`, hydrogen-bond interval `-0.6–0`, and the official default weights `-0.045`, `-0.035`, and `-0.600`.

## Search energy and reported score

The differentiable search objective is

```text
E_search = E_inter + E_intra
```

For Vina-family reporting, all poses in one docking run share the intramolecular energy of the best final search pose:

```text
E_reported = (E_search - E_intra_reference) / (1 + 0.05846 * N_rot)
```

This keeps mapping/pose comparisons consistent while preserving a stable optimization objective.

## Atom typing boundary

Official Vina receives authoritative PDBQT atom types. AnchorDock infers XS-like types from RDKit chemistry and PDB residue metadata (including explicit water donor/acceptor fallbacks):

- hydrophobic or polar carbon;
- donor/acceptor nitrogen and oxygen;
- sulfur, phosphorus, halogens, silicon, astatine, and metals.

Outputs record `AnchorDock_Atom_Typing=inferred-xs-v2`. Vina/Vinardo values are therefore labelled `kcal/mol-like`; they are not claimed to be bitwise or affinity-identical to AutoDock Vina.

## SoftDock

`softdock` is an uncalibrated Torch baseline combining smooth steric overlap, contact attraction, hydrophobic contact, and donor–acceptor attraction. Units are arbitrary. It is useful for stable local search and as a template for learned scorers.

## Custom scorers

An `nn.Module` scorer receives:

```python
(ligand_coords, receptor_coords, ligand_features, receptor_features)
```

and returns one scalar per pose. The module must be differentiable with respect to `ligand_coords`.
