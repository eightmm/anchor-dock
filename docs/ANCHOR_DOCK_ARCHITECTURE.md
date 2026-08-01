# AnchorDock architecture

## Boundary

AnchorDock separates **pose construction** from **pose evaluation**.

A mode-specific strategy constructs:

- an RDKit molecule/topology,
- an ensemble of Cartesian coordinates,
- anchor atoms that must remain fixed,
- intramolecular and intermolecular exclusion masks,
- mode-specific metadata.

The shared core then performs atom feature extraction, Vina/Vinardo-style scoring, torsional refinement, ranking, and export.

## Reference strategy

1. Parse reference and query ligand.
2. Resolve MCS mapping (`auto`, `single`, `multi`, or `cross`).
3. Generate ETKDG conformers under reference coordinate constraints.
4. Pool representatives from all applicable mappings.
5. Optionally relax non-anchor atoms.
6. Score, refine torsions, merge, rank, and export.

## Covalent strategy

1. Locate a supported protein nucleophile.
2. Detect and select a ligand warhead.
3. Validate the residue-warhead pairing.
4. Apply a reaction-class topology transform and construct a residue-linked adduct.
5. Generate conformers with the protein-side anchor fixed.
6. Align the anchor exactly and scan rotation about the anchor axis.
7. Exclude duplicated/pseudo covalent atoms from non-bonded pair scoring.
8. Score, refine torsions, rank, and export.

## Shared core invariants

- Coordinates are `[P,N,3]`; single-pose inputs may be `[N,3]`.
- Pair masks are normalized to `[P,N,M]`.
- Anchor atoms define the root rigid frame.
- All output poses are sorted by ascending final score.
- Covalent scores are not reaction energies.
