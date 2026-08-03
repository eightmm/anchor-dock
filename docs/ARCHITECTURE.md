# Architecture

## Boundary

A strategy constructs a `PreparedDockingProblem`:

- one RDKit topology;
- one receptor context;
- initial pose coordinates;
- fixed anchor atoms, when applicable;
- intramolecular and intermolecular exclusion masks;
- a prepared differentiable scorer.

`DockingEngine` then performs scorer-independent optimization and reporting.

```text
input strategy
    ├── reference MCS
    ├── residue–warhead covalent adduct
    └── explicit atom-pair interaction guide
            ↓
PreparedDockingProblem
            ↓
rigid-frame kinematics + Torch optimizer
            ↓
search energy + reported score
            ↓
standard AnchorDock SDF/JSON output
```

## Core invariants

1. Coordinates are `[poses, atoms, 3]`; single poses may enter as `[atoms, 3]`.
2. Pair masks normalize to `[poses, ligand_atoms, receptor_atoms]`.
3. Intramolecular scoring excludes 1–2, 1–3, 1–4, and same-rigid-frame pairs.
4. A torsion is disabled when fixed anchors occur on both graph sides.
5. Early-stopped Adam rows cannot move through momentum; every pose restores its best parameter state.
6. Output poses are sorted by reported score.
7. Search energy and reported score are separate quantities.
8. Kinematics and the Vina-family denominator share RDKit's strict rotor definition.
9. Receptor/source/scorer/search identities and the intramolecular reporting baseline are persisted.

## Reference strategy

1. Load and canonicalize the query.
2. evaluate contiguous and bounded alternative disjoint-fragment MCS candidates, quotienting common pattern automorphisms;
3. select `single`, `multi`, or `cross` deterministically and report candidate/global-maximum proof limits;
4. generate ETKDG conformers with the coordinate map fixed in the absolute reference frame;
5. validate exact anchors plus heavy-atom bond/1–3 geometry before clustering;
6. optionally relax each representative exactly once with fixed anchors;
7. optionally optimize torsions without moving any fixed anchor;
8. pool all mappings and report against one intramolecular reference.

## Covalent strategy

1. Resolve one protein nucleophile; ambiguous automatic selection is rejected.
2. Detect and select a supported ligand warhead.
3. Apply a reaction-class topology transform.
4. Add pseudo support and nucleophile atoms to the ligand topology.
5. select a formed-bond target from validated carbon/residue values or atom-pair distance-geometry bounds;
6. fix protein support/nucleophile coordinates, prepare noncarbon local geometry, and rigidly normalize the complete ligand branch to exact formed-bond and 1–3 geometry;
7. copy the cached receptor features and retype the bonded nucleophile with versioned product-state donor/acceptor rules;
8. optionally scan rotations around the support–nucleophile axis;
9. exclude pseudo rows and formed-bond cross pairs equivalent to intramolecular 1–2/1–3/1–4 pairs;
10. optimize remaining torsions and verify bond-length and 1–3 invariants.

## Interaction strategy

1. Resolve one exact standard-PDB receptor residue/atom and reject hydrogen, alternate-location, duplicate, absent, or ambiguous selectors.
2. Generate ligand conformers and enumerate every distinct canonical heavy-atom match selected by the single mapped SMARTS atom `:1`; fail rather than truncate at the configured cap.
3. Extract a residue-centered receptor pocket and seed bounded Haar-uniform candidate orientations with the matched ligand atom exactly at the target distance.
4. Coarse-score candidates with the unmodified scorer and preselect deterministically, distributing capacity across ligand matches and conformers.
5. Optimize SE(3) and torsions with a flat-bottom atom-pair distance guide, then continue on the same live pose model for a restraint-free release phase.
6. Reject released poses outside the requested distance window and rank survivors only by the unmodified scorer.

The selector and restraint represent a generic atom-pair distance hypothesis. They do not discover an interaction site or infer hydrogen bonds, salt bridges, metal interactions, pi interactions, protonation, tautomers, or chemical compatibility.

## Receptor contexts

`ReceptorContext` stores receptor coordinates, inferred atom features, an exact scoring-structure fingerprint, and the input PDB content fingerprint. Cache keys use content hash, device, and atom-typing version. Covalent and interaction modes additionally key bounded residue-centered pocket contexts by their resolved selectors, cutoff, heteroatom policy, device, and typing version. Product-state retyping is copy-on-write, so cached reactant contexts cannot be polluted across warheads; the consumed product features receive a new structure fingerprint while retaining the source and reactant fingerprints. Pocket extraction preserves PDB residue metadata and includes heteroatoms by default.
