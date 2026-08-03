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
    └── explicit single/multi atom-pair interaction guide
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

1. Normalize the five single-interaction keywords or the canonical ordered `interactions` list into one to eight `ALL`/`AND` constraints; mixed forms and exact duplicate items fail closed.
2. Resolve every exact standard-PDB receptor residue/atom and reject hydrogen, alternate-location, duplicate, absent, or ambiguous selectors.
3. Generate ligand conformers and independently enumerate every distinct canonical heavy-atom match selected by each SMARTS `:1` atom. Each selector is capped by `max_matches=16`; the deterministic Cartesian joint assignments are capped by `max_joint_matches=64`, with errors instead of truncation.
4. Score the union of complete residue-centered pockets around all selected receptor residues. The bounded cache key includes the complete ordered receptor selector set.
5. Apply conservative pairwise shell-distance feasibility checks to every conformer/joint-assignment pair. Passing is not a global feasibility proof; failing must mean the rigid conformer cannot satisfy that pair of shells.
6. Choose the constraint with the fewest ligand matches as the deterministic primary anchor. Seed bounded Haar-uniform orientations with its ligand atom exactly at target distance, and include every secondary flat-bottom violation in coarse preselection while distributing capacity across viable assignments and conformers.
7. Optimize SE(3) and torsions with the mean of the per-constraint weighted flat-bottom penalties, then continue on the same live pose model with all restraints removed.
8. After output-coordinate quantization, reject any released pose outside any requested window and rank survivors only by the unmodified scorer.

One ligand atom may satisfy multiple constraints, and each pose records its exact joint assignment. The constraints are generic atom-pair distance hypotheses. They do not discover receptor atoms or infer hydrogen bonds, salt bridges, metal interactions, pi interactions, protonation, tautomers, or chemical compatibility. `OR`, `ANY`, and k-of-n alternatives are separate jobs.

## Receptor contexts

`ReceptorContext` stores receptor coordinates, inferred atom features, an exact scoring-structure fingerprint, and the input PDB content fingerprint. Cache keys use content hash, device, and atom-typing version. Covalent mode additionally keys its bounded residue-centered pocket by the resolved selector. Interaction mode keys the union pocket by the complete ordered receptor selector set, cutoff, heteroatom policy, device, and typing version. Product-state retyping is copy-on-write, so cached reactant contexts cannot be polluted across warheads; the consumed product features receive a new structure fingerprint while retaining the source and reactant fingerprints. Pocket extraction preserves PDB residue metadata and includes heteroatoms by default.
