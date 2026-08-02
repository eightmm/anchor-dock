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
    └── free multistart placement
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

## Reference strategy

1. Load and canonicalize the query.
2. evaluate contiguous and bounded disjoint-fragment MCS candidates;
3. select `single`, `multi`, or `cross` deterministically;
4. generate ETKDG conformers under each coordinate map;
5. restore exact anchor coordinates before clustering;
6. optionally relax each representative exactly once with fixed anchors;
7. optionally optimize torsions without moving any fixed anchor;
8. pool all mappings and report against one intramolecular reference.

## Covalent strategy

1. Resolve one protein nucleophile; ambiguous automatic selection is rejected.
2. Detect and select a supported ligand warhead.
3. Apply a reaction-class topology transform.
4. Add pseudo support and nucleophile atoms to the ligand topology.
5. fix support, nucleophile, and electrophile coordinates to enforce bond direction and length;
6. optionally scan rotations around the support–nucleophile axis;
7. exclude pseudo rows and only the duplicated electrophile–receptor-nucleophile pair;
8. optimize remaining torsions and verify the bond-length invariant.

## Free strategy

Free mode samples conformers, translations, and axis-angle rotations inside a box, then optimizes SE(3) and torsions with a soft boundary penalty. It is deliberately described as multistart local search rather than global docking.

## Receptor contexts

`ReceptorContext` stores receptor coordinates and inferred atom features. The cache key includes absolute path, modification time, file size, device, and atom-typing version. Covalent mode additionally caches the resolved reactive anchor and residue-extracted pocket by residue, cutoff, heteroatom policy, device, and typing version. Pocket extraction preserves PDB residue metadata and includes heteroatoms by default.
