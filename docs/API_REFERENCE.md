# API reference

AnchorDock 0.5 exposes four high-level operations from `anchor_dock`:

```python
from anchor_dock import dock_batch, dock_covalent, dock_interaction, dock_reference
```

## Reference (selected options)

```python
dock_reference(
    protein_pdb,
    reference_ligand,
    query_ligand,
    output_dir="output_predictions",
    *,
    num_confs=1000,
    rmsd_threshold=1.0,
    mcs_mode="auto",              # auto | single | multi | cross
    min_mcs_atoms=3,
    min_fragment_size=5,
    max_fragments=3,
    max_mappings=64,
    mcs_timeout=10,
    match_chirality=False,
    relax=True,
    optimize=False,
    optimizer="adam",             # adam | adamw | lbfgs
    freeze_anchor=True,
    scorer="vina",                # vina | vinardo | softdock | nn.Module
    top_k=None,
    random_seed=42,
    device=None,
)
```

Every successful mapping is pooled. Pattern automorphisms preserve symmetry-related correspondences. Results expose attempted, selected, and failed mappings, exact atom-index spaces, `mcs_candidate_complete`, `mcs_max_size_proven`, and the configured candidate limit. A nonempty bounded cross search, including one considered by `auto`, prevents a claim of global completeness or maximum-size proof.

## Covalent (selected options)

```python
dock_covalent(
    protein_pdb,
    query_ligand,
    reactive_residue=None,
    output_dir="output_predictions",
    *,
    num_confs=1000,
    rotation_scan_step=30,
    rotation_top_k=50,
    optimize=False,
    scorer="vina",
    top_k=None,
    warhead_index=0,
    strict_compatibility=False,
)
```

Automatic residue selection succeeds only for one unambiguous supported nucleophile. The output is an adduct-conditioned pose ranking, not a reaction energy. Scoring uses a copy-on-write product-state type for the bonded receptor atom; results record the base and product typing versions, structured change, reactant fingerprint, and exact scoring-structure fingerprint.

Omitting `optimize` at the top-level compatibility entry point currently runs with `False` and emits a `FutureWarning`; pass it explicitly.

## Interaction-guided local search (selected options)

```python
dock_interaction(
    protein_pdb,
    query_ligand,
    output_dir="anchor_dock_interaction",
    *,
    receptor_residue=None,
    receptor_atom=None,
    ligand_smarts=None,
    target_distance=None,
    distance_tolerance=None,
    interactions=None,
    pocket_cutoff=12.0,
    include_heteroatoms=True,
    num_confs=32,
    num_candidates=128,
    preselect_k=16,
    max_matches=16,
    max_joint_matches=64,
    optimize=True,
    opt_steps=50,
    release_steps=25,
    restraint_weight=10.0,
    scorer="softdock",
    top_k=10,
)
```

The canonical multi-interaction form is an ordered non-empty list of mappings:

```python
interactions = [
    {
        "receptor_residue": "ASP189:A",
        "receptor_atom": "OD1",
        "ligand_smarts": "[N:1]",
        "target_distance": 3.0,
        "distance_tolerance": 0.5,
    },
    {
        "receptor_residue": "SER190:A",
        "receptor_atom": "OG",
        "ligand_smarts": "[O:1]",
        "target_distance": 2.9,
        "distance_tolerance": 0.4,
        "restraint_weight": 12.0,
    },
]
```

Each item has exactly the five selector/geometry fields shown above and may add `restraint_weight`; omitted item weights use the top-level value. Typed callers may use `InteractionConstraint` objects imported from `anchor_dock` in the same list. At most eight items are accepted. Exact duplicate items and supplying `interactions` together with any of the five single-interaction keyword arguments fail closed. The existing five-keyword form remains source-compatible and normalizes to a one-item list.

`receptor_residue` accepts an exact residue such as `ASP189:A` (or an omitted chain only when the residue is unique), and `receptor_atom` is an exact standard-PDB atom name. Alternate locations, duplicate or ambiguous selections, hydrogens, and absent atoms fail closed. Multi-interaction scoring uses the union of residue-centered pockets around every selected receptor residue.

Each `ligand_smarts` must contain exactly one mapped `:1` query atom. Matching is performed independently on the canonical heavy-atom ligand: all distinct matching atoms are enumerated automatically, zero matches fail, and more than `max_matches=16` matches for any selector fails instead of truncating. Multi runs form deterministic Cartesian joint assignments and fail above `max_joint_matches=64`. The same ligand atom may satisfy multiple constraints; exact atom correspondence is preserved in each joint assignment.

For each conformer and joint assignment, conservative pairwise shell-distance checks discard only provably infeasible combinations. The selector with the fewest ligand matches is the deterministic primary placement anchor. Its ligand atom is seeded at its target distance, while every secondary flat-bottom violation participates in coarse preselection; quotas remain fair across viable assignments and conformers.

All items have `ALL`/`AND` semantics. Guidance adds the mean of their weighted flat-bottom penalties,

```text
E_guide = E_search + mean_i(weight_i * relu(abs(distance_i-target_i)-tolerance_i)^2)
```

so a one-item run retains the 0.4 objective exactly. Restraints are removed for release on the same live pose model. Export requires every distance window, and survivors are ranked solely by the scorer's unmodified score. `OR`, `ANY`, and k-of-n semantics are unsupported; alternatives must be separate jobs. Distance hypotheses do not imply an interaction type, compatible chemistry, protonation state, or tautomer.

## Batch

`dock_batch` accepts homogeneous ligand sources or mixed `DockingJob` objects. Input content, metadata, effective options, scorer state/config/name/units, resolved device, NumPy/RDKit/Torch versions, and output artifact content are checked before resume. Recursive directory discovery excludes the output subtree, while source/output and reserved-manifest collisions fail before output is replaced. See [BATCH.md](BATCH.md).

## Results and output

Successful reference, covalent, and interaction calls return a dictionary containing mode/version, output, pose counts, best score/search energy, scorer identity, receptor/source fingerprints, intramolecular reference, runtime, and device. `search_parameters` records the effective search options; reconstruction also requires the adjacent input, scorer, version, fingerprint, and device fields. Torsion and optimization fields distinguish requested, actually applied, and improved behavior. Multi-interaction results additionally record the ordered specifications, resolved receptor atoms, per-selector ligand matches, selected joint assignment, target windows, per-interaction initial/guided/final distances, guide/release statistics, restraint formula and weights, and protonation limitation. One-item calls retain the 0.4 single-interaction result fields. `dock_batch` adds job and artifact fingerprints plus `batch_runtime_identity`.

SDF properties use only the versioned `AnchorDock_*` schema, including:

- `AnchorDock_Version`, `AnchorDock_Output_Schema`;
- `AnchorDock_Rank`, `AnchorDock_Pose_ID`;
- `AnchorDock_Scorer`, `AnchorDock_Scorer_Fingerprint`;
- `AnchorDock_Score`, `AnchorDock_Search_Energy`, `AnchorDock_Score_Units`, `AnchorDock_Score_Semantics`.
- `AnchorDock_Receptor_Structure_Fingerprint`, `AnchorDock_Receptor_Source_Fingerprint`;
- covalent `AnchorDock_Receptor_Reactant_Structure_Fingerprint` and `AnchorDock_Covalent_Receptor_Typing_*` fields;
- multi-interaction ordered receptor/ligand selector provenance, joint assignments, target windows, per-pose distances, and restraint metadata;
- `AnchorDock_Intramolecular_Reference`, torsion/optimization truth fields, and `AnchorDock_Search_Parameters`.

Ligand and reference inputs must each contain one connected component. Salts and mixtures fail explicitly; no implicit desalting policy is applied. Single-molecule SDF loaders (`load_ligand`, `load_reference_ligand`) require exactly one SDF record and point multi-ligand inputs to `dock_batch`. Output schema `4` adds ordered multi-interaction provenance while retaining the true RDKit conformer ID in `AnchorDock_Source_Conformer` and its representative ordinal in `AnchorDock_Source_Representative_Index`.

The advanced shared engine is available as `DockingEngine`. Removed 0.2 façade classes and internal cache hooks are not public API.
