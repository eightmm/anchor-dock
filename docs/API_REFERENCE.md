# API reference

AnchorDock 0.4 exposes four high-level operations from `anchor_dock`:

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
    receptor_residue,
    receptor_atom,
    ligand_smarts,
    target_distance,
    distance_tolerance,
    pocket_cutoff=12.0,
    include_heteroatoms=True,
    num_confs=32,
    num_candidates=128,
    preselect_k=16,
    max_matches=16,
    optimize=True,
    opt_steps=50,
    release_steps=25,
    restraint_weight=10.0,
    scorer="softdock",
    top_k=10,
)
```

`receptor_residue` accepts an exact residue such as `ASP189:A` (or an omitted chain only when the residue is unique), and `receptor_atom` is an exact standard-PDB atom name. Alternate locations, duplicate or ambiguous selections, hydrogens, and absent atoms fail closed.

`ligand_smarts` must be valid SMARTS with exactly one mapped query atom whose map number is `:1`. Matching is performed on the canonical heavy-atom ligand. All distinct matching ligand atoms are hypotheses; zero matches fail, and exceeding `max_matches` fails instead of truncating. Each selected atom is initially placed at `target_distance`, candidates are coarse-scored and preselected across matches/conformers, then guided optimization and restraint-free release run on the same live pose model. Final poses outside `target_distance ± distance_tolerance` are rejected.

The restraint is `weight * relu(abs(distance-target)-tolerance)^2`. It is guide metadata, never part of the reported physical score or search energy. This mode encodes a generic atom-pair distance hypothesis and performs no interaction-type, protonation, tautomer, or chemical-compatibility inference.

## Batch

`dock_batch` accepts homogeneous ligand sources or mixed `DockingJob` objects. Input content, metadata, effective options, scorer state/config/name/units, resolved device, NumPy/RDKit/Torch versions, and output artifact content are checked before resume. Recursive directory discovery excludes the output subtree, while source/output and reserved-manifest collisions fail before output is replaced. See [BATCH.md](BATCH.md).

## Results and output

Successful reference, covalent, and interaction calls return a dictionary containing mode/version, output, pose counts, best score/search energy, scorer identity, receptor/source fingerprints, intramolecular reference, runtime, and device. `search_parameters` records the effective search options; reconstruction also requires the adjacent input, scorer, version, fingerprint, and device fields. Torsion and optimization fields distinguish requested, actually applied, and improved behavior. Interaction results additionally record the resolved receptor atom, every ligand match, target window, initial/guided/final distances, guide/release statistics, restraint formula and weight, and protonation limitation. `dock_batch` adds job and artifact fingerprints plus `batch_runtime_identity`.

SDF properties use only the versioned `AnchorDock_*` schema, including:

- `AnchorDock_Version`, `AnchorDock_Output_Schema`;
- `AnchorDock_Rank`, `AnchorDock_Pose_ID`;
- `AnchorDock_Scorer`, `AnchorDock_Scorer_Fingerprint`;
- `AnchorDock_Score`, `AnchorDock_Search_Energy`, `AnchorDock_Score_Units`, `AnchorDock_Score_Semantics`.
- `AnchorDock_Receptor_Structure_Fingerprint`, `AnchorDock_Receptor_Source_Fingerprint`;
- covalent `AnchorDock_Receptor_Reactant_Structure_Fingerprint` and `AnchorDock_Covalent_Receptor_Typing_*` fields;
- interaction receptor/ligand selector provenance, target window, per-pose distances, and restraint metadata;
- `AnchorDock_Intramolecular_Reference`, torsion/optimization truth fields, and `AnchorDock_Search_Parameters`.

Ligand and reference inputs must each contain one connected component. Salts and mixtures fail explicitly; no implicit desalting policy is applied. Single-molecule SDF loaders (`load_ligand`, `load_reference_ligand`) require exactly one SDF record and point multi-ligand inputs to `dock_batch`. Output schema `3` records the true RDKit conformer ID in `AnchorDock_Source_Conformer` and its representative ordinal in `AnchorDock_Source_Representative_Index`.

The advanced shared engine is available as `DockingEngine`. Removed 0.2 façade classes and internal cache hooks are not public API.
