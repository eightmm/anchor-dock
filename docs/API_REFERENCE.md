# API reference

AnchorDock 0.3 exposes four high-level operations from `anchor_dock`:

```python
from anchor_dock import dock_reference, dock_covalent, dock_free, dock_batch
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

## Free local search (selected options)

```python
dock_free(
    protein_pdb,
    query_ligand,
    output_dir="anchor_dock_free",
    *,
    center=None,
    box_size=(20.0, 20.0, 20.0),
    num_confs=64,
    num_starts=128,
    optimize=True,
    scorer="softdock",
    top_k=20,
)
```

Free mode uses seeded Haar-uniform SO(3) starts for multistart local optimization; it is not Vina global search.

## Batch

`dock_batch` accepts homogeneous ligand sources or mixed `DockingJob` objects. Input content, metadata, effective options, scorer state/config/name/units, resolved device, NumPy/RDKit/Torch versions, and output artifact content are checked before resume. Recursive directory discovery excludes the output subtree, while source/output and reserved-manifest collisions fail before output is replaced. See [BATCH.md](BATCH.md).

## Results and output

Successful reference, covalent, and free calls return a dictionary containing mode/version, output, pose counts, best score/search energy, scorer identity, receptor/source fingerprints, intramolecular reference, runtime, and device. `search_parameters` records the effective search options; reconstruction also requires the adjacent input, scorer, version, fingerprint, and device fields. Torsion and optimization fields distinguish requested, actually applied, and improved behavior. `dock_batch` adds job and artifact fingerprints plus `batch_runtime_identity`.

SDF properties use only the versioned `AnchorDock_*` schema, including:

- `AnchorDock_Version`, `AnchorDock_Output_Schema`;
- `AnchorDock_Rank`, `AnchorDock_Pose_ID`;
- `AnchorDock_Scorer`, `AnchorDock_Scorer_Fingerprint`;
- `AnchorDock_Score`, `AnchorDock_Search_Energy`, `AnchorDock_Score_Units`, `AnchorDock_Score_Semantics`.
- `AnchorDock_Receptor_Structure_Fingerprint`, `AnchorDock_Receptor_Source_Fingerprint`;
- covalent `AnchorDock_Receptor_Reactant_Structure_Fingerprint` and `AnchorDock_Covalent_Receptor_Typing_*` fields;
- `AnchorDock_Intramolecular_Reference`, torsion/optimization truth fields, and `AnchorDock_Search_Parameters`.

Ligand and reference inputs must each contain one connected component. Salts and mixtures fail explicitly; no implicit desalting policy is applied.

The advanced shared engine is available as `DockingEngine`. Removed 0.2 façade classes and internal cache hooks are not public API.
