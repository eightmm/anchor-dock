# AnchorDock

AnchorDock is a Torch-native ligand pose engine with three explicit search strategies:

- **reference**: transfer one or more MCS anchors from a known ligand;
- **covalent**: construct a residue–warhead adduct and preserve its reaction geometry;
- **interaction**: test one or more explicit receptor-atom/ligand-atom distance hypotheses with bounded local search.

The three search strategies share scorer contracts, rigid-frame kinematics, pose optimization, receptor contexts, ranked output and heterogeneous batch execution. `refine_pose` is a separate operation for an already positioned ligand; it does not generate a binding-site placement. The only Python namespace is `anchor_dock`.

Prepared graph/PDBQT inputs, an analytic AD4/AutoGrid reference backend, explicit native Vina-family final scoring, and an opt-in geometry-aware optimizer are documented in [prepared inputs and map scoring](docs/PREPARED.md). These additions retain bounded local search; GPU throughput and native AutoDock-GPU end-to-end agreement require separate validation.

## Installation

```bash
uv sync --frozen --group dev
```

Python 3.12 or newer is required. Commands below use `uv run --frozen` so activating the virtual environment is not required. Use `--device cpu` for a CPU run.

Optional preparation and force-field dependencies:

```bash
uv sync --frozen --extra prepare --extra amber --group dev
```

The core installation supports inferred Vina/Vinardo scoring. `prepare` adds Meeko preparation; `amber` adds the topology importer and OpenMM reference. `native-vina` is optional numerical-oracle tooling. No external docking executable is needed by the named Torch scoring functions.

## Scoring-function refinement

Version 0.6.2 focuses on differentiable scoring in Torch: Vina/Vinardo, AD4/AutoDock-GPU map references, and an all-atom AMBER vacuum potential. Existing positioned SDF poses can be refined independently under each function. The named `amber` backend uses `TorchAmberScorer`: it imports an explicit topology once through OpenMM, then evaluates energies and gradients in Torch. `OpenMMScorer` remains a separate advanced Python backend and numerical reference.

```bash
uv run --frozen anchor-dock scorers
uv run --frozen anchor-dock refine \
  -p examples/10gs/10gs_pocket.pdb -l examples/10gs/10gs_ligand.sdf \
  --scorers vina vinardo --steps 20 --seed 713 --device cpu -o sf_comparison
```

See [scoring functions, preparation and plugins](docs/SCORING_FUNCTIONS.md). AMBER requires a complete parameterized topology and explicit all-atom mapping; AD4/AutoDock-GPU scoring requires prepared inputs and maps. The Torch AMBER implementation currently supports nonperiodic vacuum NoCutoff systems; implicit solvent and periodic electrostatics require a different implementation. Smina/GNINA, native Vina optimizer adapters and DKOES presets have been removed. Custom models can still use `NeuralScorerAdapter`.

## Reference-guided docking

```python
from anchor_dock import dock_reference

result = dock_reference(
    protein_pdb="pocket.pdb",
    reference_ligand="known_pose.sdf",
    query_ligand="CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    mcs_mode="auto",          # auto | single | multi | cross
    relax=True,               # one fixed-anchor MMFF/UFF relaxation
    optimize=True,
    scorer="vina",           # vina | vinardo
    device="cpu",
)
```

Every selected mapping is embedded independently. Pattern automorphisms recover symmetry-related correspondences, bounded candidate completeness is reported explicitly, and all poses share one recorded intramolecular reference. Cross-fragment search explores bounded alternative decompositions; its output never claims a globally exhaustive or maximum-size fragment packing.

## Covalent docking

```python
from anchor_dock import dock_covalent

result = dock_covalent(
    protein_pdb="protein.pdb",
    query_ligand="C=CC(=O)NCC",
    reactive_residue="CYS145:A",
    optimize=True,
    scorer="vina",
    device="cpu",
)
```

Covalent scoring retypes the bonded receptor nucleophile in product state on a copy of the cached pocket. Outputs preserve both reactant and scoring-structure fingerprints and record the versioned before/after type change.

The input protein's support and nucleophile coordinates are fixed first. Carbon electrophiles use validated residue-specific bond targets; sulfur, phosphorus, and other electrophiles use atom-pair distance-geometry bounds. A local geometry pass and rigid branch transform enforce the formed bond and protein-side 1–3 geometry without deforming the ligand branch. Omitting `reactive_residue` is accepted only when exactly one supported residue exists.

## Interaction-guided local docking

```python
from anchor_dock import dock_interaction

result = dock_interaction(
    protein_pdb="pocket.pdb",
    query_ligand="CCO",
    receptor_residue="ASP189:A",
    receptor_atom="OD1",
    ligand_smarts="[O:1]",
    target_distance=3.0,
    distance_tolerance=0.5,
    scorer="vina",
    device="cpu",
)
```

The mapped SMARTS must contain exactly one `:1` query atom. AnchorDock enumerates every distinct matching ligand atom up to the configured hard cap, samples seeded candidates around the selected receptor atom, and preselects fairly across matches and conformers. A flat-bottom distance guide is followed by restraint-free release; only poses still inside the requested distance window are exported and ranked by the unmodified scorer.

Multiple simultaneous constraints use the canonical ordered `interactions` list. Each item has the same five required fields and may override `restraint_weight`:

```python
result = dock_interaction(
    protein_pdb="pocket.pdb",
    query_ligand="NCCO",
    interactions=[
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
    ],
    max_joint_matches=64,
    scorer="vina",
    device="cpu",
)
```

All list items are mandatory (`AND`): every exported pose must satisfy every distance window. `OR`, `ANY`, and k-of-n semantics are not supported; run alternatives as separate jobs. At most eight items are accepted. Every SMARTS independently enumerates up to `max_matches=16` ligand anchors, and their deterministic Cartesian joint hypotheses must fit within `max_joint_matches=64` or the run fails without truncation. Exact duplicate specifications fail, while one ligand atom may legitimately satisfy more than one receptor constraint. The scored receptor context is the union of pockets around all selected residues. The interaction with the fewest ligand matches seeds primary placement, all remaining constraint violations influence preselection, guidance averages the weighted penalties, and the release and final ranking remain restraint-free and scorer-only.

These are generic atom-pair distance hypotheses, not automatic interaction-site detection or claims of hydrogen bonds, salt bridges, metal interactions, or pi interactions. AnchorDock automatically enumerates ligand anchors selected by each SMARTS, but does not choose receptor atoms or infer interaction type, protonation, tautomer, or chemical compatibility.

## One batch API

```python
from anchor_dock import DockingJob, dock_batch

jobs = [
    DockingJob.reference(
        "CCO",
        id="analog-001",
        protein_pdb="pocket.pdb",
        reference_ligand="known_pose.sdf",
        num_confs=256,
    ),
    DockingJob.covalent(
        "C=CC(=O)NCC",
        id="cov-001",
        protein_pdb="protein.pdb",
        reactive_residue="CYS145:A",
    ),
    DockingJob.interaction(
        "CCN",
        id="interaction-001",
        protein_pdb="pocket.pdb",
        receptor_residue="ASP189:A",
        receptor_atom="OD1",
        ligand_smarts="[N:1]",
        target_distance=3.0,
        distance_tolerance=0.5,
    ),
]

results = dock_batch(jobs, output_dir="screen", resume=True)
```

`dock_batch` also accepts individual ligands, RDKit molecules, iterables, generators, mappings, DataFrame-like objects, file-like objects, directories, SDF/SMI/CSV/TSV/JSON/JSONL, and gzip/bzip2/xz-compressed inputs. See [docs/BATCH.md](docs/BATCH.md).

Ligands and references must contain exactly one connected component. AnchorDock rejects salts and mixtures rather than silently selecting a fragment.

## Custom differentiable scorer

Custom Torch models can implement the shared prepared-scorer contract or use `NeuralScorerAdapter`. Registration declares preparation, units, score direction and a deterministic fingerprint. See [the extension contract](docs/SCORING_FUNCTIONS.md#optimization-and-custom-extensions); an arbitrary custom energy is not a calibrated binding affinity.

## Score interpretation

The Vina and Vinardo backends follow the modern AutoDock Vina implementation's pair functions, defaults, radii, 8 Å cutoff, and torsion transform. SMILES/SDF/PDB inputs do not carry authoritative PDBQT XS types, so AnchorDock infers XS-like types and labels the score as `kcal/mol-like`, not an exact Vina affinity. See [docs/SCORING.md](docs/SCORING.md).

Vina/Vinardo offer opt-in sparse intramolecular evaluation through `scorer_options={"intra_mode": "sparse"}` during refinement; dense evaluation remains the default. AD4 shares grid-cell and corner-weight calculations across its three map channels. Current numerical checks and executed examples are recorded in the [CPU validation report](reports/cpu-validation-2026-10-30.md). The [0.6.1 physical-scoring snapshot](reports/torch-physical-2026-10-30.md) retains its measured workloads and timings; GPU speed has not been validated.

Covalent scores are conditioned on an already formed adduct. Interaction-mode scores rank poses that survived every requested atom-pair distance filter; guide penalties are never included in `AnchorDock_Score` or `AnchorDock_Search_Energy`. Scores from different scorers or search modes should not be mixed without calibration.

## Upgrading

Version 0.6.2 hardens prepared-state/fingerprint isolation and inactive force-field terms after CPU validation. Batch resume epoch is `8`. Ranked docking SDF schema remains `4`; standalone SF result schemas remain at `v1`. Version 0.6.1 introduced pure Torch AMBER vacuum scoring and the explicit AutoDock-GPU map formulation, and removed native optimizer adapters and DKOES presets.

Version 0.6 removed SoftDock, changed the interaction default to Vina and added independent SF refinement. Earlier transitions are documented in [docs/MIGRATION.md](docs/MIGRATION.md).

Version 0.5 added bounded simultaneous `AND` interaction constraints while keeping 0.4 single-interaction calls source-compatible. It advanced the SDF output schema to `4` and batch resume epoch to `5`. Version 0.4 removed the public unconstrained search. See [migration notes](docs/MIGRATION.md) for the complete version history.

## Development

```bash
uv lock --check
uv sync --frozen --group dev
uv run --frozen ruff check src tests examples
uv run --frozen pytest -q
uv build --wheel
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Scoring](docs/SCORING.md)
- [Scoring-function catalog and refinement](docs/SCORING_FUNCTIONS.md)
- [Prepared chemistry and maps](docs/PREPARED.md)
- [Batch inputs and manifests](docs/BATCH.md)
- [Python and CLI usage](docs/USAGE.md)
- [API reference](docs/API_REFERENCE.md)
- [Migration](docs/MIGRATION.md)
- [Examples](examples/README.md)

## License

MIT

