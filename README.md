# AnchorDock

AnchorDock is a Torch-native ligand pose engine with three explicit search strategies:

- **reference**: transfer one or more MCS anchors from a known ligand;
- **covalent**: construct a residue–warhead adduct and preserve its reaction geometry;
- **interaction**: test one or more explicit receptor-atom/ligand-atom distance hypotheses with bounded local search.

All three strategies use the same scorer interface, rigid-frame kinematics, pose optimizer, receptor context, output schema, and heterogeneous batch runner. The only Python namespace is `anchor_dock`.

## Installation

```bash
uv sync --frozen --group dev
```

Python 3.12 or newer is required.

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
    scorer="vina",           # vina | vinardo | softdock
    device="cuda",
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
    scorer="softdock",
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
    scorer="softdock",
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

```python
import torch
import torch.nn as nn
from anchor_dock import dock_interaction

class MyScorer(nn.Module):
    def forward(self, ligand_coords, receptor_coords, ligand_features, receptor_features):
        distances = torch.cdist(
            ligand_coords,
            receptor_coords.unsqueeze(0).expand(ligand_coords.shape[0], -1, -1),
        )
        return -torch.exp(-distances).sum(dim=(1, 2))

result = dock_interaction(
    "pocket.pdb",
    "CCO",
    receptor_residue="ASP189:A",
    receptor_atom="OD1",
    ligand_smarts="[O:1]",
    target_distance=3.0,
    distance_tolerance=0.5,
    scorer=MyScorer(),
)
```

## Score interpretation

The Vina and Vinardo backends follow the modern AutoDock Vina implementation's pair functions, defaults, radii, 8 Å cutoff, and torsion transform. SMILES/SDF/PDB inputs do not carry authoritative PDBQT XS types, so AnchorDock infers XS-like types and labels the score as `kcal/mol-like`, not an exact Vina affinity. See [docs/SCORING.md](docs/SCORING.md).

Covalent scores are conditioned on an already formed adduct. Interaction-mode scores rank poses that survived every requested atom-pair distance filter; guide penalties are never included in `AnchorDock_Score` or `AnchorDock_Search_Energy`. Scores from different scorers or search modes should not be mixed without calibration.

## Upgrading

Version 0.5 adds bounded simultaneous `AND` interaction constraints while keeping 0.4 single-interaction calls source-compatible. It advances the SDF output schema to `4` and batch resume epoch to `5`. Version 0.4 removed the public unconstrained search; all earlier transitions remain documented in [docs/MIGRATION.md](docs/MIGRATION.md).

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
- [Batch inputs and manifests](docs/BATCH.md)
- [Python and CLI usage](docs/USAGE.md)
- [API reference](docs/API_REFERENCE.md)
- [Migration](docs/MIGRATION.md)
- [Examples](examples/README.md)

## License

MIT
