# AnchorDock

AnchorDock is a Torch-native ligand pose engine with three explicit search strategies:

- **reference**: transfer one or more MCS anchors from a known ligand;
- **covalent**: construct a residue–warhead adduct and preserve its reaction geometry;
- **free**: randomized multistart local optimization over translation, rotation, and torsions.

All three strategies use the same scorer interface, rigid-frame kinematics, pose optimizer, receptor context, output schema, and heterogeneous batch runner. The only Python namespace is `anchor_dock`.

## Installation

```bash
uv sync --group dev
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

## Free local docking

```python
from anchor_dock import dock_free

result = dock_free(
    protein_pdb="pocket.pdb",
    query_ligand="CCO",
    center=(12.0, -3.0, 8.0),
    box_size=(20.0, 20.0, 20.0),
    num_starts=256,
    scorer="softdock",
)
```

This is a seeded multistart **local** search with Haar-uniform SO(3) starts, not a reproduction of AutoDock Vina's global Monte-Carlo search.

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
    DockingJob.free(
        "CCN",
        id="free-001",
        protein_pdb="pocket.pdb",
        num_starts=128,
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
from anchor_dock import dock_free

class MyScorer(nn.Module):
    def forward(self, ligand_coords, receptor_coords, ligand_features, receptor_features):
        distances = torch.cdist(
            ligand_coords,
            receptor_coords.unsqueeze(0).expand(ligand_coords.shape[0], -1, -1),
        )
        return -torch.exp(-distances).sum(dim=(1, 2))

result = dock_free(
    "pocket.pdb",
    "CCO",
    scorer=MyScorer(),
    num_starts=64,
)
```

## Score interpretation

The Vina and Vinardo backends follow the modern AutoDock Vina implementation's pair functions, defaults, radii, 8 Å cutoff, and torsion transform. SMILES/SDF/PDB inputs do not carry authoritative PDBQT XS types, so AnchorDock infers XS-like types and labels the score as `kcal/mol-like`, not an exact Vina affinity. See [docs/SCORING.md](docs/SCORING.md).

Covalent scores are conditioned on an already formed adduct. Free-mode scores rank local starts. Scores from different scorers or search modes should not be mixed without calibration.

## Upgrading from 0.2

Version 0.3 keeps a one-release `FutureWarning` adapter for unambiguous 0.2 call names and keywords. It does **not** preserve 0.2 score values or SDF tags: atom typing, pair masks, score reporting, covalent geometry, and the output schema changed. The unsupported `vina_lp` preset and old low-level namespaces fail explicitly. See [docs/MIGRATION.md](docs/MIGRATION.md).

## Development

```bash
uv lock --check
uv sync --frozen --group dev
uv run ruff check src tests examples
uv run pytest -q
uv build --wheel
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Scoring](docs/SCORING.md)
- [Batch inputs and manifests](docs/BATCH.md)
- [Python and CLI usage](docs/USAGE.md)
- [API reference](docs/API_REFERENCE.md)
- [0.2 to 0.3 migration](docs/MIGRATION.md)
- [Examples](examples/README.md)

## License

MIT
