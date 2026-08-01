# AnchorDock

AnchorDock is a unified ligand-pose prediction package built around **explicit geometric anchors**. It supports two complementary modes on one shared differentiable scoring and torsion-refinement engine:

- **Reference mode** — transfers a query ligand onto a known reference ligand through single, symmetry-aware multi-position, or multi-fragment MCS anchors.
- **Covalent mode** — detects an electrophilic warhead, constructs a residue-linked adduct, anchors it to a protein nucleophile, and explores the remaining conformational degrees of freedom.

The former `lig-mcs-align` and `cov-vina` implementations are now maintained together so scoring, masks, conformer handling, batched kinematics, optimization, pocket caching, and pose export cannot drift independently.

## Python API

### Reference-guided docking

```python
from anchor_dock import dock_reference

result = dock_reference(
    protein_pdb="pocket.pdb",
    ref_ligand="reference.sdf",
    query_ligand="CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    mcs_mode="auto",
    optimize=True,
)
```

### Covalent docking

```python
from anchor_dock import dock_covalent

result = dock_covalent(
    protein_pdb="protein.pdb",
    query_ligand="C=CC(=O)NCC",
    reactive_residue="CYS145:A",
    optimize=True,
)
```

### Batch covalent docking

```python
from anchor_dock import dock_covalent_batch

results = dock_covalent_batch(
    protein_pdb="protein.pdb",
    ligands=["C=CC(=O)NC", "O=CCc1ccccc1"],
    reactive_residue="CYS145:A",
)
```

## Compatibility

Existing code remains valid:

```python
from lig_align import run_pipeline, run_batch
from cov_vina import run_covalent_pipeline, run_batch_docking
```

New code should use `anchor_dock` directly. See [docs/MIGRATION.md](docs/MIGRATION.md).

## Shared architecture

```text
anchor_dock/
├── core/          # scoring, features, masks, conformers, kinematics, optimizer, I/O
├── reference/     # reference-ligand MCS strategy
└── covalent/      # warhead/residue strategy and adduct construction

lig_align/         # backward-compatible reference API
cov_vina/          # backward-compatible covalent API
```

Both modes use the same five-term Vina/Vinardo-style non-bonded score, interaction precomputation, pairwise exclusion masks, batched forward kinematics, and gradient optimization. Mode-specific code is limited to generating the initial anchored pose ensemble and constraints.

## Score interpretation

AnchorDock scores are **pose-ranking scores conditioned on the selected anchor**. In covalent mode, interactions around the already-formed bond are excluded to avoid double-counting and severe artificial clashes. The reported number is not a covalent reaction free energy and should not be compared directly with an unconstrained reference-mode score.

Results include:

- `mode`
- `score_semantics`
- anchor and warhead metadata where applicable
- initial/final Vina score and optimization delta
- all saved poses sorted by ascending score

## CLI

```bash
anchor-dock reference \
  -p pocket.pdb -r reference.sdf -q "SMILES" --optimize

anchor-dock covalent \
  -p protein.pdb -r CYS145:A -q "C=CC(=O)NCC" --optimize
```

## Development

```bash
uv sync --group dev
uv run pytest
```

The test suite covers the shared scorer and gradients, single/batched kinematics, legacy imports, warhead/adduct transforms, PDB metadata preservation, and an end-to-end covalent smoke test.

## License

MIT
