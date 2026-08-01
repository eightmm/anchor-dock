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

The `lig_align` and `cov_vina` namespaces have been removed. Everything lives
under `anchor_dock`:

```python
from anchor_dock import dock_reference, dock_reference_batch
from anchor_dock import dock_covalent, dock_covalent_batch

# the reference implementation, if you need it directly
from anchor_dock.reference import LigandAligner, run_batch, run_pipeline
```

See [docs/MIGRATION.md](docs/MIGRATION.md) for the full mapping.

## Shared architecture

```text
anchor_dock/
├── core/          # scoring, features, masks, conformers, kinematics, optimizer, I/O
├── reference/     # reference-ligand MCS strategy: pipeline, MCS, conformers, I/O, export
└── covalent/      # warhead/residue strategy and adduct construction
```

Both modes share one five-term Vina/Vinardo-style non-bonded score, interaction
precomputation, pairwise exclusion mask builder, batched forward kinematics, and
gradient optimizer, all from `anchor_dock.core`. Mode-specific code is limited to
building the initial anchored pose ensemble and its constraints.

One duplication is still deliberate. `anchor_dock.reference` keeps its own
`conformers`, `io`, and `output` modules rather than reusing the `core`
equivalents, because the two are not interchangeable:

- reference MMFF-relaxes cluster representatives; `core.conformers` does not,
  because MMFF is unreliable on covalent adduct topologies
- reference attaches explicit hydrogens at load time, which its MCS and
  rotatable-bond accounting assume; the covalent strategy adds them at embed time
- reference writes the `MCS_Position` tag, which has no covalent counterpart

Merging them would change pose geometry, so it is an open decision rather than a
refactor. `tests/test_reference_regression.py` pins the current numbers so any
attempt has to prove it.

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
