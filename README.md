# LigAlign

A ligand pose prediction pipeline that uses reference ligand coordinates as a geometric anchor.
MCS (Maximum Common Substructure) matching → conformer generation → Vina scoring → gradient-based torsion optimization.

## Pipeline Overview

```
Input: Protein PDB + Reference SDF + Query (SMILES/SDF)
                        │
                  MCS Search ─────── ref↔query atom mapping (auto/single/multi/cross)
                        │
               Conformer Generation ─ ETKDGv3 with MCS atoms pinned to ref coords
                        │
                RMSD Clustering ──── representative selection (Butina)
                        │
              MMFF Relaxation ────── force field optimization for non-MCS atoms
                        │
                Vina Scoring ─────── 5-term differentiable energy function
                        │
           Torsion Optimization ──── (optional) gradient-based torsion refinement
                        │
                  Export SDF ─────── all poses sorted by energy (ascending)
```

All poses are saved to `predicted_poses.sdf`, sorted by energy (best first).
For multi-position MCS modes, poses from all positions are pooled and ranked together.

## Quick Start

### Environment

```bash
uv venv
source .venv/bin/activate
uv sync
```

### CLI

```bash
uv run python scripts/run_pipeline.py \
  -p examples/10gs/10gs_pocket.pdb \
  -r examples/10gs/10gs_ligand.sdf \
  -q "CC(C)Cc1ccc(cc1)C(C)C(=O)O" \
  -o output/
```

### Python API

```python
from lig_align import run_pipeline

results = run_pipeline(
    protein_pdb="examples/10gs/10gs_pocket.pdb",
    ref_ligand="examples/10gs/10gs_ligand.sdf",
    query_ligand="CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    output_dir="output",
)

print(results["best_score"])   # kcal/mol
print(results["output_file"])  # predicted_poses.sdf
print(results["num_poses"])    # number of saved poses
```

### Batch Processing

```python
from lig_align import run_batch

results = run_batch(
    protein_pdb="pocket.pdb",
    ref_ligand="reference.sdf",
    query_ligands=["CCO", "c1ccccc1", "CC(=O)O"],
    optimize=True,
)
```

## Options Reference

### Conformer Generation

| Option | CLI | Default | Description |
|--------|-----|---------|-------------|
| `num_confs` | `-n` | 1000 | Number of conformers to generate |
| `rmsd_threshold` | `--rmsd_threshold` | 1.0 | RMSD clustering threshold in Angstrom |

**Controlling pose diversity**: Because MCS atoms are pinned to reference coordinates,
there is no rotation augmentation. Pose diversity comes entirely from ETKDG torsion
angle sampling. To increase diversity, raise `num_confs` or lower `rmsd_threshold`.

### MCS Matching

| Option | CLI | Default | Description |
|--------|-----|---------|-------------|
| `mcs_mode` | `--mcs_mode` | `auto` | MCS search mode |
| `min_fragment_size` | `--min_fragment_size` | 5 | Minimum fragment size for cross mode |
| `max_fragments` | `--max_fragments` | 3 | Maximum number of fragments for cross mode |

**Mode selection guide:**

- **`auto`** (recommended): Analyzes molecular symmetry and fragment distribution to pick the best mode automatically.
- **`single`**: Fastest. Uses one best contiguous MCS mapping.
- **`multi`**: Explores all symmetric MCS placements on the reference. Useful for symmetric scaffolds (e.g. benzene ring mappings).
- **`cross`**: Combines multiple non-contiguous fragments. Useful when query and reference share disconnected substructures.

In `multi`/`cross` modes, conformers are generated independently per MCS position.
All poses are pooled and ranked by energy together.

### Force Field Relaxation

| Option | CLI | Default | Description |
|--------|-----|---------|-------------|
| `mmff_optimize` | `--no_mmff` (to disable) | `True` | Apply MMFF94 relaxation to non-MCS atoms |

Falls back to UFF automatically if MMFF94 fails.
Skipped when all atoms are MCS-mapped (nothing to relax).

### Vina Scoring

| Option | CLI | Default | Description |
|--------|-----|---------|-------------|
| `weight_preset` | `--weight_preset` | `vina` | Energy function weight preset |
| `torsion_penalty` | `--no_torsion_penalty` (to disable) | `True` | Include torsional entropy penalty |

**Presets:**
- `vina`: AutoDock Vina default weights
- `vina_lp`: local preference weights
- `vinardo`: Vinardo weights

Uses a 5-term energy model: gauss1, gauss2, repulsion, hydrophobic, hydrogen bond.
With `torsion_penalty=True`, an entropy term proportional to the number of rotatable bonds is added.

### Gradient-Based Torsion Optimization

| Option | CLI | Default | Description |
|--------|-----|---------|-------------|
| `optimize` | `--optimize` | `False` | Enable gradient-based torsion optimization |
| `optimizer` | `--optimizer` | `adam` | Optimizer type (`adam`, `adamw`, `lbfgs`) |
| `opt_steps` | `--opt_steps` | 100 | Number of optimization steps |
| `opt_lr` | `--opt_lr` | 0.05 | Learning rate |
| `opt_batch_size` | `--opt_batch_size` | 128 | Batch size (reduce if OOM) |
| `freeze_mcs` | `--free_mcs` (to unfreeze) | `True` | Keep MCS atoms fixed during optimization |

**Usage guide:**
- Scoring-based ranking works without `--optimize`
- Enabling `--optimize` typically improves scores by 0.3-1.0 kcal/mol on average
- `lbfgs` converges to higher quality solutions but is slower than `adam`
- Keep `freeze_mcs=True` (default) to preserve reference anchoring
- Reduce `opt_batch_size` for large ligands that cause OOM

### Full CLI Example

```bash
uv run python scripts/run_pipeline.py \
  -p protein.pdb \
  -r ref_ligand.sdf \
  -q "SMILES_or_path.sdf" \
  -o output_dir \
  -n 2000 \
  --rmsd_threshold 0.8 \
  --mcs_mode auto \
  --optimize \
  --optimizer adam \
  --opt_steps 200 \
  --opt_lr 0.05 \
  --opt_batch_size 64 \
  --weight_preset vina \
  --torsion_penalty
```

## Output

### SDF File

`predicted_poses.sdf` — all poses sorted by energy in ascending order (best first).

Per-pose properties:

| Property | Description |
|----------|-------------|
| `Vina_Score` | Final Vina energy (kcal/mol) |
| `Vina_Score_Initial` | Pre-optimization energy |
| `Vina_Score_Delta` | Energy change from optimization |
| `Rank` | Energy ranking (1 = best) |
| `MCS_Position` | MCS placement index (multi/cross modes only) |

Molecule-level properties:

| Property | Description |
|----------|-------------|
| `MCS_Num_Atoms` | Number of MCS-matched atoms |
| `MCS_Ref_Coverage` / `MCS_Query_Coverage` | MCS coverage (%) |
| `LigAlign_MCS_Mode` | Resolved MCS mode actually used |
| `LigAlign_Gradient_Optimized` | Whether optimization was performed |

### Return Dict (Python API)

```python
{
    "output_file": "output/predicted_poses.sdf",
    "num_poses": 42,
    "best_score": -7.231,
    "runtime": 3.14,
    "num_conformers": 1000,
    "num_representatives": 42,
    "mcs_size": 12,
    "mcs_positions": 1,
    "best_position": 1,
    "canonical_smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "device": "cuda",
}
```

## Project Layout

```
lig-align/
├── src/lig_align/          # core package
│   ├── pipeline.py         # run_pipeline(), run_batch()
│   ├── aligner.py          # LigandAligner (step-by-step API)
│   ├── molecular/          # MCS, conformer, MMFF relaxation, features
│   ├── scoring/            # Vina 5-term scoring, weight presets, masks
│   ├── optimization/       # gradient-based torsion optimization
│   ├── alignment/          # Kabsch alignment, forward kinematics
│   ├── selection/          # pose export
│   └── io/                 # input parsing, pocket loading
├── scripts/                # CLI entry points, visualization
├── tests/                  # regression and feature tests
├── examples/10gs/          # sample protein/ligand/output
├── docs/                   # detailed documentation
└── reports/                # progress and presentation assets
```

## Further Documentation

- [docs/USAGE.md](docs/USAGE.md): Installation and common workflows
- [docs/API_REFERENCE.md](docs/API_REFERENCE.md): Python and script-level interfaces
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): Pipeline stages and implementation notes
