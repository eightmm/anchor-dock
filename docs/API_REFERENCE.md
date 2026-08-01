# API Reference

## High-Level Python API

Main entry point:

```python
from anchor_dock import dock_reference

results = dock_reference(
    protein_pdb="protein.pdb",
    ref_ligand="ref.sdf",
    query_ligand="SMILES",
    output_dir="output",
    num_confs=1000,
    rmsd_threshold=1.0,
    optimize=True,
    optimizer="lbfgs",
    verbose=True,
)
```

Typical return keys:

```python
{
    "output_file": "output/predicted_pose_top3.sdf",
    "num_poses": 3,
    "best_score": -6.038,
    "runtime": 23.5,
    "num_conformers": 1000,
    "num_representatives": 22,
    "mcs_size": 10,
    "mcs_positions": 1,
    "canonical_smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "device": "cuda",
}
```

## Key Parameters

```text
protein_pdb        Protein PDB path
ref_ligand         Reference SDF path
query_ligand       Query SMILES or SDF path
output_dir         Output directory
num_confs          Number of conformers to generate
rmsd_threshold     RMSD clustering threshold
mcs_mode           auto | single | multi | cross
optimize           Enable torsion optimization
optimizer          adam | adamw | lbfgs
opt_steps          Number of optimization steps
opt_lr             Optimization learning rate
opt_batch_size     Number of poses processed per optimization batch (default: 128)
freeze_mcs         Keep MCS atoms fixed during optimization
weight_preset      vina | vina_lp | vinardo
torsion_penalty    Apply torsional entropy penalty (default: True)
verbose            Print progress
```

Current batching note:

- `opt_batch_size` currently batches multiple poses of the same molecule
- it is not yet a fully vectorized mixed-molecule optimizer

## Low-Level API

For stepwise control, use `LigandAligner`.

```python
from anchor_dock.reference import LigandAligner

aligner = LigandAligner(device="cuda")
mapping = aligner.step2_find_mcs(ref_mol, query_mol)
query_mol, rep_cids = aligner.step1_generate_conformers(
    query_mol,
    num_confs=1000,
    rmsd_threshold=1.0,
)
aligned = aligner.step3_batched_kabsch_alignment(ref_coords, query_coords, mapping)
scores = aligner.step4_vina_scoring(aligned, pocket_coords, query_feat, pocket_feat)
optimized = aligner.step6_refine_pose(
    query_mol,
    mcs_indices,
    aligned,
    pocket_coords,
    query_feat,
    pocket_feat,
    num_steps=100,
    batch_size=8,
)
```

## Script Inventory

- `scripts/run_pipeline.py`: end-to-end pose generation
- `scripts/optimize_pose.py`: optimize a single input pose
- `scripts/vis_comparison_grid.py`: generate comparison panels across optimization settings
- `scripts/vis_opt_gif.py`: make optimization animations
- `scripts/vis_ref_opt_gif.py`: compare reference-guided optimization trajectories
