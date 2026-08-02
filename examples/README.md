# Examples

- `reference.py`: reference-MCS docking using the bundled 10gs inputs.
- `covalent.py`: command-line covalent example for a user-provided protein.
- `free.py`: command-line multistart local docking example.
- `batch.py`: Python mixed-job batch example.
- `custom_scorer.py`: differentiable Torch scorer adapter.
- `batch/jobs.jsonl`, `batch/ligands.smi`, `batch/ligands.csv`: manifest/input templates.

Only `reference.py` is runnable with bundled molecular inputs. The covalent, free, batch, and custom-scorer files are templates that require user-provided receptor/reference paths as shown in each file or manifest.
