# Examples

- `reference.py`: reference-MCS docking using the bundled 10gs inputs.
- `covalent.py`: command-line covalent example for a user-provided protein.
- `interaction.py`: explicit atom-pair interaction-guided example.
- `batch.py`: Python mixed-job batch example.
- `custom_scorer.py`: differentiable Torch scorer adapter.
- `batch/jobs.jsonl`, `batch/ligands.smi`, `batch/ligands.csv`: manifest/input templates.

Only `reference.py` is runnable with bundled molecular inputs. The covalent, interaction, batch, and custom-scorer files are templates that require user-provided receptor/reference paths and an appropriate explicit interaction hypothesis as shown in each file or manifest.

`batch/ligands.smi` is a homogeneous ligand list; supply the protein and all five interaction fields to `dock_batch` or the batch CLI. The JSONL and CSV templates carry those fields per interaction job/row.
