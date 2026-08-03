# Examples

- `reference.py`: reference-MCS docking using the bundled 10gs inputs.
- `covalent.py`: command-line covalent example for a user-provided protein.
- `interaction.py`: explicit atom-pair interaction-guided example.
- `interaction_multi.py`: two simultaneous `AND` interaction constraints using the canonical list form.
- `batch.py`: Python mixed-job batch example.
- `custom_scorer.py`: differentiable Torch scorer adapter.
- `batch/jobs.jsonl`, `batch/ligands.smi`, `batch/ligands.csv`: manifest/input templates.

Only `reference.py` is runnable with bundled molecular inputs. The covalent, interaction, interaction-multi, batch, and custom-scorer files are templates that require user-provided receptor/reference paths and appropriate explicit interaction hypotheses as shown in each file or manifest.

`batch/ligands.smi` is a homogeneous ligand list; supply the protein and either all five single-interaction fields or a canonical `interactions` list to `dock_batch` or the batch CLI. The JSONL and CSV templates carry the single fields per interaction job/row; JSON/JSONL is preferred for nested multi-interaction lists.
