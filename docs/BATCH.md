# Batch execution

## Homogeneous inputs

```python
from anchor_dock import dock_batch

results = dock_batch(
    ["CCO", "CCN", "c1ccccc1"],
    mode="free",
    protein_pdb="pocket.pdb",
    output_dir="screen",
    num_starts=128,
)
```

A `(ligand, name)` tuple creates one named ligand. Lists and tuples otherwise represent collections.

## Mixed jobs

```python
from anchor_dock import DockingJob, dock_batch

jobs = [
    DockingJob.reference("CCO", protein_pdb="p.pdb", reference_ligand="r.sdf"),
    DockingJob.covalent("C=CC(=O)N", protein_pdb="p.pdb", reactive_residue="CYS145:A"),
    DockingJob.free("CCN", protein_pdb="p.pdb"),
]
results = dock_batch(jobs)
```

## JSONL manifest

```json
{"id":"ref-001","mode":"reference","smiles":"CCO","protein_pdb":"pocket.pdb","reference_ligand":"known.sdf","options":{"mcs_mode":"multi"}}
{"id":"cov-001","mode":"covalent","smiles":"C=CC(=O)NCC","protein_pdb":"protein.pdb","reactive_residue":"CYS145:A"}
{"id":"free-001","mode":"free","smiles":"CCN","protein_pdb":"pocket.pdb","options":{"num_starts":256,"scorer":"softdock"}}
```

```python
results = dock_batch("jobs.jsonl", output_dir="mixed", resume=True)
```

## Accepted sources

- SMILES, InChI, RDKit `Chem.Mol`, `LigandRecord`, or `DockingJob`;
- list, tuple, iterator, generator, mapping;
- DataFrame-like objects supporting `to_dict(orient="records")`;
- text or binary file-like objects;
- SDF, MOL, MOL2, PDB, SMI, SMILES, InChI, TXT;
- CSV, TSV, JSON, JSONL, NDJSON;
- directories containing supported files;
- gzip, bzip2, xz, and lzma compressed text or SDF inputs.

## Per-row options

CSV/TSV columns outside the identity and input fields are treated as docking
options. Booleans, numbers, JSON arrays and JSON objects are converted from text:

```csv
smiles,name,optimize,opt_steps,box_size,top_k
CCO,ethanol,true,100,"[20,20,20]",10
```

Arbitrary non-option information belongs in the explicit `metadata` object. A
JSON/JSONL row may also put all overrides inside an `options` object.

## Errors and resume

```python
results = dock_batch(
    source,
    on_error="record",  # record | raise | skip
    resume=True,
)
```

Every job receives a sanitized deterministic directory name and a `result.json`. The root also receives `results.jsonl`. Each result stores a signature over the ligand/receptor/reference file identities and all effective options. `resume=True` reuses a result only when that signature still matches; changing an option or replacing an input file reruns the job.

Molecule-level execution is sequential. Pose scoring and optimization inside each job remain batched. Full-receptor and residue-extracted covalent contexts are reused through atom-typing-aware caches.
