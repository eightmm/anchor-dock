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

Each ligand/reference must contain one connected component. Salts and mixtures are rejected instead of being silently desalted.

When a directory is scanned recursively, the resolved `output_dir` subtree is
excluded so that results from the current or an earlier run are not ingested as
new inputs. An explicit supported input file inside `output_dir` is accepted
only when it is outside the reserved root manifests (`results.jsonl` and
`summary.json`) and every job output directory. Using the output directory
itself as a directory source, or placing any discoverable batch, ligand,
receptor, or reference input inside a directory that the run will write, fails
before output is created or replaced.

Finite iterators, generic iterables, DataFrame-like inputs, and file-like inputs
are frozen or buffered for this collision preflight before execution begins.
For very large batches, prefer an SDF, SMI, CSV, JSONL, or compressed input file
instead of constructing a large in-memory iterator.

JSON objects may contain either `jobs` or `ligands`, but not both. An explicitly
empty container such as `{"jobs": []}` is a successful zero-item batch rather
than an invalid manifest.

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

Every job receives a sanitized deterministic directory name and a `result.json`. The root also receives `results.jsonl` and `summary.json`. Manifests are written atomically. A new run first publishes `status: running` and invalidates the previous root result log; only normal completion publishes `status: complete`. Each signature covers package/schema, ligand/receptor/reference content hashes, metadata, all effective options, custom scorer code/state/config/name/units, the resolved device, and NumPy/RDKit/Torch versions. The same runtime identity is recorded as `batch_runtime_identity` in success and failure records. A successful record also stores output artifact hash and size. `resume=True` reuses only a matching success whose artifact is inside the job directory and still has exactly that content. Failed, missing, stale, corrupt, replaced, external, or runtime-mismatched artifacts rerun.

Malformed rows and molecule records are reported as batch failures rather than silently dropped. `on_error="record"` continues, `"skip"` continues without returning the failed item, and `"raise"` stops at the first error.

Molecule-level execution is sequential. Pose scoring and optimization inside each job remain batched. Full-receptor and residue-extracted covalent contexts are reused through atom-typing-aware caches.
