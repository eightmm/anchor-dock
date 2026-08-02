from anchor_dock import DockingJob, dock_batch

jobs = [
    DockingJob.reference("CCO", protein_pdb="pocket.pdb", reference_ligand="known.sdf", id="reference-1"),
    DockingJob.covalent("C=CC(=O)NCC", protein_pdb="protein.pdb", reactive_residue="CYS145:A", id="covalent-1"),
    DockingJob.free("CCN", protein_pdb="pocket.pdb", id="free-1"),
]
print(dock_batch(jobs, output_dir="batch_output", resume=True))
