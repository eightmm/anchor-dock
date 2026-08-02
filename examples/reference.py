from pathlib import Path

from anchor_dock import dock_reference

ROOT = Path(__file__).resolve().parent
result = dock_reference(
    ROOT / "10gs" / "10gs_pocket.pdb",
    ROOT / "10gs" / "10gs_ligand.sdf",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    ROOT / "output" / "reference",
    num_confs=128,
    optimize=True,
)
print(result)
