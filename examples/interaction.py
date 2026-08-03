import argparse

from anchor_dock import dock_interaction

parser = argparse.ArgumentParser()
parser.add_argument("protein")
parser.add_argument("residue", help="exact residue selector, for example ASP189:A")
parser.add_argument("receptor_atom", help="exact non-hydrogen PDB atom name")
parser.add_argument("ligand")
parser.add_argument("ligand_smarts", help="SMARTS with exactly one mapped :1 atom")
parser.add_argument("target_distance", type=float)
parser.add_argument("distance_tolerance", type=float)
args = parser.parse_args()

print(
    dock_interaction(
        args.protein,
        args.ligand,
        "interaction_output",
        receptor_residue=args.residue,
        receptor_atom=args.receptor_atom,
        ligand_smarts=args.ligand_smarts,
        target_distance=args.target_distance,
        distance_tolerance=args.distance_tolerance,
    )
)
