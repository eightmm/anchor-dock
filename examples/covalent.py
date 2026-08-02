import argparse

from anchor_dock import dock_covalent

parser = argparse.ArgumentParser()
parser.add_argument("protein")
parser.add_argument("residue")
parser.add_argument("ligand")
args = parser.parse_args()

print(dock_covalent(args.protein, args.ligand, args.residue, "covalent_output"))
