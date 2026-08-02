import argparse

from anchor_dock import dock_free

parser = argparse.ArgumentParser()
parser.add_argument("protein")
parser.add_argument("ligand")
args = parser.parse_args()

print(dock_free(args.protein, args.ligand, "free_output", num_starts=128))
