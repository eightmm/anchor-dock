import argparse

from anchor_dock import dock_interaction

parser = argparse.ArgumentParser()
parser.add_argument("protein", help="PDB receptor containing the explicitly selected residues")
parser.add_argument("ligand", help="SMILES, InChI, or a supported single-molecule file")
args = parser.parse_args()

print(
    dock_interaction(
        args.protein,
        args.ligand,
        "interaction_multi_output",
        interactions=[
            {
                "receptor_residue": "ASP189:A",
                "receptor_atom": "OD1",
                "ligand_smarts": "[N:1]",
                "target_distance": 3.0,
                "distance_tolerance": 0.5,
            },
            {
                "receptor_residue": "SER190:A",
                "receptor_atom": "OG",
                "ligand_smarts": "[O:1]",
                "target_distance": 2.9,
                "distance_tolerance": 0.4,
                "restraint_weight": 12.0,
            },
        ],
        max_joint_matches=64,
    )
)
