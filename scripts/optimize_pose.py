import argparse
import copy

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from lig_align.aligner import LigandAligner
from lig_align.io import load_pocket_bundle


def optimize_single_pose(protein_pdb: str, ligand_sdf: str, out_sdf: str, num_steps: int = 100, lr: float = 0.05, torsion_penalty: bool = True, weight_preset: str = 'vina', optimizer: str = 'adam'):
    """
    Loads an existing 3D ligand pose (e.g., the native reference) and optimizes its torsions 
    in place against the protein pocket without running conformer generation or alignment.
    """
    RDLogger.DisableLog("rdApp.warning")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    aligner = LigandAligner(device=device)

    print(f"Loading protein pocket from {protein_pdb}...")
    pocket_bundle = load_pocket_bundle(protein_pdb, device, aligner.compute_vina_features)
    
    print(f"Loading 3D ligand from {ligand_sdf}...")
    suppl = Chem.SDMolSupplier(ligand_sdf)
    ligand_mol = suppl[0]
    ligand_mol = Chem.AddHs(ligand_mol, addCoords=True)
    
    init_coords = torch.tensor(ligand_mol.GetConformer().GetPositions(), dtype=torch.float32, device=device)
    pocket_coords = pocket_bundle.coords
    
    query_features = aligner.compute_vina_features(ligand_mol)
    pocket_features = pocket_bundle.features
    
    num_rotatable_bonds = None
    if torsion_penalty:
        num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(ligand_mol)
    
    # Initial Score
    init_score = aligner.step4_vina_scoring(
        init_coords.unsqueeze(0), pocket_coords, query_features, pocket_features, num_rotatable_bonds, weight_preset
    ).item()
    print(f"\nInitial Vina Score ({weight_preset}): {init_score:.3f} kcal/mol")
    
    # We want to optimize the entire molecule. We pick atom 0 to be the rigid "root" anchor
    # so the molecule doesn't fly away, but all internal rotatable bonds are free to twist.
    ref_indices = [0]
    
    print(f"Running Torsion Optimization for {num_steps} steps using {optimizer.upper()}...")
    opt_coords = aligner.step6_refine_pose(
        mol=ligand_mol,
        ref_indices=ref_indices,
        init_coords=init_coords,
        pocket_coords=pocket_coords,
        query_features=query_features,
        pocket_features=pocket_features,
        num_steps=num_steps,
        lr=lr,
        freeze_mcs=False,
        num_rotatable_bonds=num_rotatable_bonds,
        weight_preset=weight_preset,
        optimizer=optimizer
    )
    
    final_score = aligner.step4_vina_scoring(
        opt_coords.unsqueeze(0), pocket_coords, query_features, pocket_features, num_rotatable_bonds, weight_preset
    ).item()
    print(f"Final Vina Score ({weight_preset}):   {final_score:.3f} kcal/mol (Delta: {final_score - init_score:.3f})")
    
    # Save 
    out_mol = copy.deepcopy(ligand_mol)
    conf = out_mol.GetConformer()
    opt_numpy = opt_coords.cpu().numpy()
    for i in range(out_mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (float(opt_numpy[i,0]), float(opt_numpy[i,1]), float(opt_numpy[i,2])))
        
    out_mol.SetProp("Vina_Score_Init", f"{init_score:.3f}")
    out_mol.SetProp("Vina_Score_Opt", f"{final_score:.3f}")
    
    writer = Chem.SDWriter(out_sdf)
    writer.write(out_mol)
    writer.close()
    print(f"\nOptimized structure saved to {out_sdf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize an existing 3D ligand pose directly in the pocket")
    parser.add_argument("-p", "--protein", required=True, help="Path to protein PDB")
    parser.add_argument("-l", "--ligand", required=True, help="Path to 3D ligand SDF to optimize")
    parser.add_argument("-o", "--out_sdf", required=True, help="Output SDF path")
    parser.add_argument("--steps", type=int, default=200, help="Number of gradient steps")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw", "lbfgs"], default="adam", help="Optimizer: adam, adamw, or lbfgs (default: adam)")
    parser.set_defaults(torsion_penalty=True)
    torsion_group = parser.add_mutually_exclusive_group()
    torsion_group.add_argument("--torsion_penalty", dest="torsion_penalty", action="store_true",
                               help="Include the standard AutoDock Vina torsional entropy penalty (default)")
    torsion_group.add_argument("--no_torsion_penalty", dest="torsion_penalty", action="store_false",
                               help="Disable the torsional entropy penalty")
    parser.add_argument("--weight_preset", type=str, choices=["vina", "vina_lp", "vinardo"], default="vina", help="Preset dictionary for Vina functional weights")
    args = parser.parse_args()

    optimize_single_pose(args.protein, args.ligand, args.out_sdf, args.steps, args.lr, args.torsion_penalty, args.weight_preset, args.optimizer)
