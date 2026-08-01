import argparse

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from rdkit import Chem, RDLogger

from lig_align.aligner import LigandAligner
from lig_align.alignment import LigandKinematics
from lig_align.io import load_pocket_bundle, process_query_ligand
from lig_align.io.visualization import draw_molecule_3d, get_2d_image
from lig_align.molecular.relax import relax_pose_with_fixed_core
from lig_align.scoring import compute_intramolecular_mask, vina_scoring


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", required=True, help="SMILES to optimize")
    parser.add_argument("-o", "--output", required=True, help="Output GIF path")
    parser.add_argument("-p", "--protein", required=True, help="Protein PDB path")
    parser.add_argument("-r", "--ref", required=True, help="Reference SDF path")
    parser.add_argument("-t", "--title", required=False, default="Optimization", help="Plot Title")
    parser.add_argument("--steps", type=int, default=100, help="Number of optimization steps")
    parser.add_argument("--free_mcs", action="store_true", help="Allow MCS atoms to optimize as well")
    parser.set_defaults(torsion_penalty=True)
    torsion_group = parser.add_mutually_exclusive_group()
    torsion_group.add_argument("--torsion_penalty", dest="torsion_penalty", action="store_true",
                               help="Include the standard AutoDock Vina torsional entropy penalty (default)")
    torsion_group.add_argument("--no_torsion_penalty", dest="torsion_penalty", action="store_false",
                               help="Disable the torsional entropy penalty")
    parser.add_argument("--weight_preset", type=str, choices=["vina", "vina_lp", "vinardo"], default="vina", help="Preset dictionary for Vina functional weights")
    args = parser.parse_args()

    RDLogger.DisableLog('rdApp.warning')
    
    protein_pdb = args.protein
    ref_sdf = args.ref
    query_smiles = args.query
    out_gif = args.output
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    aligner = LigandAligner(device=device)

    # 1. Loading
    ref_mol = Chem.SDMolSupplier(ref_sdf)[0]
    pocket_bundle = load_pocket_bundle(protein_pdb, device, aligner.compute_vina_features)


    query_mol, _ = process_query_ligand(query_smiles)
    
    # 2. MCS & Coords mapping
    mapping = aligner.step2_find_mcs(ref_mol, query_mol)
    ref_conf = ref_mol.GetConformer()
    coordMap = {}
    for ref_idx, query_idx in mapping:
        pos = ref_conf.GetAtomPosition(ref_idx)
        coordMap[query_idx] = pos
        
    query_indices = [m[1] for m in mapping]
    ref_indices = [m[0] for m in mapping]
    num_mcs_atoms = len(mapping)
    overlap = num_mcs_atoms / ref_mol.GetNumHeavyAtoms() * 100

    # 3. Generate initial candidate
    query_mol, rep_cids = aligner.step1_generate_conformers(query_mol, num_confs=50, coordMap=coordMap)
    
    if len(rep_cids) == 0:
        print("Failed to generate conformers.")
        return

    # 4. Enforce MCS via MMFF94
    cid = rep_cids[0]
    conf = query_mol.GetConformer(cid)
    for ref_idx, query_idx in mapping:
        pos = ref_conf.GetAtomPosition(ref_idx)
        conf.SetAtomPosition(query_idx, pos)
        
    applied, message = relax_pose_with_fixed_core(query_mol, cid, set(query_indices), max_iters=500)
    print(f"Relaxation status: {message}")
    
    init_coords = torch.tensor(conf.GetPositions(), dtype=torch.float32, device=device)
    
    # 5. Extract Features
    pocket_coords = pocket_bundle.coords
    query_feat = aligner.compute_vina_features(query_mol)
    pocket_feat = pocket_bundle.features
    ref_coords_numpy = torch.tensor(ref_mol.GetConformer().GetPositions()).cpu().numpy()
    
    # 6. Setup Kinematics and Optimizer
    freeze_mcs_flag = not args.free_mcs
    model = LigandKinematics(query_mol, query_indices, init_coords, device, freeze_mcs=freeze_mcs_flag)
    
    num_rotatable_bonds = None
    if args.torsion_penalty:
        from rdkit.Chem import rdMolDescriptors
        num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(query_mol)
    
    if model.num_torsions == 0:
        print(f"Skipping {args.title} (overlap {overlap:.1f}%): No rotatable bonds found.")
        return
        
    intra_mask = compute_intramolecular_mask(query_mol, device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    losses = []
    coords_history = []
    
    print(f"Tracking Torsion Optimization for {args.title} (overlap: {overlap:.1f}%, torsions: {model.num_torsions})...")
    for step in range(args.steps):
        optimizer.zero_grad()
        coords = model()
        loss = vina_scoring(coords.unsqueeze(0), pocket_coords, query_feat, pocket_feat, num_rotatable_bonds, args.weight_preset, intramolecular_mask=intra_mask)
        loss = loss.sum()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        coords_history.append(coords.detach().cpu().numpy())
        
    init_numpy = init_coords.cpu().numpy()
    
    # 7. Animation Setup (2x2 Grid)
    print("Generating 2x2 GIF animation...")
    fig = plt.figure(figsize=(16, 12))
    
    ax_topleft = fig.add_subplot(2, 2, 1)
    ax_topright = fig.add_subplot(2, 2, 2)
    ax_botleft = fig.add_subplot(2, 2, 3)
    ax_botright = fig.add_subplot(2, 2, 4, projection='3d')
    
    # Render 2D images
    ref_img = get_2d_image(ref_mol, highlight_atoms=ref_indices)
    query_img = get_2d_image(query_mol, highlight_atoms=query_indices, align_ref=ref_mol, match_pairs=mapping)
    
    ax_topleft.imshow(ref_img)
    ax_topleft.axis('off')
    ax_topleft.set_title("Reference Molecule\nGold Highlight: MCS Anchor", fontsize=14, fontweight='bold')
    
    ax_topright.imshow(query_img)
    ax_topright.axis('off')
    ax_topright.set_title(f"Query Variant: {args.title}\nGold Highlight: Aligned MCS Anchor", fontsize=14, fontweight='bold')
    
    pad = 2.0
    min_x, max_x = ref_coords_numpy[:, 0].min() - pad, ref_coords_numpy[:, 0].max() + pad
    min_y, max_y = ref_coords_numpy[:, 1].min() - pad, ref_coords_numpy[:, 1].max() + pad
    min_z, max_z = ref_coords_numpy[:, 2].min() - pad, ref_coords_numpy[:, 2].max() + pad
    
    def update(frame):
        ax_botleft.clear()
        ax_botright.clear()
        
        # Energy Landscape
        ax_botleft.plot(losses[:frame+1], color='#e74c3c', linewidth=3)
        ax_botleft.set_xlim(0, args.steps)
        padding = 5
        ax_botleft.set_ylim(min(losses) - padding, max(losses) + padding)
        ax_botleft.set_title(f"Energy Minimization (Step {frame+1}/{args.steps})", fontsize=14, fontweight='bold')
        ax_botleft.set_xlabel("Adam Step", fontsize=12)
        ax_botleft.set_ylabel("Vina Energy Score (kcal/mol)", fontsize=12)
        ax_botleft.grid(True, linestyle='--', alpha=0.7)
        ax_botleft.scatter([frame], [losses[frame]], color='black', s=100, zorder=5)
        ax_botleft.text(frame + 2, losses[frame], f"{losses[frame]:.2f}", fontsize=12, fontweight='bold')
        
        # 3D Structure Overlay
        ax_botright.set_title(f"3D Opt Trajectory\n({model.num_torsions} Flexible Torsions)", fontsize=14, fontweight='bold')
        ax_botright.set_xlim(min_x, max_x)
        ax_botright.set_ylim(min_y, max_y)
        ax_botright.set_zlim(min_z, max_z)
        
        draw_molecule_3d(ax_botright, ref_mol, ref_coords_numpy, color='#2ecc71', alpha=0.3, label="Reference Native")
        draw_molecule_3d(ax_botright, query_mol, init_numpy, color='gray', alpha=0.15, label="Initial Pose", highlight_indices=query_indices)
        draw_molecule_3d(ax_botright, query_mol, coords_history[frame], color='#1abc9c', alpha=1.0, label="Current", highlight_indices=query_indices)
        
        ax_botright.legend()
        ax_botright.set_xticks([])
        ax_botright.set_yticks([])
        ax_botright.set_zticks([])
        
        # Return all updated axes
        return ax_botleft, ax_botright

    ani = FuncAnimation(fig, update, frames=len(coords_history), interval=100, blit=False)
    ani.save(out_gif, writer=PillowWriter(fps=10))
    print(f"GIF saved to {out_gif}")

if __name__ == "__main__":
    main()
