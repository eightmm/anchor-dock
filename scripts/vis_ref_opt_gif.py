import argparse

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from lig_align.aligner import LigandAligner
from lig_align.alignment import LigandKinematics
from lig_align.io import load_pocket_bundle
from lig_align.io.visualization import draw_molecule_3d
from lig_align.scoring import compute_intramolecular_mask, vina_scoring


def main():
    parser = argparse.ArgumentParser(description="Create a GIF of a single SDF pose being optimized")
    parser.add_argument("-p", "--protein", required=True, help="Protein PDB file")
    parser.add_argument("-l", "--ligand", required=True, help="Ligand SDF file (with 3D coords)")
    parser.add_argument("-o", "--output", required=True, help="Output GIF path")
    parser.add_argument("-t", "--title", required=False, default="Reference Relaxation", help="Plot Title")
    parser.add_argument("--steps", type=int, default=100, help="Number of optimization steps")
    parser.set_defaults(torsion_penalty=True)
    torsion_group = parser.add_mutually_exclusive_group()
    torsion_group.add_argument("--torsion_penalty", dest="torsion_penalty", action="store_true",
                               help="Include the standard AutoDock Vina torsional entropy penalty (default)")
    torsion_group.add_argument("--no_torsion_penalty", dest="torsion_penalty", action="store_false",
                               help="Disable the torsional entropy penalty")
    parser.add_argument("--weight_preset", type=str, choices=["vina", "vina_lp", "vinardo"], default="vina", help="Preset dictionary for Vina functional weights")
    args = parser.parse_args()

    RDLogger.DisableLog('rdApp.warning')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    aligner = LigandAligner(device=device)

    # 1. Loading
    pocket_bundle = load_pocket_bundle(args.protein, device, aligner.compute_vina_features)
    ligand_mol = Chem.SDMolSupplier(args.ligand)[0]
    ligand_mol = Chem.AddHs(ligand_mol, addCoords=True)
    
    init_coords = torch.tensor(ligand_mol.GetConformer().GetPositions(), dtype=torch.float32, device=device)
    pocket_coords = pocket_bundle.coords
    
    # 2. Extract Features
    query_feat = aligner.compute_vina_features(ligand_mol)
    pocket_feat = pocket_bundle.features
    
    num_rotatable_bonds = None
    if args.torsion_penalty:
        num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(ligand_mol)
        
    # 3. Setup Kinematics and Optimizer
    root_idx = 0 # Arbitrary root anchor to prevent rigid-body drift
    model = LigandKinematics(ligand_mol, [root_idx], init_coords, device, freeze_mcs=False)
    
    if model.num_torsions == 0:
        print(f"Skipping {args.title}: No rotatable bonds found.")
        return
        
    intra_mask = compute_intramolecular_mask(ligand_mol, device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    losses = []
    coords_history = []
    
    print(f"Tracking Torsion Optimization for {args.title} (torsions: {model.num_torsions})...")
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
    
    # 4. Animation Setup
    print("Generating GIF animation...")
    fig = plt.figure(figsize=(14, 6))
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Bounding box
    pad = 2.0
    min_x, max_x = init_numpy[:, 0].min() - pad, init_numpy[:, 0].max() + pad
    min_y, max_y = init_numpy[:, 1].min() - pad, init_numpy[:, 1].max() + pad
    min_z, max_z = init_numpy[:, 2].min() - pad, init_numpy[:, 2].max() + pad
    
    def update(frame):
        ax1.clear()
        ax2.clear()
        
        # Energy Landscape
        ax1.plot(losses[:frame+1], color='#e74c3c', linewidth=3)
        ax1.set_xlim(0, args.steps)
        padding = 5
        ax1.set_ylim(min(losses) - padding, max(losses) + padding)
        ax1.set_title(f"Energy Minimization (Step {frame+1}/{args.steps})\n{args.title}", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Adam Step", fontsize=12)
        ax1.set_ylabel("Vina Energy Score (kcal/mol)", fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.scatter([frame], [losses[frame]], color='black', s=100, zorder=5)
        ax1.text(frame + 2, losses[frame], f"{losses[frame]:.2f}", fontsize=12, fontweight='bold')
        
        # 3D Structure Overlay
        ax2.set_title(f"3D Opt Trajectory\n({model.num_torsions} Flexible Torsions)", fontsize=14, fontweight='bold')
        ax2.set_xlim(min_x, max_x)
        ax2.set_ylim(min_y, max_y)
        ax2.set_zlim(min_z, max_z)
        
        draw_molecule_3d(ax2, ligand_mol, init_numpy, color='gray', alpha=0.2, label="Crystal Pose", highlight_indices=[root_idx])
        draw_molecule_3d(ax2, ligand_mol, coords_history[frame], color='#3498db', alpha=1.0, label="Optimizing", highlight_indices=[root_idx])
        
        ax2.legend()
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])
        
        return ax1, ax2

    ani = FuncAnimation(fig, update, frames=len(coords_history), interval=100, blit=False)
    ani.save(args.output, writer=PillowWriter(fps=10))
    print(f"GIF saved to {args.output}")

if __name__ == "__main__":
    main()
