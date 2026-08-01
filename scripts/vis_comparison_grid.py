"""
Create 2x2 comparison grid visualizations for different optimization methods.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors

from anchor_dock.core.features import compute_vina_features
from anchor_dock.core.kinematics import LigandKinematics
from anchor_dock.core.masks import compute_intramolecular_mask
from anchor_dock.core.scoring import vina_scoring
from anchor_dock.reference import LigandAligner, load_pocket_bundle, process_query_ligand

RDLogger.DisableLog('rdApp.warning')


def draw_molecule_3d(ax, mol, coords, color, alpha, label, highlight_indices=None):
    """Draw molecule in 3D with bonds."""
    coords_np = coords.cpu().numpy() if isinstance(coords, torch.Tensor) else coords

    # Scatter atoms
    xs, ys, zs = coords_np[:, 0], coords_np[:, 1], coords_np[:, 2]

    if highlight_indices:
        # Draw non-highlighted atoms
        mask = np.ones(len(coords_np), dtype=bool)
        mask[highlight_indices] = False
        ax.scatter(xs[mask], ys[mask], zs[mask], c=color, s=50, alpha=alpha*0.5, label=label)
        # Draw highlighted atoms
        ax.scatter(xs[highlight_indices], ys[highlight_indices], zs[highlight_indices],
                  c='#e67e22', s=100, alpha=1.0, marker='o', edgecolors='black', linewidths=2)
    else:
        ax.scatter(xs, ys, zs, c=color, s=50, alpha=alpha, label=label)

    # Draw bonds
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]],
               c=color, alpha=alpha*0.5, linewidth=1.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='upper right')


def optimize_pose(mol, ref_indices, init_coords, pocket_coords, query_features, pocket_features,
                 device, num_steps=100, lr=0.05, freeze_mcs=True,
                 torsion_penalty=True, weight_preset='vina'):
    """Run optimization and return trajectory."""

    # Check if molecule has rotatable bonds
    test_model = LigandKinematics(mol, ref_indices, init_coords, device, freeze_mcs=freeze_mcs)
    if test_model.num_torsions == 0:
        print("  No rotatable bonds - skipping optimization")
        return init_coords.unsqueeze(0), torch.tensor([0.0])

    # Setup model and optimizer
    model = LigandKinematics(mol, ref_indices, init_coords, device, freeze_mcs=freeze_mcs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Precompute mask
    intra_mask = compute_intramolecular_mask(mol, device)

    # Calculate number of rotatable bonds
    num_rot_bonds = None
    if torsion_penalty:
        num_rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

    # Track trajectory (save every 10 steps)
    trajectory = []
    scores = []

    for step in range(num_steps):
        optimizer.zero_grad()
        coords = model()

        loss = vina_scoring(coords.unsqueeze(0), pocket_coords, query_features, pocket_features,
                           num_rot_bonds, weight_preset, intramolecular_mask=intra_mask)

        if step % 10 == 0:
            trajectory.append(coords.detach().clone())
            scores.append(loss.item())

        loss.sum().backward()
        optimizer.step()

    # Final result
    with torch.no_grad():
        final_coords = model()
        final_score = vina_scoring(final_coords.unsqueeze(0), pocket_coords, query_features,
                                   pocket_features, num_rot_bonds, weight_preset,
                                   intramolecular_mask=intra_mask)
        trajectory.append(final_coords.detach().clone())
        scores.append(final_score.item())

    return torch.stack(trajectory), torch.tensor(scores)


def create_2x2_comparison(query_smiles, output_path, protein_pdb=None, ref_sdf=None,
                         num_steps=100, title_prefix=""):
    """
    Create 2x2 grid comparing:
    - Top-left: Fixed MCS + Vina
    - Top-right: Free MCS + Vina
    - Bottom-left: Fixed MCS + Vinardo
    - Bottom-right: Free MCS + Torsion Penalty
    """

    # Default paths
    if protein_pdb is None:
        protein_pdb = "examples/10gs/10gs_pocket.pdb"
    if ref_sdf is None:
        ref_sdf = "examples/10gs/10gs_ligand.sdf"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    aligner = LigandAligner(device=device)

    print("Loading data...")
    print(f"  Protein: {protein_pdb}")
    print(f"  Reference: {ref_sdf}")

    # Load reference and pocket
    ref_mol = Chem.SDMolSupplier(ref_sdf)[0]
    pocket_bundle = load_pocket_bundle(protein_pdb, device, aligner.compute_vina_features)
    pocket_mol = pocket_bundle.mol

    if ref_mol is None or pocket_mol is None:
        raise ValueError("Failed to load molecules")

    # Process query
    query_mol, canonical_smiles = process_query_ligand(query_smiles)
    print(f"  Query: {canonical_smiles}")

    # Find MCS
    mapping = aligner.step2_find_mcs(ref_mol, query_mol)
    query_indices = [m[1] for m in mapping]
    print(f"  MCS size: {len(mapping)} atoms")

    # Generate single conformer with MCS constraints
    coordMap = {}
    ref_conf = ref_mol.GetConformer()
    for ref_idx, query_idx in mapping:
        pos = ref_conf.GetAtomPosition(ref_idx)
        coordMap[query_idx] = pos

    AllChem.EmbedMolecule(query_mol, coordMap=coordMap, randomSeed=42)

    # Get initial coordinates
    init_coords = torch.tensor(query_mol.GetConformer().GetPositions(),
                              dtype=torch.float32, device=device)

    # Get pocket coordinates
    pocket_coords = pocket_bundle.coords

    # Compute features
    query_features = compute_vina_features(query_mol, device)
    pocket_features = pocket_bundle.features

    print("\nRunning 4 optimization methods...")

    # Method 1: Fixed MCS + Vina
    print("  1/4 Fixed MCS + Vina...")
    traj1, scores1 = optimize_pose(query_mol, query_indices, init_coords, pocket_coords,
                                   query_features, pocket_features, device,
                                   num_steps=num_steps, freeze_mcs=True,
                                   weight_preset='vina')

    # Method 2: Free MCS + Vina
    print("  2/4 Free MCS + Vina...")
    traj2, scores2 = optimize_pose(query_mol, query_indices, init_coords, pocket_coords,
                                   query_features, pocket_features, device,
                                   num_steps=num_steps, freeze_mcs=False,
                                   weight_preset='vina')

    # Method 3: Fixed MCS + Vinardo
    print("  3/4 Fixed MCS + Vinardo...")
    traj3, scores3 = optimize_pose(query_mol, query_indices, init_coords, pocket_coords,
                                   query_features, pocket_features, device,
                                   num_steps=num_steps, freeze_mcs=True,
                                   weight_preset='vinardo')

    # Method 4: Free MCS + No Torsion Penalty
    print("  4/4 Free MCS + No Torsion Penalty...")
    traj4, scores4 = optimize_pose(query_mol, query_indices, init_coords, pocket_coords,
                                   query_features, pocket_features, device,
                                   num_steps=num_steps, freeze_mcs=False,
                                   torsion_penalty=False, weight_preset='vina')

    # Create 2x2 plot with 6 columns per method
    print("\nCreating comprehensive visualization...")
    fig = plt.figure(figsize=(24, 16))

    methods = [
        ("Fixed MCS + Vina", traj1, scores1, True),
        ("Free MCS + Vina", traj2, scores2, False),
        ("Fixed MCS + Vinardo", traj3, scores3, True),
        ("Free MCS + No Torsion Penalty", traj4, scores4, False),
    ]

    for idx, (method_name, traj, scores, freeze_mcs) in enumerate(methods, 1):
        row_base = idx - 1
        color = '#3498db' if freeze_mcs else '#e74c3c'
        steps = np.arange(len(scores)) * 10

        # Column 1: 3D Initial vs Final
        ax1 = fig.add_subplot(4, 6, row_base * 6 + 1, projection='3d')
        draw_molecule_3d(ax1, query_mol, traj[0], '#95a5a6', 0.3, 'Initial',
                        highlight_indices=query_indices if freeze_mcs else None)
        draw_molecule_3d(ax1, query_mol, traj[-1], color, 0.9, 'Optimized',
                        highlight_indices=query_indices if freeze_mcs else None)
        ax1.set_title(f"{method_name}\n3D Structure", fontsize=10, fontweight='bold')

        # Column 2: Energy Trajectory with improvement fill
        ax2 = fig.add_subplot(4, 6, row_base * 6 + 2)
        ax2.plot(steps, scores.numpy(), color=color, linewidth=2, label='Energy')
        ax2.fill_between(steps, scores[0].item(), scores.numpy(),
                        where=(scores.numpy() < scores[0].item()),
                        color='green', alpha=0.2, label='Improvement')
        ax2.axhline(y=scores[0].item(), color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Step', fontsize=9)
        ax2.set_ylabel('Vina Score (kcal/mol)', fontsize=9)
        ax2.set_title(f'Energy Trajectory\nΔ = {scores[-1].item() - scores[0].item():.3f}',
                     fontsize=10, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        # Column 3: RMSD (Overall + MCS separately)
        ax3 = fig.add_subplot(4, 6, row_base * 6 + 3)
        rmsds = []
        rmsds_mcs = []
        for frame in traj:
            # Overall RMSD
            rmsd = torch.sqrt(((frame - traj[0])**2).sum(dim=1).mean()).item()
            rmsds.append(rmsd)
            # MCS RMSD
            if len(query_indices) > 0:
                rmsd_mcs = torch.sqrt(((frame[query_indices] - traj[0][query_indices])**2).sum(dim=1).mean()).item()
                rmsds_mcs.append(rmsd_mcs)

        ax3.plot(steps, rmsds, color=color, linewidth=2, label='All Atoms')
        if rmsds_mcs:
            ax3.plot(steps, rmsds_mcs, color='#e67e22', linewidth=2, linestyle='--',
                    label=f'MCS Only ({len(query_indices)} atoms)')
        ax3.set_xlabel('Step', fontsize=9)
        ax3.set_ylabel('RMSD (Å)', fontsize=9)
        ax3.set_title(f'Structural Deviation\nFinal: {rmsds[-1]:.2f} Å',
                     fontsize=10, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)

        # Column 4: Max Atom Displacement
        ax4 = fig.add_subplot(4, 6, row_base * 6 + 4)
        max_atom_moves = []
        for frame in traj:
            max_move = torch.sqrt(((frame - traj[0])**2).sum(dim=1)).max().item()
            max_atom_moves.append(max_move)
        ax4.plot(steps, max_atom_moves, color='purple', linewidth=2)
        ax4.fill_between(steps, 0, max_atom_moves, color='purple', alpha=0.2)
        ax4.set_xlabel('Step', fontsize=9)
        ax4.set_ylabel('Max Displacement (Å)', fontsize=9)
        ax4.set_title(f'Max Atom Movement\nPeak: {max(max_atom_moves):.2f} Å',
                     fontsize=10, fontweight='bold')
        ax4.grid(alpha=0.3)

        # Column 5: Energy vs RMSD Scatter (colored by step)
        ax5 = fig.add_subplot(4, 6, row_base * 6 + 5)
        scatter = ax5.scatter(rmsds, scores.numpy(), c=steps, cmap='viridis',
                            s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        # Add trajectory line
        ax5.plot(rmsds, scores.numpy(), color='gray', alpha=0.3, linewidth=1, zorder=0)
        # Highlight start and end
        ax5.scatter([rmsds[0]], [scores[0].item()], c='red', s=150, marker='o',
                   edgecolors='black', linewidth=2, label='Start', zorder=10)
        ax5.scatter([rmsds[-1]], [scores[-1].item()], c='green', s=150, marker='*',
                   edgecolors='black', linewidth=2, label='End', zorder=10)
        ax5.set_xlabel('RMSD (Å)', fontsize=9)
        ax5.set_ylabel('Vina Score (kcal/mol)', fontsize=9)
        ax5.set_title('Energy Landscape', fontsize=10, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('Step', fontsize=8)
        ax5.legend(fontsize=8)
        ax5.grid(alpha=0.3)

        # Column 6: Summary Statistics Box
        ax6 = fig.add_subplot(4, 6, row_base * 6 + 6)
        ax6.axis('off')

        score_improvement = scores[-1].item() - scores[0].item()
        final_rmsd = rmsds[-1]
        max_rmsd = max(rmsds)
        final_rmsd_mcs = rmsds_mcs[-1] if rmsds_mcs else 0.0

        summary_text = f"""
ENERGY
Initial:  {scores[0].item():>7.3f} kcal/mol
Final:    {scores[-1].item():>7.3f} kcal/mol
ΔE:       {score_improvement:>7.3f} kcal/mol
Change:   {score_improvement/scores[0].item()*100:>6.1f}%

STRUCTURE
Final RMSD:     {final_rmsd:>5.2f} Å
Max RMSD:       {max_rmsd:>5.2f} Å
MCS RMSD:       {final_rmsd_mcs:>5.2f} Å
Max atom move:  {max(max_atom_moves):>5.2f} Å

OPTIMIZATION
Steps:      {num_steps:>5d}
Freeze MCS: {'Yes' if freeze_mcs else 'No':>5s}
Atoms:      {len(traj[0]):>5d}
MCS atoms:  {len(query_indices):>5d}
"""

        # Color-coded improvement indicator
        status_color = '#27ae60' if score_improvement < 0 else '#e74c3c'
        status_text = '✓ IMPROVED' if score_improvement < 0 else '✗ WORSENED'

        ax6.text(0.5, 0.95, status_text, fontsize=12, fontweight='bold',
                ha='center', va='top', color=status_color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=status_color, alpha=0.2))

        ax6.text(0.5, 0.80, summary_text, fontsize=9, ha='center', va='top',
                family='monospace', bbox=dict(boxstyle='round,pad=0.8',
                facecolor='lightgray', alpha=0.3))

    # Overall title
    fig.suptitle(f'{title_prefix}Optimization Method Comparison\nQuery: {canonical_smiles}',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")

    return {
        'methods': [m[0] for m in methods],
        'improvements': [scores[-1].item() - scores[0].item() for _, _, scores, _ in methods],
        'final_rmsds': [torch.sqrt(((traj[-1] - traj[0])**2).sum(dim=1).mean()).item()
                       for _, traj, _, _ in methods]
    }


def main():
    parser = argparse.ArgumentParser(description="Create 2x2 optimization comparison grid")
    parser.add_argument("-q", "--query", required=True, help="Query SMILES")
    parser.add_argument("-o", "--output", required=True, help="Output PNG path")
    parser.add_argument("-p", "--protein", default="examples/10gs/10gs_pocket.pdb",
                       help="Protein PDB file")
    parser.add_argument("-r", "--reference", default="examples/10gs/10gs_ligand.sdf",
                       help="Reference ligand SDF")
    parser.add_argument("--steps", type=int, default=100, help="Optimization steps")
    parser.add_argument("--title", default="", help="Title prefix")

    args = parser.parse_args()

    results = create_2x2_comparison(
        args.query,
        args.output,
        protein_pdb=args.protein,
        ref_sdf=args.reference,
        num_steps=args.steps,
        title_prefix=args.title
    )

    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    for method, improvement, rmsd in zip(results['methods'],
                                         results['improvements'],
                                         results['final_rmsds']):
        print(f"{method:30s} | Δ Score: {improvement:+.3f} | RMSD: {rmsd:.2f} Å")
    print("="*70)


if __name__ == "__main__":
    main()
