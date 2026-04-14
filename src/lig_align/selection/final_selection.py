import torch
from rdkit import Chem
from rdkit.Geometry import Point3D
from typing import List

def final_selection(mol: Chem.Mol,
                    representative_cids: List[int],
                    aligned_coords: torch.Tensor,
                    scores: torch.Tensor,
                    initial_scores: torch.Tensor = None,
                    position_labels: List[int] = None,
                    output_path: str = "output.sdf") -> torch.Tensor:
    """
    Sort all poses by energy (ascending) and save to SDF.

    Args:
        mol: RDKit molecule
        representative_cids: List of conformer IDs
        aligned_coords: Aligned coordinates [N_poses, N_atoms, 3]
        scores: Vina scores [N_poses]
        initial_scores: Optional pre-optimization scores aligned to the same pose order
        position_labels: Optional MCS position index per pose (1-based)
        output_path: Output SDF file path

    Returns:
        Sorted indices (best energy first)
    """
    selected_indices = torch.argsort(scores)
    print(f"Saving all {len(selected_indices)} poses sorted by energy...")

    print(f"\nTop 5 energies:")
    for rank in range(min(5, len(selected_indices))):
        idx_int = int(selected_indices[rank].item())
        orig_cid = representative_cids[idx_int]
        score = scores[idx_int].item()
        print(f"Rank {rank+1}: Conformer {orig_cid}, Energy: {score:.3f} kcal/mol")

    if len(selected_indices) > 5:
        print(f"... ({len(selected_indices) - 5} more poses)")

    writer = Chem.SDWriter(output_path)
    for rank, idx in enumerate(selected_indices):
        idx_int = int(idx.item())
        orig_cid = representative_cids[idx_int]
        score = scores[idx_int].item()
        initial_score = initial_scores[idx_int].item() if initial_scores is not None else None

        out_mol = Chem.Mol(mol)
        new_conf = Chem.Conformer(out_mol.GetNumAtoms())

        best_coords = aligned_coords[idx_int].cpu().numpy()

        for atom_idx in range(out_mol.GetNumAtoms()):
            x, y, z = best_coords[atom_idx]
            new_conf.SetAtomPosition(atom_idx, Point3D(float(x), float(y), float(z)))

        out_mol.RemoveAllConformers()
        new_conf.SetId(0)
        out_mol.AddConformer(new_conf)

        out_mol.SetProp("_Name", f"Conformer_{orig_cid}_Rank_{rank+1}")
        out_mol.SetProp("Vina_Score", f"{score:.4f}")
        out_mol.SetProp("Vina_Score_Final", f"{score:.4f}")
        if initial_score is not None:
            out_mol.SetProp("Vina_Score_Initial", f"{initial_score:.4f}")
            out_mol.SetProp("Vina_Score_Delta", f"{score - initial_score:.4f}")
        out_mol.SetProp("Rank", str(rank+1))
        if position_labels is not None:
            out_mol.SetProp("MCS_Position", str(position_labels[idx_int]))

        writer.write(out_mol)

    writer.close()
    print(f"✓ Saved {len(selected_indices)} poses to {output_path}")
    return selected_indices
