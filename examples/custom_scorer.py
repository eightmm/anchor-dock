import torch
import torch.nn as nn

from anchor_dock import dock_interaction


class ContactScorer(nn.Module):
    def forward(self, ligand_coords, receptor_coords, ligand_features, receptor_features):
        del ligand_features, receptor_features
        receptor = receptor_coords.unsqueeze(0).expand(ligand_coords.shape[0], -1, -1)
        distances = torch.cdist(ligand_coords, receptor)
        return -torch.exp(-((distances - 3.5) / 1.5).square()).sum(dim=(1, 2))


result = dock_interaction(
    "pocket.pdb",
    "CCO",
    receptor_residue="ASP189:A",
    receptor_atom="OD1",
    ligand_smarts="[O:1]",
    target_distance=3.0,
    distance_tolerance=0.5,
    scorer=ContactScorer(),
    num_candidates=64,
)
print(result)
