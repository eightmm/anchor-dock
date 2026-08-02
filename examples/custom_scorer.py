import torch
import torch.nn as nn

from anchor_dock import dock_free


class ContactScorer(nn.Module):
    def forward(self, ligand_coords, receptor_coords, ligand_features, receptor_features):
        del ligand_features, receptor_features
        receptor = receptor_coords.unsqueeze(0).expand(ligand_coords.shape[0], -1, -1)
        distances = torch.cdist(ligand_coords, receptor)
        return -torch.exp(-((distances - 3.5) / 1.5).square()).sum(dim=(1, 2))


result = dock_free("pocket.pdb", "CCO", scorer=ContactScorer(), num_starts=64)
print(result)
