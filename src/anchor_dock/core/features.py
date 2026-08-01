"""Atom feature extraction shared by all AnchorDock strategies."""

from __future__ import annotations

import os

import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

_FDEF = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
_FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(_FDEF)


def compute_vina_features(mol: Chem.Mol, device: torch.device | str) -> dict[str, torch.Tensor]:
    """Return Vina-style vdW, hydrophobe, donor and acceptor atom features."""
    device = torch.device(device)
    num_atoms = mol.GetNumAtoms()
    ptable = Chem.GetPeriodicTable()

    radii = torch.zeros(num_atoms, dtype=torch.float32, device=device)
    hydrophobic = torch.zeros_like(radii)
    donor = torch.zeros_like(radii)
    acceptor = torch.zeros_like(radii)

    for idx, atom in enumerate(mol.GetAtoms()):
        radii[idx] = ptable.GetRvdw(atom.GetAtomicNum())

    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
    except Exception:
        pass

    for feature in _FEATURE_FACTORY.GetFeaturesForMol(mol):
        family = feature.GetFamily()
        for atom_idx in feature.GetAtomIds():
            if family == "Hydrophobe":
                hydrophobic[atom_idx] = 1.0
            elif family == "Donor":
                donor[atom_idx] = 1.0
            elif family == "Acceptor":
                acceptor[atom_idx] = 1.0

    return {
        "vdw": radii,
        "hydro": hydrophobic,
        "hbd": donor,
        "hba": acceptor,
    }
