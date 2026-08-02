"""XS-like atom typing for differentiable Vina-family scoring."""

from __future__ import annotations

import os
from collections.abc import Iterable

import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

ATOM_TYPING_VERSION = "inferred-xs-v2"

_FDEF = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
_FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(_FDEF)

_VINA_RADIUS = {
    "C_H": 1.9,
    "C_P": 1.9,
    "N_P": 1.8,
    "N_D": 1.8,
    "N_A": 1.8,
    "N_DA": 1.8,
    "O_P": 1.7,
    "O_D": 1.7,
    "O_A": 1.7,
    "O_DA": 1.7,
    "S_P": 2.0,
    "P_P": 2.1,
    "F_H": 1.5,
    "Cl_H": 1.8,
    "Br_H": 2.0,
    "I_H": 2.2,
    "Si": 2.2,
    "At": 2.3,
    "Met_D": 1.2,
    "X": 1.8,
    "H": 0.0,
}

_VINARDO_RADIUS = {
    **_VINA_RADIUS,
    "C_H": 2.0,
    "C_P": 2.0,
    "N_P": 1.7,
    "N_D": 1.7,
    "N_A": 1.7,
    "N_DA": 1.7,
    "O_P": 1.6,
    "O_D": 1.6,
    "O_A": 1.6,
    "O_DA": 1.6,
}

_METALS = {
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 55, 56, 57,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
}

# Fallback protein chemistry for unsanitized PDB-derived molecules. RDKit's
# feature factory is used first; these rules fill missing obvious cases.
_PROTEIN_DONORS: dict[str, set[str]] = {
    "ARG": {"NE", "NH1", "NH2"},
    "ASN": {"ND2"},
    "GLN": {"NE2"},
    "HIS": {"ND1", "NE2"},
    "LYS": {"NZ"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TRP": {"NE1"},
    "TYR": {"OH"},
}
_PROTEIN_ACCEPTORS: dict[str, set[str]] = {
    "ASN": {"OD1"},
    "ASP": {"OD1", "OD2"},
    "GLN": {"OE1"},
    "GLU": {"OE1", "OE2"},
    "HIS": {"ND1", "NE2"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
}


def _feature_atom_sets(mol: Chem.Mol) -> tuple[set[int], set[int]]:
    donors: set[int] = set()
    acceptors: set[int] = set()
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
        for feature in _FEATURE_FACTORY.GetFeaturesForMol(mol):
            if feature.GetFamily() == "Donor":
                donors.update(feature.GetAtomIds())
            elif feature.GetFamily() == "Acceptor":
                acceptors.update(feature.GetAtomIds())
    except Exception:
        # Protein PDB blocks and reaction products can be only partially
        # sanitized. Residue/element rules below still provide deterministic
        # typing instead of silently failing the entire scoring pass.
        pass
    return donors, acceptors


def _apply_pdb_fallbacks(mol: Chem.Mol, donors: set[int], acceptors: set[int]) -> None:
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        idx = atom.GetIdx()
        residue = info.GetResidueName().strip().upper()
        name = info.GetName().strip().upper()
        if residue in {"HOH", "WAT", "H2O", "DOD", "SOL"} and name in {"O", "OW", "OH2"}:
            donors.add(idx)
            acceptors.add(idx)
            continue
        if name == "N" and residue != "PRO":
            donors.add(idx)
        if name in {"O", "OXT"}:
            acceptors.add(idx)
        if name in _PROTEIN_DONORS.get(residue, set()):
            donors.add(idx)
        if name in _PROTEIN_ACCEPTORS.get(residue, set()):
            acceptors.add(idx)


def _carbon_is_polar(atom: Chem.Atom) -> bool:
    return any(neighbor.GetAtomicNum() not in {1, 6} for neighbor in atom.GetNeighbors())


def infer_xs_atom_types(mol: Chem.Mol) -> list[str]:
    """Infer Vina XS-like atom types from an RDKit molecule.

    Exact PDBQT parity requires authoritative AutoDock atom types. This function
    provides a deterministic approximation for SMILES/SDF/PDB workflows and the
    resulting scorer records ``inferred-xs-v2`` in every output.
    """
    donors, acceptors = _feature_atom_sets(mol)
    _apply_pdb_fallbacks(mol, donors, acceptors)

    result: list[str] = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atomic_number = atom.GetAtomicNum()
        donor = idx in donors
        acceptor = idx in acceptors
        if atomic_number == 1:
            atom_type = "H"
        elif atomic_number == 6:
            atom_type = "C_P" if _carbon_is_polar(atom) else "C_H"
        elif atomic_number == 7:
            atom_type = "N_DA" if donor and acceptor else "N_D" if donor else "N_A" if acceptor else "N_P"
        elif atomic_number == 8:
            atom_type = "O_DA" if donor and acceptor else "O_D" if donor else "O_A" if acceptor else "O_P"
        elif atomic_number == 16:
            atom_type = "S_P"
        elif atomic_number == 15:
            atom_type = "P_P"
        elif atomic_number == 9:
            atom_type = "F_H"
        elif atomic_number == 17:
            atom_type = "Cl_H"
        elif atomic_number == 35:
            atom_type = "Br_H"
        elif atomic_number == 53:
            atom_type = "I_H"
        elif atomic_number == 14:
            atom_type = "Si"
        elif atomic_number == 85:
            atom_type = "At"
        elif atomic_number in _METALS:
            atom_type = "Met_D"
        else:
            atom_type = "X"
        result.append(atom_type)
    return result


def _to_indicator(indices: Iterable[int], size: int, device: torch.device) -> torch.Tensor:
    values = torch.zeros(size, dtype=torch.float32, device=device)
    valid = [idx for idx in indices if 0 <= idx < size]
    if valid:
        values[torch.tensor(valid, dtype=torch.long, device=device)] = 1.0
    return values


def compute_atom_features(mol: Chem.Mol, device: torch.device | str) -> dict[str, torch.Tensor | tuple[str, ...] | str]:
    """Return tensors required by Vina, Vinardo and SoftDock scorers."""
    device = torch.device(device)
    atom_types = infer_xs_atom_types(mol)
    donors, acceptors = _feature_atom_sets(mol)
    _apply_pdb_fallbacks(mol, donors, acceptors)

    radius_vina = torch.tensor([_VINA_RADIUS[atom_type] for atom_type in atom_types], dtype=torch.float32, device=device)
    radius_vinardo = torch.tensor(
        [_VINARDO_RADIUS[atom_type] for atom_type in atom_types], dtype=torch.float32, device=device
    )
    active = torch.tensor([atom_type != "H" for atom_type in atom_types], dtype=torch.bool, device=device)
    hydrophobic = torch.tensor(
        [atom_type in {"C_H", "F_H", "Cl_H", "Br_H", "I_H"} for atom_type in atom_types],
        dtype=torch.float32,
        device=device,
    )
    donor = _to_indicator(donors, mol.GetNumAtoms(), device)
    acceptor = _to_indicator(acceptors, mol.GetNumAtoms(), device)

    # Match official XS semantics: sulfur is S_P and does not participate in the
    # non-directional hydrogen-bond term; metals are donors.
    for idx, atom_type in enumerate(atom_types):
        if atom_type == "S_P":
            donor[idx] = 0.0
            acceptor[idx] = 0.0
        elif atom_type == "Met_D":
            donor[idx] = 1.0
            acceptor[idx] = 0.0
        elif atom_type == "H":
            donor[idx] = 0.0
            acceptor[idx] = 0.0

    return {
        "radius_vina": radius_vina,
        "radius_vinardo": radius_vinardo,
        "hydrophobic": hydrophobic,
        "donor": donor,
        "acceptor": acceptor,
        "active": active,
        "xs_types": tuple(atom_types),
        "typing_version": ATOM_TYPING_VERSION,
    }
