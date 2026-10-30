"""Versioned Vina-family XS typing from molecular graphs or prepared AD4 types.

AD4-compatible typing is a force-field convention, not a universal medicinal
chemistry H-bond definition. In particular, the AD4 preset assigns oxygen OA
and treats several trigonal nitrogen motifs as nonacceptors. PDB residue
fallbacks remain explicitly inferred; unresolved histidines are diagnosed.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from rdkit import Chem

ATOM_TYPING_VERSION = "inferred-xs-v3"
AUTHORITATIVE_TYPING_VERSION = "prepared-ad4-xs-v1"

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

# Supported generic metal inventory in official Vina atom_constants.h.
# Generic Met_D is not a directional/coordination model.
_METAL_ELEMENTS = {"Mg": 12, "Ca": 20, "Mn": 25, "Fe": 26, "Zn": 30,
                   "Cu": 29, "Na": 11, "K": 19, "Hg": 80, "Co": 27,
                   "U": 92, "Cd": 48, "Ni": 28}
_METALS = set(_METAL_ELEMENTS.values())

# Meeko's AD4 preset and official Vina AD->XS conversion are the reference
# conventions, not RDKit BaseFeatures.fdef pharmacophore donor/acceptor labels.
# These independently applied motif predicates exclude neutral trigonal N in
# aromatic/conjugated amide/triazene environments from the default NA class.
_NONACCEPTOR_N = tuple(Chem.MolFromSmarts(pattern) for pattern in (
    "[#7X3v3][a]", "[#7X3v3][#6X3v4]", "[#7X3v3][NX2]=[*]",
))
_PROTEIN_DONORS = {
    "ARG": {"NE", "NH1", "NH2"}, "ASN": {"ND2"}, "GLN": {"NE2"},
    "LYS": {"NZ"}, "LYN": {"NZ"}, "SER": {"OG"}, "THR": {"OG1"},
    "TRP": {"NE1"}, "TYR": {"OH"},
}
_PROTEIN_ACCEPTORS = {
    "ASN": {"OD1"}, "ASP": {"OD1", "OD2"}, "ASH": {"OD1", "OD2"},
    "GLN": {"OE1"}, "GLU": {"OE1", "OE2"}, "GLH": {"OE1", "OE2"},
    "LYN": {"NZ"}, "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
}
_PROTEIN_RESIDUES = set((
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL "
    "ASH GLH CYM CYX LYN HID HIE HIP HSD HSE HSP"
).split())
_HISTIDINE_STATES = {
    "HID": ({"ND1"}, {"NE2"}), "HSD": ({"ND1"}, {"NE2"}),
    "HIE": ({"NE2"}, {"ND1"}), "HSE": ({"NE2"}, {"ND1"}),
    "HIP": ({"ND1", "NE2"}, set()), "HSP": ({"ND1", "NE2"}, set()),
}


def _explicit_hydrogen(atom: Chem.Atom) -> bool:
    return any(neighbor.GetAtomicNum() == 1 for neighbor in atom.GetNeighbors())


def _feature_atom_sets(mol: Chem.Mol) -> tuple[set[int], set[int]]:
    """Infer graph roles using AD4-compatible N/O predicates, once per call."""
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    nonacceptor_n = {
        hit[0] for query in _NONACCEPTOR_N for hit in mol.GetSubstructMatches(query)
    }
    donors, acceptors = set(), set()
    for atom in mol.GetAtoms():
        idx, number = atom.GetIdx(), atom.GetAtomicNum()
        if number not in (7, 8):
            continue
        if atom.GetTotalNumHs(includeNeighbors=True) > 0:
            donors.add(idx)
        if number == 8 or (idx not in nonacceptor_n and atom.GetFormalCharge() <= 0):
            acceptors.add(idx)
    return donors, acceptors


def _residue_key(info: Chem.AtomPDBResidueInfo) -> tuple[str, int, str, str]:
    return (info.GetChainId(), info.GetResidueNumber(), info.GetInsertionCode(), info.GetResidueName().strip().upper())


def _apply_pdb_fallbacks(mol: Chem.Mol, donors: set[int], acceptors: set[int]) -> tuple[str, ...]:
    """Apply declared residue templates, never choose an unspecified HIS tautomer."""
    ambiguities = []
    histidine_hydrogens: dict[tuple[str, int, str, str], set[str]] = {}
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info and info.GetResidueName().strip().upper() in {"HIS", *_HISTIDINE_STATES} and _explicit_hydrogen(atom):
            histidine_hydrogens.setdefault(_residue_key(info), set()).add(info.GetName().strip().upper())
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None or atom.GetAtomicNum() not in (7, 8):
            continue
        idx = atom.GetIdx()
        residue, name = info.GetResidueName().strip().upper(), info.GetName().strip().upper()
        # Unsanitized PDB connectivity cannot establish reliable implicit H or
        # aromatic bond orders. Replace graph guesses with a declared template.
        donors.discard(idx)
        acceptors.discard(idx)
        if residue in {"HOH", "WAT", "H2O", "DOD", "SOL"} and name in {"O", "OW", "OH2"}:
            donors.add(idx)
            acceptors.add(idx)
            continue
        if residue not in _PROTEIN_RESIDUES:
            ambiguities.append(f"atom {idx}: unresolved PDB residue {residue}/{name}; provide prepared AD4 types")
            continue
        if residue in _HISTIDINE_STATES and name in {"ND1", "NE2"}:
            donor_names, acceptor_names = _HISTIDINE_STATES[residue]
            observed = histidine_hydrogens.get(_residue_key(info), set()) & {"ND1", "NE2"}
            if observed and observed != donor_names:
                ambiguities.append(f"atom {idx}: {residue} label conflicts with explicit histidine H; provide consistent preparation")
                continue
            if name in donor_names:
                donors.add(idx)
            if name in acceptor_names:
                acceptors.add(idx)
            continue
        if residue == "HIS" and name in {"ND1", "NE2"}:
            protonated = histidine_hydrogens.get(_residue_key(info), set()) & {"ND1", "NE2"}
            if protonated:
                if name in protonated:
                    donors.add(idx)
                elif atom.GetFormalCharge() <= 0:
                    acceptors.add(idx)
            else:
                ambiguities.append(f"atom {idx}: unresolved HIS {name} protonation; provide explicit H, HID/HIE/HIP, or AD4 types")
            continue
        if name == "N" and residue != "PRO":
            donors.add(idx)
        if name in {"O", "OXT"}:
            acceptors.add(idx)
        if name in _PROTEIN_DONORS.get(residue, set()):
            donors.add(idx)
        if name in _PROTEIN_ACCEPTORS.get(residue, set()):
            acceptors.add(idx)
        if _explicit_hydrogen(atom):
            donors.add(idx)
        if atom.GetFormalCharge() > 0:
            acceptors.discard(idx)
    return tuple(ambiguities)


def _carbon_is_polar(atom: Chem.Atom) -> bool:
    return any(neighbor.GetAtomicNum() not in {1, 6} for neighbor in atom.GetNeighbors())


def _infer_xs_atom_types_from_sets(mol: Chem.Mol, donors: set[int], acceptors: set[int]) -> list[str]:
    simple = {1: "H", 16: "S_P", 15: "P_P", 9: "F_H", 17: "Cl_H", 35: "Br_H", 53: "I_H", 14: "Si", 85: "At"}
    result = []
    for atom in mol.GetAtoms():
        idx, number = atom.GetIdx(), atom.GetAtomicNum()
        donor, acceptor = idx in donors, idx in acceptors
        if number == 6:
            kind = "C_P" if _carbon_is_polar(atom) else "C_H"
        elif number in (7, 8):
            role = "DA" if donor and acceptor else "D" if donor else "A" if acceptor else "P"
            kind = ("N_" if number == 7 else "O_") + role
        else:
            kind = simple.get(number, "Met_D" if number in _METALS else "X")
        result.append(kind)
    return result


def _inferred_types(mol: Chem.Mol, strict: bool) -> tuple[list[str], tuple[str, ...]]:
    graph_ambiguities: tuple[str, ...] = ()
    try:
        donors, acceptors = _feature_atom_sets(mol)
    except (RuntimeError, ValueError) as error:
        if not any(atom.GetPDBResidueInfo() is not None for atom in mol.GetAtoms()):
            raise ValueError("graph atom typing requires valid bond/protonation information") from error
        donors, acceptors = set(), set()
        graph_ambiguities = ("graph-role inference unavailable; PDB residue-only inference used",)
    ambiguities = graph_ambiguities + _apply_pdb_fallbacks(mol, donors, acceptors)
    atom_types = _infer_xs_atom_types_from_sets(mol, donors, acceptors)
    unknown = tuple(f"atom {i}: unsupported element {mol.GetAtomWithIdx(i).GetSymbol()}" for i, kind in enumerate(atom_types) if kind == "X")
    ambiguities += unknown
    if strict and ambiguities:
        raise ValueError("ambiguous atom typing: " + "; ".join(ambiguities))
    return atom_types, ambiguities


def infer_xs_atom_types(mol: Chem.Mol, *, strict: bool = False) -> list[str]:
    """Infer versioned XS-like types; strict mode rejects diagnosed ambiguity.

    Complete graph/protonation information is required for ligand inference.
    PDB residue templates remain an approximation. Authoritative prepared
    atom types should use compute_atom_features_from_ad4 instead.
    """
    return _inferred_types(mol, strict)[0]


def _features_from_xs(atom_types: Sequence[str], device: torch.device | str, *, version: str,
                      source: str, ambiguities: tuple[str, ...] = ()) -> dict[str, object]:
    device = torch.device(device)
    return {
        "radius_vina": torch.tensor([_VINA_RADIUS[t] for t in atom_types], dtype=torch.float32, device=device),
        "radius_vinardo": torch.tensor([_VINARDO_RADIUS[t] for t in atom_types], dtype=torch.float32, device=device),
        "hydrophobic": torch.tensor([t in {"C_H", "F_H", "Cl_H", "Br_H", "I_H"} for t in atom_types], dtype=torch.float32, device=device),
        "donor": torch.tensor([t in {"N_D", "N_DA", "O_D", "O_DA", "Met_D"} for t in atom_types], dtype=torch.float32, device=device),
        "acceptor": torch.tensor([t in {"N_A", "N_DA", "O_A", "O_DA"} for t in atom_types], dtype=torch.float32, device=device),
        "active": torch.tensor([t != "H" for t in atom_types], dtype=torch.bool, device=device),
        "xs_types": tuple(atom_types),
        "typing_version": version,
        "typing_source": source,
        "typing_ambiguities": ambiguities,
    }


def compute_atom_features(mol: Chem.Mol, device: torch.device | str, *, strict: bool = False) -> dict[str, object]:
    """Infer AD4-compatible graph roles; PDB ambiguities are explicit metadata."""
    atom_types, ambiguities = _inferred_types(mol, strict)
    return _features_from_xs(atom_types, device, version=ATOM_TYPING_VERSION,
                             source="graph-and-residue-inference", ambiguities=ambiguities)


def compute_atom_features_from_ad4(mol: Chem.Mol, ad4_types: Sequence[str],
                                  device: torch.device | str) -> dict[str, object]:
    """Convert authoritative atom-order-aligned AD4 types to ordinary XS types.

    The graph must retain polar hydrogens and their bonds before heavy-atom
    selection. This function neither guesses protonation nor assigns charges.
    Macrocycle dummy/water types need their own scoring terms and are rejected.
    """
    if isinstance(ad4_types, str):
        raise ValueError("ad4_types must be an atom-order-aligned sequence")
    ad4_types = tuple(ad4_types)
    if len(ad4_types) != mol.GetNumAtoms():
        raise ValueError("one authoritative AD4 type is required per graph atom")
    mol.UpdatePropertyCache(strict=False)
    expected_elements = {"C": 6, "A": 6, "N": 7, "NA": 7, "O": 8, "OA": 8,
                         "S": 16, "SA": 16, "P": 15, "H": 1, "HD": 1,
                         "F": 9, "Cl": 17, "Br": 35, "I": 53, "Si": 14, "At": 85,
                         **_METAL_ELEMENTS}
    for atom, kind in zip(mol.GetAtoms(), ad4_types, strict=True):
        if kind not in expected_elements:
            raise ValueError(f"unsupported authoritative AD4 type {kind!r}")
        if atom.GetAtomicNum() != expected_elements[kind]:
            raise ValueError(f"AD4 type {kind!r} disagrees with element/order at atom {atom.GetIdx()}")
        if kind == "HD" and not any(n.GetAtomicNum() in (7, 8, 9, 15, 16) for n in atom.GetNeighbors()):
            raise ValueError("authoritative HD requires a bonded polar heavy atom")
        if atom.GetAtomicNum() in (7, 8) and atom.GetTotalNumHs(includeNeighbors=False) > 0:
            raise ValueError("authoritative conversion requires explicit polar-H graph atoms")
    donors, acceptors = set(), set()
    for atom, kind in zip(mol.GetAtoms(), ad4_types, strict=True):
        idx = atom.GetIdx()
        if kind in {"NA", "OA"}:
            acceptors.add(idx)
        if atom.GetAtomicNum() in (7, 8) and any(ad4_types[n.GetIdx()] == "HD" for n in atom.GetNeighbors()):
            donors.add(idx)
    atom_types = _infer_xs_atom_types_from_sets(mol, donors, acceptors)
    return _features_from_xs(atom_types, device, version=AUTHORITATIVE_TYPING_VERSION, source="authoritative-ad4-input")
