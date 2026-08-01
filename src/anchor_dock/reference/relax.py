"""Helpers for safe force-field relaxation after MCS coordinate placement."""

from rdkit import Chem
from rdkit.Chem import AllChem


def relax_pose_with_fixed_core(mol: Chem.Mol,
                               conf_id: int,
                               fixed_indices: set[int],
                               max_iters: int = 500,
                               mmff_props=None) -> tuple[bool, str]:
    """
    Relax non-core atoms with MMFF, falling back to UFF when needed.

    Args:
        mmff_props: Precomputed MMFF properties (avoids recomputation per conformer).

    Returns:
        (applied, message)
    """
    num_atoms = mol.GetNumAtoms()
    movable_atoms = num_atoms - len(fixed_indices)

    if movable_atoms <= 0:
        return False, "skipped: all query atoms are fixed by the MCS"
    if movable_atoms < 2:
        return False, "skipped: fewer than two atoms remain movable"

    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
    except Exception:
        pass

    if mmff_props is None:
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
    if mmff_props is not None:
        ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id)
        if ff is not None:
            for atom_idx in fixed_indices:
                ff.AddFixedPoint(atom_idx)
            try:
                ff.Minimize(maxIts=max_iters)
                return True, "applied: MMFF"
            except RuntimeError:
                pass

    uff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
    if uff is not None:
        for atom_idx in fixed_indices:
            uff.AddFixedPoint(atom_idx)
        try:
            uff.Minimize(maxIts=max_iters)
            return True, "applied: UFF fallback"
        except RuntimeError as exc:
            return False, f"failed: MMFF and UFF raised {exc}"

    return False, "failed: no usable force field could be constructed"
