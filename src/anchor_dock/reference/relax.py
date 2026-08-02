"""Constrained force-field relaxation for reference-guided poses."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass(frozen=True)
class RelaxationResult:
    requested: bool
    applied: bool
    method: str
    message: str


def relax_pose_with_fixed_core(
    mol: Chem.Mol,
    conf_id: int,
    fixed_indices: set[int],
    *,
    requested: bool = True,
    max_iters: int = 500,
    mmff_props: object | None = None,
) -> RelaxationResult:
    """Relax movable atoms exactly once while keeping all anchor atoms fixed."""
    if not requested:
        return RelaxationResult(False, False, "none", "disabled")
    movable_atoms = mol.GetNumAtoms() - len(fixed_indices)
    if movable_atoms <= 0:
        return RelaxationResult(True, False, "none", "all atoms are fixed by the anchor")
    if movable_atoms < 2:
        return RelaxationResult(True, False, "none", "fewer than two atoms remain movable")

    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
    except Exception:
        pass

    if mmff_props is None:
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
    if mmff_props is not None:
        try:
            force_field = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id)
        except Exception:
            force_field = None
        if force_field is not None:
            for atom_idx in fixed_indices:
                force_field.AddFixedPoint(int(atom_idx))
            try:
                status = force_field.Minimize(maxIts=max_iters)
                message = "converged" if status == 0 else "iteration limit reached"
                return RelaxationResult(True, True, "MMFF94", message)
            except RuntimeError:
                pass

    try:
        force_field = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
    except Exception:
        force_field = None
    if force_field is not None:
        for atom_idx in fixed_indices:
            force_field.AddFixedPoint(int(atom_idx))
        try:
            status = force_field.Minimize(maxIts=max_iters)
            message = "converged" if status == 0 else "iteration limit reached"
            return RelaxationResult(True, True, "UFF", message)
        except RuntimeError as exc:
            return RelaxationResult(True, False, "none", f"MMFF and UFF failed: {exc}")

    return RelaxationResult(True, False, "none", "no usable force field")
