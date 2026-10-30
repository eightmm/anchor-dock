"""Compare graph inference to real Meeko preparation of seeded 3D molecules.

Optional audit dependency: meeko==0.8.0 (with gemmi/scipy). It is not a runtime
AnchorDock dependency. This ordinary-ligand fixture audit does not certify all
chemistry, receptor preparation, macrocycles, charges, or full Vina scores.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from anchor_dock.core.features import compute_atom_features, compute_atom_features_from_ad4

CASES = {
    'acetate': 'CC(=O)[O-]', 'acetic_acid': 'CC(=O)O', 'methylamine': 'CN',
    'dimethylamine': 'CNC', 'trimethylamine': 'CN(C)C', 'ammonium': 'C[NH3+]',
    'quaternary_ammonium': 'C[N+](C)(C)C', 'imidazole': 'c1ncc[nH]1',
    'pyridine': 'c1ccncc1', 'pyrrole': 'c1cc[nH]c1', 'aniline': 'Nc1ccccc1',
    'amide': 'CC(=O)N', 'nitrile': 'CC#N', 'nitromethane': 'C[N+](=O)[O-]',
    'phosphate': 'OP(=O)([O-])[O-]', 'sulfate': 'OS(=O)(=O)[O-]',
    'phenol': 'Oc1ccccc1', 'alcohol': 'CCO', 'thiol': 'CS', 'sulfide': 'CSC',
    'sulfonamide': 'CS(=O)(=O)N', 'urea': 'NC(=O)N',
    'guanidinium': 'NC(=[NH2+])N', 'fluoromethane': 'CF',
}


def _reference_xs(mol, ad4):
    # Independently express official Vina model::assign_types for these
    # ordinary fixtures. Authoritative acceptors are AD NA/OA, donors bonded HD.
    result = []
    for atom, kind in zip(mol.GetAtoms(), ad4, strict=True):
        z = atom.GetAtomicNum()
        if z == 1:
            xs = 'H'
        elif z == 6:
            xs = 'C_P' if any(n.GetAtomicNum() not in (1, 6) for n in atom.GetNeighbors()) else 'C_H'
        elif z in (7, 8):
            donor = any(ad4[n.GetIdx()] == 'HD' for n in atom.GetNeighbors())
            acceptor = kind in ('NA', 'OA')
            role = 'DA' if donor and acceptor else 'D' if donor else 'A' if acceptor else 'P'
            xs = ('N_' if z == 7 else 'O_') + role
        else:
            xs = {16: 'S_P', 15: 'P_P', 9: 'F_H', 17: 'Cl_H', 35: 'Br_H', 53: 'I_H'}[z]
        result.append(xs)
    return result


def audit(seed=713):
    from meeko import MoleculePreparation

    rows = []
    for name, smiles in CASES.items():
        heavy = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(heavy)
        options = AllChem.ETKDGv3()
        options.randomSeed, options.numThreads = seed, 1
        if AllChem.EmbedMolecule(mol, options) != 0:
            raise RuntimeError(f'embedding failed: {name}')
        setup = MoleculePreparation().prepare(mol)[0]
        if len(setup.atoms) != mol.GetNumAtoms() or any(a.index != i for i, a in enumerate(setup.atoms)):
            raise RuntimeError(f'prepared atom mapping changed: {name}')
        native = [a.atom_type for a in setup.atoms]
        expected = _reference_xs(mol, native)
        inferred = list(compute_atom_features(heavy, 'cpu', strict=True)['xs_types'])
        explicit = list(compute_atom_features(mol, 'cpu', strict=True)['xs_types'])
        supplied = list(compute_atom_features_from_ad4(mol, native, 'cpu')['xs_types'])
        match = inferred == expected[:heavy.GetNumAtoms()] and explicit == expected and supplied == expected
        rows.append(dict(name=name, smiles=smiles, native_ad4=native, expected_xs=expected,
                         inferred_heavy=inferred, inferred_explicit=explicit, supplied_ad4_xs=supplied, match=match))
    return dict(meeko=importlib.metadata.version('meeko'), rdkit=importlib.metadata.version('rdkit'),
                seed=seed, cases=len(rows), all_match=all(r['match'] for r in rows), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=713)
    args = parser.parse_args()
    report = audit(args.seed)
    payload = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end='')
    if not report['all_match']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
