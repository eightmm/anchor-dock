import os
from pathlib import Path

from rdkit import Chem

from lig_align.aligner import LigandAligner
from lig_align.io import process_query_ligand

EXAMPLES = [
    (
        "query_low_acemetacin",
        "Acemetacin",
        "COc1ccc2c(c1)c(C(=O)NCc1ccccc1)c(CC(=O)O)n2C",
    ),
    (
        "query_mid_diclofenac",
        "Diclofenac",
        "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    ),
    (
        "query_high_flurbiprofen",
        "Flurbiprofen",
        "CC(C)c1ccc(cc1)[C@@H](C)C(=O)O",
    ),
    (
        "ref_lower_cyclohexyl_ala",
        "Cyclohexyl-plus-Ala analog",
        "[NH3+][C@@H](CCC(=O)N[C@@H](CSC1CCCCC1)C(=O)N[C@@H](C(=O)[O-])C)C(=O)[O-]",
    ),
    (
        "ref_mid_pyridyl_cyclohexyl",
        "Pyridyl-plus-cyclohexyl analog",
        "[NH3+][C@@H](CCC(=O)N[C@@H](CSC1CCCCC1)C(=O)N[C@@H](C(=O)[O-])c1ccccn1)C(=O)[O-]",
    ),
    (
        "ref_high_cyclohexylmethyl",
        "Cyclohexylmethyl analog",
        "[NH3+][C@@H](CCC(=O)N[C@@H](CSC1CCCCC1)C(=O)N[C@@H](C(=O)[O-])c1ccccc1)C(=O)[O-]",
    ),
]


def compute_coverage(ref_mol, smiles: str):
    aligner = LigandAligner(device="cpu")
    query_mol, canonical = process_query_ligand(smiles)
    mapping = aligner.step2_find_mcs(ref_mol, query_mol, return_all_positions=False)
    ref_cov = len(mapping) / ref_mol.GetNumHeavyAtoms() * 100
    query_cov = len(mapping) / query_mol.GetNumHeavyAtoms() * 100
    return canonical, len(mapping), ref_cov, query_cov


def main():
    repo_root = Path(__file__).resolve().parents[1]
    protein = repo_root / "examples/10gs/10gs_pocket.pdb"
    ref_sdf = repo_root / "examples/10gs/10gs_ligand.sdf"
    out_dir = repo_root / "examples/10gs/visualizations/coverage"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_mol = Chem.SDMolSupplier(str(ref_sdf))[0]
    if ref_mol is None:
        raise ValueError(f"Failed to load reference ligand: {ref_sdf}")

    for slug, label, smiles in EXAMPLES:
        canonical, mcs_atoms, ref_cov, query_cov = compute_coverage(ref_mol, smiles)
        title = f"{label} | MCS {mcs_atoms} atoms | Ref {ref_cov:.1f}% | Query {query_cov:.1f}%"
        output = out_dir / f"{slug}_100steps.gif"
        cmd = (
            f'UV_CACHE_DIR=.uv-cache uv run python scripts/vis_opt_gif.py '
            f'-q "{canonical}" '
            f'-o "{output}" '
            f'-p "{protein}" '
            f'-r "{ref_sdf}" '
            f'-t "{title}" '
            f'--steps 100'
        )
        print(cmd)
        rc = os.system(cmd)
        if rc != 0:
            raise SystemExit(f"Failed to render {label} (exit={rc})")


if __name__ == "__main__":
    main()
