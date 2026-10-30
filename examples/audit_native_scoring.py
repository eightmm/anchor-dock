"""Compare full prepared-ligand scores with native Vina/Vinardo.

Optional audit tools (not runtime dependencies)::

    uv pip install meeko==0.8.0 vina==1.2.7 gemmi scipy
    uv run --no-sync python examples/audit_native_scoring.py --output report.json

The controlled receptor is prepared glycine. A bundled 10gs *ligand* is included;
this is a scoring-contract experiment, not a 10gs docking-quality benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import tempfile
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from anchor_dock.core.native_vina import NATIVE_VINA_SCORE_CONTRACT, native_vina_components
from anchor_dock.core.prepared import PreparedLigand
from anchor_dock.core.scoring import PairwiseScorer

REPOSITORY = Path(__file__).resolve().parents[1]
CASES = (("ethanol", "CCO"), ("secondary_amine", "CCNCCO"),
         ("amide", "CC(=O)NCCO"), ("pyridine", "c1ccncc1CCO"),
         ("acetate", "CC(=O)[O-]"), ("nitro", "CC[N+](=O)[O-]"),
         ("branched", "CC(C)CC(=O)NCCO"))


def index_mapping(pdbqt: str) -> dict[int, int]:
    """Use Meeko's explicit original-index map, never guessed atom matching."""
    mapping = {}
    for line in pdbqt.splitlines():
        if line.startswith("REMARK INDEX MAP "):
            values = [int(v) for v in line.split()[3:]]
            if len(values) % 2:
                raise ValueError("invalid Meeko INDEX MAP")
            for original, serial in zip(values[::2], values[1::2], strict=True):
                if serial in mapping:
                    raise ValueError("duplicate Meeko INDEX MAP serial")
                mapping[serial] = original - 1
    if not mapping:
        raise ValueError("Meeko did not emit an explicit index map")
    return mapping


def with_coordinates(mol: Chem.Mol, xyz: np.ndarray) -> Chem.Mol:
    result = Chem.Mol(mol)
    result.RemoveAllConformers()
    conformer = Chem.Conformer(result.GetNumAtoms())
    for index, value in enumerate(xyz):
        conformer.SetAtomPosition(index, tuple(float(v) for v in value))
    result.AddConformer(conformer)
    return result


def prepare(mol: Chem.Mol):
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    setups = MoleculePreparation().prepare(mol)
    if len(setups) != 1:
        raise ValueError("audit supports one ordinary prepared ligand")
    text, success, error = PDBQTWriterLegacy.write_string(setups[0], add_index_map=True)
    if not success:
        raise ValueError(error)
    return PreparedLigand.from_pdbqt(mol, text, serial_to_atom_index=index_mapping(text)), text


def embed(smiles: str, seed: int) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    if AllChem.EmbedMolecule(mol, parameters) != 0:
        raise ValueError(f"conformer generation failed: {smiles}")
    return mol


def poses(mol: Chem.Mol):
    xyz = np.asarray(mol.GetConformer().GetPositions())
    heavy = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    xyz = xyz - xyz[heavy].mean(axis=0)
    for pose, (angle, shift) in enumerate(((0.0, (4.0, 1.2, 0.3)), (0.7, (5.5, -1.1, 0.8)),
                                          (-1.2, (7.0, 2.0, -0.5)))):
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        yield pose, with_coordinates(mol, xyz @ rotation.T + np.asarray(shift))


def audit(seed: int = 713, fixture_output: Path | None = None) -> dict[str, object]:
    from vina import Vina

    versions = {name: importlib.metadata.version(name) for name in ("rdkit", "meeko", "vina", "torch")}
    if versions["vina"] != "1.2.7" or versions["meeko"] != "0.8.0":
        raise RuntimeError("this private-binding audit requires vina==1.2.7 and meeko==0.8.0; revalidate other versions")
    torch.set_num_threads(1)
    rec_mol = embed("NCC(=O)O", seed)
    xyz = np.asarray(rec_mol.GetConformer().GetPositions())
    receptor, rec_text = prepare(with_coordinates(rec_mol, xyz - xyz.mean(axis=0)))
    molecules = [(name, embed(smiles, seed + i + 1)) for i, (name, smiles) in enumerate(CASES)]
    path_10gs = REPOSITORY / "examples/10gs/10gs_ligand.sdf"
    mol_10gs = Chem.SDMolSupplier(str(path_10gs), removeHs=False)[0]
    if mol_10gs is None:
        raise ValueError("failed to read bundled 10gs ligand")
    molecules.append(("10gs_ligand", Chem.AddHs(mol_10gs, addCoords=True)))
    rows = []
    fixture_cases = {}
    with tempfile.TemporaryDirectory(prefix="anchor-native-score-") as directory:
        path = Path(directory) / "receptor.pdbqt"
        path.write_text("\n".join(line for line in rec_text.splitlines()
                                  if line.startswith(("ATOM  ", "HETATM"))) + "\n")
        for name in ("vina", "vinardo"):
            native = Vina(sf_name=name, cpu=1, seed=seed, no_refine=False, verbosity=0)
            native.set_receptor(str(path))
            # Required for the box-validity check; final scoring uses explicit
            # rigid-receptor atoms with no_refine=False, not these search maps.
            native.compute_vina_maps(center=[3., 0., 0.], box_size=[30., 30., 30.], spacing=.375)
            for ligand_name, molecule in molecules:
                for pose, placed in poses(molecule):
                    ligand, text = prepare(placed)
                    native.set_ligand_from_string(text)
                    # Explicit zero intra reference, unrounded pinned private
                    # binding; public Vina.score rounds to three decimals.
                    expected = list(native._vina.score(0.0))
                    xyz = ligand.coordinates("cpu", dtype=torch.float64).unsqueeze(0)
                    results = {}
                    for cap in (None, 1000.0):
                        scorer = PairwiseScorer(name, positive_energy_cap=cap).prepare(
                            ligand.features("cpu"), receptor.coordinates("cpu", dtype=torch.float64),
                            receptor.features("cpu"), num_rotatable_bonds=ligand.vina_num_tors,
                            intramolecular_mask=ligand.intramolecular_mask("cpu"))
                        raw = scorer.raw_components(xyz)
                        score = scorer.report(raw, intramolecular_reference=0.0)
                        results["analytic" if cap is None else "native_positive_cap"] = dict(
                            intermolecular=float(raw.intermolecular[0]),
                            intramolecular=float(raw.intramolecular[0]),
                            total_zero_reference=float(score.score[0]))
                    raw = native_vina_components(scorer, xyz)
                    score = scorer.report(raw, intramolecular_reference=0.0)
                    results["native_fast_reference"] = dict(
                        intermolecular=float(raw.intermolecular[0]), intramolecular=float(raw.intramolecular[0]),
                        total_zero_reference=float(score.score[0]))
                    components = dict(intermolecular=expected[1] + expected[2],
                                      intramolecular=expected[3] + expected[4] + expected[5],
                                      total_zero_reference=expected[0])
                    if pose == 0 and ligand_name in {"secondary_amine", "branched", "10gs_ligand"}:
                        fixture = fixture_cases.setdefault(ligand_name, dict(
                            molblock=Chem.MolToMolBlock(placed), pdbqt=text,
                            serial_to_atom_index=index_mapping(text), expected={},
                            vina_num_tors=ligand.vina_num_tors,
                            intramolecular_pairs=ligand.intramolecular_pairs))
                        fixture["expected"][name] = components
                    errors = {mode: {key: abs(values[key] - components[key]) for key in components}
                              for mode, values in results.items()}
                    rows.append(dict(ligand=ligand_name, pose=pose, scorer=name,
                                     atoms=ligand.mol.GetNumAtoms(), torsdof=ligand.torsdof,
                                     vina_num_tors=ligand.vina_num_tors,
                                     intra_pairs=len(ligand.intramolecular_pairs),
                                     pdbqt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                     native=components, values=results, absolute_errors=errors))
    errors = {mode: {component: max(row["absolute_errors"][mode][component] for row in rows)
                     for component in ("intermolecular", "intramolecular", "total_zero_reference")}
              for mode in ("analytic", "native_positive_cap", "native_fast_reference")}
    if fixture_output is not None:
        fixture_output.parent.mkdir(parents=True, exist_ok=True)
        fixture_output.write_text(json.dumps(dict(
            source="examples/audit_native_scoring.py; Vina._vina.score(0.0), rigid glycine receptor",
            versions=versions, seed=seed,
            receptor=dict(molblock=Chem.MolToMolBlock(with_coordinates(rec_mol, np.asarray(
                rec_mol.GetConformer().GetPositions()) - np.asarray(rec_mol.GetConformer().GetPositions()).mean(axis=0))),
                pdbqt=rec_text, serial_to_atom_index=index_mapping(rec_text)),
            cases=fixture_cases), indent=2) + "\n")
    return dict(scope="complete prepared-ligand scoring in a controlled multi-atom rigid receptor",
                seed=seed, python=platform.python_version(),
                versions=versions,
                cuda_available=torch.cuda.is_available(), device="cpu",
                native_call="Vina._vina.score(0.0), no_refine=False; private binding pinned to 1.2.7",
                native_score_contract=NATIVE_VINA_SCORE_CONTRACT,
                preparation="Meeko explicit-H graph, INDEX MAP, shared PDBQT-rounded coordinates",
                receptor="ETKDGv3 glycine, prepared once and held rigid",
                ligand_10gs_sha256=hashlib.sha256(path_10gs.read_bytes()).hexdigest(),
                cases=len(rows), max_absolute_errors=errors,
                limitations=["Analytic equations do not emulate native radial lookup tables.",
                             "Native distance-based bond perception may differ from supplied chemistry graph.",
                             "No macrocycle, flexible-receptor, metal-coordination or GPU validation.",
                             "Scoring-contract check; no docking-quality or affinity-prediction claim."], rows=rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=713)
    parser.add_argument("--output", type=Path, default=Path("native-scoring-parity.json"))
    parser.add_argument("--fixture-output", type=Path, help="also write three pinned native-oracle fixtures")
    args = parser.parse_args()
    result = audit(args.seed, args.fixture_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
