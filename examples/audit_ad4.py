"""Native AD4 component audit using optional Meeko/Vina, not runtime dependencies.

Maps are synthetic affine AutoGrid-format fields, not receptor-generated maps.
Run: python examples/audit_ad4.py --output audit.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from anchor_dock.core.ad4 import AD4PairEnergy, AD4Scorer
from anchor_dock.core.prepared import PreparedLigand


def _native_fast_table_probe(prepared):
    """Diagnostic only: reproduce native fast table sampling in r^2 (factor 32)."""
    xyz = prepared.coordinates(dtype=torch.float64)
    total = 0.
    for i, j in prepared.intramolecular_pairs:
        squared_distance = (xyz[i] - xyz[j]).square().sum().item()
        if squared_distance >= 20.48 ** 2:
            continue
        index = int(squared_distance * 32)
        samples = torch.zeros(2, 2, 3, dtype=torch.float64)
        samples[:, 1, 0] = torch.tensor([index / 32, (index + 1) / 32], dtype=torch.float64).sqrt()
        term = AD4PairEnergy([prepared.ad4_types[i], prepared.ad4_types[j]],
                             [prepared.charges[i], prepared.charges[j]], [(0, 1)], dtype=torch.float64)
        mean = term(samples).mean().item()
        total += mean / (1 + max(mean, 0.) / 1000.)
    return total


def audit() -> dict:
    import meeko
    import vina
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    rows = []
    axis = torch.arange(21, dtype=torch.float64) - 10
    z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
    header = ("GRID_PARAMETER_FILE synthetic.gpf\nGRID_DATA_FILE synthetic.maps.fld\n"
              "MACROMOLECULE synthetic.pdbqt\nSPACING 1\nNELEMENTS 20 20 20\nCENTER 0 0 0\n")
    for smiles in ("CCCCCO", "CCOCCN", "CCCOCC(=O)N", "c1ccccc1CCCO"):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        params = AllChem.ETKDGv3()
        params.randomSeed = 713
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError("audit conformer generation failed")
        setup, = MoleculePreparation().prepare(mol)
        text, valid, error = PDBQTWriterLegacy.write_string(setup, add_index_map=True)
        if not valid:
            raise RuntimeError(error)
        mapping = {}
        for line in text.splitlines():
            if line.startswith("REMARK INDEX MAP "):
                pairs = list(map(int, line.split()[3:]))
                mapping.update({pairs[i + 1]: pairs[i] - 1 for i in range(0, len(pairs), 2)})
        prepared = PreparedLigand.from_pdbqt(mol, text, serial_to_atom_index=mapping)
        types = sorted(set(prepared.ad4_types))
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "synthetic"
            maps = {t: (1. + i * .2 + .02 * x + .03 * y - .01 * z) for i, t in enumerate(types)}
            maps["e"] = 2. - .04 * x + .01 * y + .02 * z
            maps["d"] = .5 + .03 * y + .01 * z
            for kind, values in maps.items():
                Path(f"{prefix}.{kind}.map").write_text(header + "".join(f"{v:.12g}\n" for v in values.reshape(-1).tolist()))
            native = vina.Vina(sf_name="ad4", cpu=1, verbosity=0)
            native.set_ligand_from_string(text)
            native.load_maps(str(prefix))
            exact = native._vina.score()
            scorer = AD4Scorer.from_autogrid(prefix, types, dtype=torch.float64, map_mode="vina")
            features = prepared.features()
            features["ad4_charges"] = torch.tensor(prepared.charges, dtype=torch.float64)
            own = scorer.prepare(features, torch.zeros(1, 3), {}, intramolecular_mask=prepared.intramolecular_mask())
            coords = prepared.coordinates(dtype=torch.float64).requires_grad_(True)
            result = own.module.components(coords)
            gradient, = torch.autograd.grad(result.search_energy.sum(), coords)
            rows.append(dict(smiles=smiles, atoms=len(prepared.ad4_types), pairs=len(prepared.intramolecular_pairs),
                             torsdof=prepared.torsdof, native_inter=exact[1], own_inter=result.intermolecular.item(),
                             native_intra=exact[5], own_intra=result.intramolecular.item(), native_score=exact[0],
                             own_score=result.score.item(), finite_gradient=bool(torch.isfinite(gradient).all()),
                             native_fast_table_probe=_native_fast_table_probe(prepared),
                             map_source_sha256=scorer.map_source_sha256, prepared_fingerprint=prepared.fingerprint))
    result = dict(scope="prepared flexible ligands, synthetic affine AutoGrid maps, native Vina AD4 CPU; no native ADGPU or real receptor maps",
                seed=713, versions=dict(rdkit=rdBase.rdkitVersion, meeko=meeko.__version__, vina=vina.__version__),
                max_inter_abs_error=max(abs(r["native_inter"] - r["own_inter"]) for r in rows),
                max_intra_abs_error=max(abs(r["native_intra"] - r["own_intra"]) for r in rows),
                max_score_abs_error=max(abs(r["native_score"] - r["own_score"]) for r in rows),
                max_native_fast_table_probe_abs_error=max(abs(r["native_fast_table_probe"] - r["native_intra"]) for r in rows),
                rows=rows)
    if any(result[key] > 1e-8 for key in ("max_inter_abs_error", "max_score_abs_error", "max_native_fast_table_probe_abs_error")):
        raise AssertionError("native AD4 component parity changed; inspect the audit rows and dependency versions")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    result = audit()
    payload = json.dumps(result, indent=2) + "\n"
    if options.output:
        options.output.write_text(payload)
    print(payload)
