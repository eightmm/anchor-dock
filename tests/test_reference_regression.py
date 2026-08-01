"""Numerical regression guard for reference mode.

The expected values were recorded from the reference pipeline before it moved out
of the retired ``lig_align`` namespace, and re-verified byte-identical after.
They exist so a future refactor of conformer generation, ligand loading, MCS
selection, or pose export cannot silently shift docking output.

Scores carry a relative tolerance because torch reduces in float32, so the last
digits may differ on other hardware. Pose counts, MCS sizes and ranking order are
asserted exactly: those are discrete and must not drift at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem

from anchor_dock.reference import run_pipeline

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "10gs"
PROTEIN = str(EXAMPLES / "10gs_pocket.pdb")
REFERENCE = str(EXAMPLES / "10gs_ligand.sdf")

PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
IBUPROFEN = "CC(C(=O)O)c1ccc(cc1)CC(C)C"

# case id -> (pipeline kwargs, expected best score, expected pose count, expected MCS size)
CASES = {
    "single_vina": (dict(query_ligand=PARACETAMOL, mcs_mode="single"), -6.479089, 3, 6),
    "single_vinardo": (dict(query_ligand=PARACETAMOL, mcs_mode="single", weight_preset="vinardo"),
                       -1.524636, 3, 6),
    "multi": (dict(query_ligand=IBUPROFEN, mcs_mode="multi"), -3.681152, 5, 10),
}


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_reference_output_is_stable(case_id: str, tmp_path: Path) -> None:
    kwargs, expected_score, expected_poses, expected_mcs = CASES[case_id]
    result = run_pipeline(
        protein_pdb=PROTEIN,
        ref_ligand=REFERENCE,
        output_dir=str(tmp_path),
        num_confs=30,
        rmsd_threshold=1.0,
        optimize=False,
        device="cpu",
        verbose=False,
        **kwargs,
    )

    assert result["mcs_size"] == expected_mcs
    assert result["num_representatives"] == expected_poses
    assert result["best_score"] == pytest.approx(expected_score, rel=1e-4)

    poses = [m for m in Chem.SDMolSupplier(result["output_file"], removeHs=False) if m is not None]
    assert len(poses) == expected_poses

    scores = [float(m.GetProp("Vina_Score")) for m in poses]
    assert scores == sorted(scores), "poses must be written best-energy-first"
    assert scores[0] == pytest.approx(expected_score, rel=1e-3)

    ranks = [int(m.GetProp("Rank")) for m in poses]
    assert ranks == list(range(1, expected_poses + 1))
