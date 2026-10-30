"""The public prepared/map/engine APIs must compose into an actual refinement."""

import torch
from rdkit import Chem

from anchor_dock import AD4Scorer, DockingEngine, PreparedLigand
from anchor_dock.core.autogrid import AutoGridMap, AutoGridMapSet
from anchor_dock.core.io import receptor_context_from_mol


def test_prepared_ad4_cartesian_guide_release_and_reporting():
    graph = Chem.MolFromSmiles("CCCC")
    xyz = torch.tensor([[-1., 1., 0.], [0., 1., 0.], [1., 1.5, 0.], [2., 1.5, 1.]])
    prepared = PreparedLigand.from_arrays(
        graph, ["C"] * 4, [0.] * 4, coordinates=xyz, rotatable_bonds=[(1, 2)], torsdof=1,
    )
    # An affine synthetic receptor field makes movement and reporting exact.
    values = torch.arange(-4., 5.).reshape(1, 1, 9).expand(9, 9, 9).clone()

    def field(v):
        return AutoGridMap(v, 1., (8, 8, 8), (0., 0., 0.), "0" * 64, "synthetic", "synthetic.pdbqt")

    maps = AutoGridMapSet(("C",), (field(values),), field(torch.zeros_like(values)), field(torch.zeros_like(values)))
    receptor_graph = Chem.MolFromSmiles("C")
    receptor_graph.AddConformer(Chem.Conformer(1))
    receptor = receptor_context_from_mol(receptor_graph, "cpu")
    engine = DockingEngine(AD4Scorer(maps), optimizer="cartesian", learning_rate=1., num_steps=3, device="cpu")
    problem = engine.prepare(prepared.mol, receptor, prepared_ligand=prepared)
    guided, final, guide_stats, release_stats = engine.optimize_se3(
        problem, xyz[None], 1, centers=xyz[None, 1], rotation_vectors=torch.zeros(1, 3), release_steps=2,
    )
    assert guided.shape == final.shape == (1, 4, 3)
    assert guide_stats.num_poses == release_stats.num_poses == 1
    initial_report, final_report = engine.report_scores(problem, xyz[None], final)
    assert final_report.search_energy < initial_report.search_energy
    torch.testing.assert_close(final_report.score, final_report.intermolecular + .2983)
    torch.testing.assert_close(
        torch.linalg.vector_norm(final[:, 1:] - final[:, :-1], dim=-1),
        torch.linalg.vector_norm(xyz[None, 1:] - xyz[None, :-1], dim=-1),
    )
    assert problem.prepared_input_fingerprint == prepared.fingerprint
