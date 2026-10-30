# Prepared inputs and map scoring

High-level reference, covalent and interaction pipelines retain inferred preparation. The lower-level `DockingEngine` also accepts an authoritative prepared ligand and rigid receptor, preserving supplied chemistry without choosing protonation or tautomer states.

## Explicit graph and mapping

PDBQT supplies coordinates, AD4 types, charges and a torsion tree, but not a complete chemical graph. Supply an RDKit graph with intended bonds, stereochemistry and explicit polar H, plus an explicit PDBQT-serial-to-graph-index mapping. A serial is not an atom index. Meeko `REMARK INDEX MAP` can supply this correspondence; coordinate-nearest-neighbor guesses are inappropriate.

```python
from pathlib import Path
from rdkit import Chem
from anchor_dock import PreparedLigand, receptor_context_from_prepared_pdbqt

graph = Chem.SDMolSupplier("prepared_ligand.sdf", removeHs=False)[0]
prepared = PreparedLigand.from_pdbqt(
    graph, Path("ligand.pdbqt").read_text(),
    serial_to_atom_index=ligand_serial_to_graph_index,
)
receptor = receptor_context_from_prepared_pdbqt(
    receptor_graph_with_polar_h, Path("receptor.pdbqt").read_text(),
    serial_to_atom_index=receptor_serial_to_graph_index, device="cpu",
)
```

The importer checks elements, coordinates, unique mapping, tree closure and graph agreement. Omitted atoms may only be nonpolar C-bonded H. Flexible receptor records, macrocycle closure/dummy types and unsupported types fail explicitly. `prepared.mol` and `prepared.coordinates()` share the selected original graph order. `original_atom_indices` and `pdbqt_serials` map results back. Graph, stereochemistry, charges, types, coordinates, mapping and branches enter the fingerprint.

`PreparedLigand.from_arrays(...)` accepts already aligned data. Its empty default branch list declares a rigid ligand. Prepared branches drive both intra mobility and kinematics. AD4 `torsdof` and endpoint-derived `vina_num_tors` are separate; the latter can be half-integral.

## Continuous refinement and final scoring

```python
import torch
from anchor_dock import DockingEngine, PairwiseScorer

engine = DockingEngine(
    PairwiseScorer("vina", positive_energy_cap=1000.0),
    device="cpu", optimizer="lbfgs", num_steps=100,
)
problem = engine.prepare(prepared.mol, receptor, prepared_ligand=prepared)
xyz = prepared.coordinates("cpu")[None]
guided, final, guide_stats, release_stats = engine.optimize_se3(
    problem, xyz, pivot_atom_index=0,
    centers=xyz[:, 0], rotation_vectors=torch.zeros(1, 3),
)
initial_score, final_score = engine.report_scores(problem, xyz, final)
```

Choose a heavy pivot. Coordinates are angstroms; angles are radians. Optional positive-energy curl follows Vina's per-ligand-atom receptor sum and per-intra-pair convention. The uncapped default preserves the prior continuous objective and reduction order. Scores are lower-is-better with `kcal/mol-like` units.

The explicit `native_vina_score(problem.scorer, final.detach(), intramolecular_reference=...)` API uses the ordinary native FAST radial-table convention and cap for final scoring. It is nondifferentiable and belongs after optimization. Prepared graph, mobility and rotor conventions must match native preparation. See the [48-case audit](../reports/native-scoring-parity-2026-10-30.md) for precision and limits.

## AutoGrid / AD4 reference backend

```python
from anchor_dock import AD4Scorer, DockingEngine

scorer = AD4Scorer.from_autogrid(
    "maps/receptor", sorted(set(prepared.ad4_types)),
    device="cpu", map_mode="raw", formulation="adgpu",
    unbound_model="same_as_bound",
)
engine = DockingEngine(scorer, device="cpu", optimizer="cartesian", learning_rate=1.0)
problem = engine.prepare(prepared.mol, receptor, prepared_ligand=prepared)
```

The loader reads `prefix.TYPE.map`, `prefix.e.map` and `prefix.d.map`. It validates headers, finite samples, counts, common receptor label and geometry. `NELEMENTS` counts intervals; nodes are one larger. Samples vary x fastest, tensors use `[channel,z,y,x]`, and origin is `center - nelements*spacing/2`.

The backend includes intermolecular maps, prepared intra contributors, additive torsion and explicit unbound reporting. `same_as_bound` reports `inter + torsion`; `fixed` requires `unbound_energy` and reports `inter + intra + torsion - unbound`. Search minimizes `inter + intra`. Maps already contain weights: `AD4Weights` controls analytic intra/torsion terms without reweighting map samples.

`map_mode="raw"` uses raw samples and upper-exclusive bounds. `"vina"` applies native positive map-sample and intra-pair curl conventions with closed bounds. `formulation="standard"` uses analytic AD4 dielectric/Gaussian terms; `"adgpu"` uses inspected GPU forward approximations with exact differentiation of that forward. These choices reflect differences between native programs.

**Maps are the authoritative receptor field.** Their contents, geometry and score conventions are fingerprinted; source file hashes are exposed. The engine receptor graph is not physically checked against the map-generating receptor. Supply the matching receptor and retain its preparation record. Per-receptor-atom exclusions, covalent corrections and flexible receptors are unsupported. Analytic intra terms still differ from native radial tables. Native AutoDock-GPU end-to-end agreement and GPU throughput remain unmeasured.

## Geometry-aware optimizer

`optimizer="cartesian"` is opt-in across Python and CLI (`--optimizer cartesian --opt-lr 1`). It uses a damped Cartesian pullback metric, bounds actual atom displacement (default 0.25 Å/step) and checks the actual objective's Armijo condition. Grid-domain failures reject trials; invalid initial poses error. The existing raw rotation-vector chart is retained.

The dense reference is bounded to 256 parameters/pose and two million Jacobian elements. Engine entry points process it one pose at a time. Adam remains the default; geometric scaling does not establish faster or more accurate docking. Low-level `optimize_pose_module` exposes `geometry_damping`, `max_atom_displacement` and `max_backtracks`. CUDA correctness and throughput require a CUDA-enabled run.
