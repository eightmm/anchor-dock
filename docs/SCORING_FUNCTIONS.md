# Torch scoring functions and existing-pose refinement (0.6.2)

AnchorDock focuses on scoring functions that expose a Torch objective for pose optimization: Vina/Vinardo empirical scores, AD4/AutoDock-GPU map references and an all-atom AMBER vacuum potential. SoftDock, DKOES presets and public Smina/GNINA/native Vina optimizer adapters have been removed. Unknown or removed names fail explicitly. Interaction docking defaults to Vina.

## Current catalog

| Name | Objective and units | Preparation / limits |
|---|---|---|
| `vina`, `vinardo` | Continuous empirical score, `kcal/mol-like` | Inferred XS or authoritative prepared inputs |
| `ad4` | AD4 map/pair reference, `kcal/mol-like` | Existing AutoGrid maps and prepared ligand chemistry |
| `adgpu` | Explicit AutoDock-GPU map/pair reference formulation, `kcal/mol-like` | Existing AutoGrid maps and prepared ligand chemistry |
| `amber` | Pure Torch full-complex AMBER potential, `kcal/mol` | Complete prmtop/inpcrd, explicit all-atom ligand indices; nonperiodic vacuum NoCutoff |
| `vina-final`, `vinardo-final` | Nondifferentiable native FAST-table reference, `kcal/mol-like` | Authoritative prepared input; pattern optimization only |

These are seven named entries with different preparation and reporting contracts. `adgpu` is a scoring reference, not a native AutoDock-GPU search implementation. `vina-final` and `vinardo-final` evaluate native-table conventions without invoking a native optimizer. Physical potential energies and empirical docking scores have different meanings; no cross-function raw ranking is provided.

Continuous Torch objectives support Adam, AdamW, LBFGS, Cartesian pullback and derivative-free pattern optimization. The catalog declares each function's capabilities and required constructor options. It is used by `refine_pose` and the prepared engine; the three guided docking CLI commands expose inferred Vina/Vinardo scoring. A catalog entry does not automatically supply chemistry, maps, topology or compatible covalent exclusions. Optional dependencies, external maps and topology files remain explicit requirements.

## Existing-pose refinement

```bash
uv sync --frozen --group dev
uv run --frozen anchor-dock scorers
uv run --frozen anchor-dock refine -p protein.pdb -l positioned_pose.sdf \
  --scorers vina vinardo --steps 100 --device cpu -o sf_comparison
```

```python
from anchor_dock import refine_pose, refine_with_scorers

one = refine_pose(
    "protein.pdb", "positioned_pose.sdf",
    scorer="vina", optimizer="lbfgs", num_steps=100,
)
comparison = refine_with_scorers(
    "protein.pdb", "positioned_pose.sdf", ["vina", "vinardo"],
    num_steps=100, on_error="record",
)
```

Every function starts independently from the same source pose. Prepared routes may add polar H, merge nonpolar H and round PDBQT coordinates/charges; preparation policies and output-to-input atom maps are recorded. With `output_dir` supplied, each successful function writes `refined.sdf` and `result.json`; a sweep adds `comparison.json` with successes and explicit errors. Without `output_dir`, results remain in memory. Existing result files are not overwritten. Reported metrics declare units and direction, while search objectives are always lower-is-better. Raw scores are never averaged across functions.

Refinement requires an existing positioned 3D ligand. Missing/2D coordinate generation is available separately through `load_ligand_for_refinement` and `prepare_ligand`. Callers must place those conformers in the receptor coordinate frame before refinement; local optimization does not perform that placement.

## Prepared chemistry and map scoring

Install `anchor-dock[prepare]` for automatic Meeko preparation. `preparation="auto"` follows the scorer's profile, while `preparation="meeko"` explicitly requests prepared input. Whole-receptor PDB/PQR preparation precedes pocket selection. It follows declared residue states/templates, supplied H and explicit alternate-location policies; it does not estimate pKa or choose tautomers. Unknown residues, ambiguous HIS and incomplete termini fail until their state or selection is declared.

```python
result = refine_pose(
    "protein.pqr", "positioned_pose.sdf", scorer="vina",
    preparation="meeko",
    preparation_options={
        "receptor": {"set_template": {"A:1": "NGLY", "A:2": "CGLY"}},
    },
)
```

The template IDs above are illustrative and must describe the actual receptor. PQR preparation preserves supplied charges, including merged nonpolar-H charge. PB radii are not substituted for XS or Lennard-Jones radii. Vina/Vinardo do not add a Coulomb term merely because charges are present. Cut pockets with undeclared ends can fail preparation; use a correctly prepared complete receptor.

For `ad4` or `adgpu`, pass `scorer_options={"maps_prefix": "maps/receptor", "atom_types": ["C", "OA", "HD"]}` with the actual required map types. The maps already encode receptor electrostatics and other intermolecular terms: changing receptor charges requires regenerated maps. Prepared ligand charges, atom types, mobility pairs and TORSDOF determine the ligand-side terms. `ReceptorPreparation.autogrid_gpf()` generates a GPF; AutoGrid execution remains external. Maps are authoritative, and selecting a function does not generate them.

Prepared objects can be reused across functions to avoid repeating Meeko preparation:

```python
from anchor_dock import prepare_ligand, prepare_receptor, refine_with_scorers

ligand = prepare_ligand("positioned_pose.sdf", generate_coordinates_if_missing=False)
receptor = prepare_receptor("protein.pdb")  # supply actual residue/template choices if required
results = refine_with_scorers(
    receptor, ligand, ["vina", "vinardo"], num_steps=100, device="cpu",
)
```

This reuses prepared chemistry in memory. It does not create a persistent target-artifact cache or generate AutoGrid maps. To retain prepared files across processes, keep their source graph, PDBQT serial mapping and preparation metadata together; PDBQT alone omits chemical bond orders. See [prepared input contracts](PREPARED.md).

AD4 and AutoDock-GPU conventions differ in their declared pair formulation. The map reference does not establish native search parity, native update-direction parity or global docking performance. See [prepared inputs and scoring conventions](PREPARED.md).

## AMBER in pure Torch

```bash
uv sync --frozen --extra amber --group dev
```

```python
result = refine_pose(
    None, "all_atom_ligand_pose.sdf", scorer="amber",
    scorer_options={
        "prmtop": "complex.prmtop", "inpcrd": "complex.inpcrd",
        "ligand_atom_indices": [100, 101, 102],  # replace with every ligand atom
    },
    optimizer="lbfgs", num_steps=100,
)
```

The named `amber` entry constructs `TorchAmberScorer`. OpenMM reads the explicit AMBER topology/coordinates once and supplies the parameterized System for import. Runtime bond, angle, periodic torsion, Lennard-Jones, Coulomb and exception energies and their gradients are evaluated in Torch; scoring does not call an OpenMM Context. The original `NeuralScorerAdapter` remains a separate custom-model extension.

`TorchAmberScorer.from_system()` also imports an explicitly parameterized OpenMM System and reference positions. Import accepts `device`, `dtype=torch.float64`, `pair_chunk_size=128` and `pair_cache_max_pairs=200000`. Compute uses FP64 and accepts FP32 or FP64 coordinate inputs. `.to(device, dtype=torch.float64)` moves imported state before preparation. `pair_chunk_size` is an atom-tile width, not a pair count. If the candidate count `L*R + L*(L-1)/2` is within `pair_cache_max_pairs`, moving-pair indices and exception-adjusted coefficients are prepared once; set the limit to `0` to force tiled evaluation. Cached evaluation is also tiled, and fixed–fixed energy is prepared once. Cache policy is recorded with the scorer state. This backend supports nonperiodic vacuum NoCutoff systems. GB/implicit solvent, periodic boundaries, PME, virtual sites, parameter offsets and unsupported force terms fail explicitly.

Zero-strength Lennard-Jones contributions are disabled before distance powers are evaluated, including Coulomb-only exception pairs. Genuine singular active geometry still raises a physical-domain error.

A PDBQT or PQR charge array alone cannot replace bonded parameters, atom mapping, exclusions and pair exceptions from the complete topology. Ligand parameters and protein parameters must belong to the same declared force-field setup. The combined topology contains the fixed receptor. If a separate protein is supplied, its coordinates/order must match. The zero-based `ligand_atom_indices` list is ordered like the supplied ligand graph and must include every mapped atom, including H. Where topology connectivity is supplied/imported, induced ligand edges and H-parent boundaries are checked; otherwise `from_system` can check only the declared count/elements and explicit map. Use the topology's ordering and chemical state; do not assume an unrelated SDF has the same atom order.

Reported `score` and named `complex_potential` represent a full-complex potential, not binding affinity or an MM/GBSA binding-energy subtraction. Search uses a fixed-reference energy difference so a large receptor constant does not conceal small pose improvements. Public physical-energy metrics do not relabel placeholder inter/intra decompositions as measured components.

## OpenMM reference and advanced backend

`OpenMMScorer` remains available as an explicit Python object for numerical comparison and its supported advanced options. It is distinct from the catalog's Torch `amber` entry: OpenMM Context energies and forces are connected to Torch by a first-derivative bridge. Its supported implicit-solvent models include GBn2; those settings are not transferred to `TorchAmberScorer`.

```python
from anchor_dock import OpenMMScorer, refine_pose

with OpenMMScorer.from_amber(
    "complex.prmtop", "complex.inpcrd", ligand_atom_indices,
    implicit_solvent="GBn2",
) as reference:
    result = refine_pose(None, all_atom_ligand_pose, scorer=reference)
```

The same full-topology, all-atom mapping and full-potential interpretation apply. The OpenMM bridge supports exact first derivatives of its configured potential; virtual sites and second derivatives are unsupported. The optional `native-vina` dependency group is retained for direct Vina scoring oracle checks, not public native optimization services.

## Optimization and custom extensions

SE(3)+torsion optimization preserves the declared rigid-frame geometry. Explicit H must not alter inferred heavy-graph rotor detection; authoritative prepared torsion branches are retained separately. The optional Cartesian optimizer uses a damped coordinate Jacobian pullback and objective-checked steps. Trials with invalid physical geometry are backtracked; an invalid initial pose still raises an error. It is a bounded reference method, not an energy Hessian solver or a global-search claim.

Pattern search polls parameter-coordinate directions, checks actual atom displacement and accepts objective decreases. It has evaluation, DOF and minimum-step limits and can be expensive. `optimizer_options` exposes the declared pattern budgets and Cartesian/backtracking settings. An explicitly selected AdamW keeps pose weight decay at zero; an omitted refinement optimizer resolves to Adam for differentiable functions and pattern for table references. `learning_rate` defaults to 0.05 for Adam/AdamW, 0.5 for LBFGS, 1.0 for Cartesian, and 0.1 for pattern in `refine_pose`; guided docking uses its separate `opt_lr` default. Zero steps evaluates the prepared starting pose without optimizer steps; selected preparation policies still apply.

Vina/Vinardo support `PairwiseScorer("vina", intra_mode="sparse")`, or `scorer_options={"intra_mode": "sparse"}` with `refine_pose`. This opt-in mode calculates only contributing intramolecular pairs. The default dense calculation contract is retained: full-scoring cost can be dominated by intermolecular work, and FP32 direct distances can differ numerically from dense `cdist` evaluation. Intramolecular-only CPU gains do not establish an end-to-end speedup.

Vina/Vinardo factories copy supplied configuration rather than sharing caller-owned weights. Each `prepare()` snapshots configuration, receptor coordinates, ligand/receptor features and pair masks. Later caller-data or factory-weight changes affect a new preparation, not an existing prepared scorer. Explicit factory weight changes are reflected in its fingerprint, including reduction order. The prepared object's public fields are still mutable; edit inputs and prepare again when creating a new reproducible state.

AD4's `TrilinearGrid.sample_channels` shares cell lookup and eight-corner weights across its three map channels. Validation covers bitwise-equal forward values, small-ULP backward differences and the native AD4 golden reference. The [current CPU validation report](../reports/cpu-validation-2026-10-30.md) records numerical checks and executed workflows. The [historical 0.6.1 physical-scoring snapshot](../reports/torch-physical-2026-10-30.md) retains its CPU workload timings; GPU performance remains unverified.

```python
from anchor_dock import (
    ScoringFunctionInfo, register_scoring_function, NeuralScorerAdapter,
)

register_scoring_function(
    ScoringFunctionInfo(
        "my-model", "learned", "torch", ("autograd", "pattern"),
        "inferred", "arbitrary",
    ),
    lambda: NeuralScorerAdapter(my_model, name="my-model", units="arbitrary"),
)
```

A factory supplies `prepare()` and the `PreparedScorer` contract, including a deterministic fingerprint. Search energy is lower-is-better; a reported metric can declare a different direction. Installed packages may expose the `anchor_dock.scorers` entry-point group returning `ScoringFunctionRegistration`; `load_scoring_plugins()` loads them explicitly. Custom model parameters, preparation and units remain the adapter author's responsibility.

Standalone results use `anchor-dock-sf-refinement-v1` and `anchor-dock-sf-comparison-v1`, separate from ranked docking SDF schema `4`. Version 0.6.2 / batch epoch `8` invalidates prior resume artifacts. Dated 0.6 reports describe a historical catalog that included now-removed adapters.

## Validation scope

Validation covers source equations, finite-difference/autograd derivatives, optimizer descent, atom identity and independent starting poses. Optional tests compare supported Torch AMBER energies/gradients with an independent OpenMM reference and exercise actual preparation when dependencies are installed. Native Vina remains a scoring oracle for the relevant reference checks. Test counts and numerical tolerances belong to their recorded snapshots. Use the [current CPU validation report](../reports/cpu-validation-2026-10-30.md) for this release; historical reports are not current benchmark claims.

GPU throughput, broad protein-ligand docking accuracy, binding free-energy prediction and native AutoDock-GPU search parity require separate evidence. No such performance or accuracy claim follows from sharing a scoring formula.
