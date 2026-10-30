# Scoring functions and existing-pose refinement (0.6)

SoftDock has been removed. Unrecognized/removed names fail explicitly; there is no silent substitution. Interaction docking now defaults to Vina. Existing reference/covalent/interaction strategies remain available, and the new refinement API preserves an already positioned 3D pose.

## Implemented routes

| Names | Optimization | Preparation / boundaries |
|---|---|---|
| vina, vinardo | Torch Adam/AdamW/LBFGS/Cartesian or pattern | Inferred XS or supplied prepared inputs |
| dkoes_fast, dkoes_scoring_old | Same Torch optimizers | Independent continuous implementations of pinned empirical equations |
| dkoes_scoring | Same Torch optimizers | Prepared AD4 types/charges for both molecules; includes desolvation |
| ad4 | Torch optimizers | Explicit AutoGrid maps + prepared input; ordinary rigid receptor reference |
| vina-final, vinardo-final | Pattern search | Nondifferentiable native FAST-table final energies; prepared input |
| amber | OpenMM force-to-Torch gradient or pattern | Combined prmtop/inpcrd and explicit all-atom ligand indices |
| vina-native:vina / vinardo / ad4 | Native Vina Python local optimization | Optional vina + Meeko; AD4 also needs maps |
| smina:SF, gnina:SF | Native CLI minimization | Installed executable + prepared input |
| gnina:cnn | Native CNN refinement | Installed GNINA/model; distinct empirical and CNN reporting |

Native empirical SF names are vina, vinardo, dkoes_fast, dkoes_scoring, dkoes_scoring_old and ad4_scoring. These are backend-qualified entries, not 25 independent physical models. GNINA/Smina Vinardo parameters can differ from the modern Vina lineage. ad4_scoring is not an alias for the AutoDock4 map backend.

The catalog reports dependencies, preparation and required constructor options. An implemented adapter can still require an absent executable, external maps, topology or model. Proprietary Glide/GOLD/PLANTS scores are not reimplemented or claimed equivalent; a separately supplied licensed adapter can use the plugin interface.

## Usage

```bash
uv sync --frozen --all-extras --group dev
anchor-dock scorers
anchor-dock refine -p protein.pdb -l positioned_pose.sdf \
  --scorers vina vinardo dkoes_fast dkoes_scoring_old \
  --steps 100 -o sf_comparison
```

The core install works for the continuous Vina/Vinardo/DKOES fast/old routes. Extras are prepare (Meeko), native-vina and amber (OpenMM). Smina/GNINA binaries are installed separately; failures are recorded rather than replaced by a different scorer.

```python
from anchor_dock import refine_pose, refine_with_scorers

one = refine_pose(
    "protein.pdb", "positioned_pose.sdf",
    scorer="dkoes_fast", optimizer="lbfgs", num_steps=100,
)
comparison = refine_with_scorers(
    "protein.pdb", "positioned_pose.sdf",
    ["vina", "vinardo", "dkoes_fast", "dkoes_scoring_old"],
    num_steps=100, on_error="record",
)
```

Every SF starts independently from the same source input. Prepared routes may add polar H, merge nonpolar H and round PDBQT coordinates/charges; those changes and output-to-input atom maps are recorded. Raw scores are never averaged or globally ranked across functions. Each run has its own refined.sdf/result.json; comparison.json retains successes and explicit errors.

Refinement requires an existing positioned 3D ligand. SMILES/2D conformer generation is available separately through load_ligand_for_refinement/prepare_ligand, but generated conformers must be placed through a reference or interaction docking strategy first. Refining a conformer near the origin against an unrelated distant receptor is not silently accepted as a placement algorithm.

## Prepared chemistry

preparation=auto uses the scorer's profile; preparation=meeko explicitly requests prepared input. PDB/PQR receptor preparation runs on the full supplied structure before any pocket selection. It retains declared chemical states and explicit H; it does not estimate pKa/tautomers. Unknown residues, ambiguous HIS, altlocs and incomplete termini require explicit template/selection options rather than state guesses.

```python
result = refine_pose(
    "protein.pqr", "positioned_pose.sdf",
    scorer="dkoes_scoring",
    preparation_options={"receptor": {"set_template": {"A:1": "NGLY", "A:2": "CGLY"}}},
)
```

PQR preserves supplied charges, including merged nonpolar-H charge, instead of silently recalculating them. PB radii are not reused as XS or LJ radii. PDBQT quantization is recorded. The example template IDs are illustrative and must describe the actual receptor. Cut pockets with undeclared chain ends can fail preparation; use a correctly prepared complete receptor. ReceptorPreparation.autogrid_gpf generates a GPF but does not execute AutoGrid.

For ad4 pass scorer_options containing maps_prefix and atom_types. Maps already encode receptor electrostatics: changing receptor charges requires regenerated maps, not just a context update. Native Vina AD4 similarly requires maps_prefix.

## AMBER physical energy

```python
result = refine_pose(
    None, "all_atom_ligand_pose.sdf", scorer="amber",
    scorer_options={
        "prmtop": "complex.prmtop", "inpcrd": "complex.inpcrd",
        "ligand_atom_indices": [100, 101, 102],  # replace with every real ligand atom
        "implicit_solvent": "GBn2",
    },
    optimizer="lbfgs",
)
```

The topology contains the fixed receptor. If a separate protein is supplied, its coordinates/order must match. All ligand atoms, including H, must match explicit particle indices. This is a full-complex potential, not binding affinity or an MM/GBSA binding-energy subtraction. Native OpenMM forces provide the exact first derivative; virtual sites and second derivatives are unsupported.

Search subtracts a fixed reference energy in FP64 before casting to Torch, preserving small improvements in the presence of a large receptor constant. Reported score and named complex_potential remain full energy. Protocol inter/intra placeholders are omitted from public result metrics.

## Native and derivative-free optimization

Native routes use their engine's local optimizer, not AnchorDock's interaction-guide/covalent contracts. GNINA CNN refinement optimizes native pose loss plus internal/bounds terms; CNNscore, CNNaffinity and empirical affinity remain separate reported quantities with their own units/directions. The empirical affinity is not relabeled as CNN loss.

CLI padding follows the native local box; hydrogen-only branches are retained, initial/final force caps match, AD4 maps enter fingerprints, and GNINA GPU selection respects CUDA_VISIBLE_DEVICES. Zero steps is real score-only. Native Vina seed 0 is mapped to effective seed 1 and recorded because the underlying zero means random.

Pattern search is a bounded deterministic reference for detached/discontinuous scores. It polls parameter-coordinate directions, checks actual atom displacement and accepts only objective decreases. It has evaluation/DOF/minimum-step limits and is expensive; it does not imply native search parity or global search. optimizer_options expose pattern budgets and Cartesian/backtracking settings. Native cpu/timeout/padding use native_options.

## Plugins and reporting

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

A factory supplies prepare() and the PreparedScorer contract, or refine() for a native route. Search energy is always lower-is-better; reporting may declare higher-is-better separately. A deterministic fingerprint is required. Native engines, checkpoints and parameters remain independent dependencies. Installed packages can expose the anchor_dock.scorers entry-point group returning ScoringFunctionRegistration; load_scoring_plugins() loads them explicitly.

The new files use anchor-dock-sf-refinement-v1 / anchor-dock-sf-comparison-v1 result schemas, separate from ranked docking SDF schema 4. Version 0.6 and batch epoch 6 invalidate old default-SoftDock resumes.

## Validation scope

Tests execute Vina Python, OpenMM and Meeko when their extras are installed. Smina/GNINA CLI contract tests use explicit stubs because those binaries are unavailable here. DKOES equations are independently checked against scalar references and derivatives, not claimed native-executable parity. GPU throughput, actual CNN execution and broad docking accuracy remain unmeasured. Exact native PDBQT bond perception can differ from a supplied chemical graph.

DKOES/native source reference: [GNINA pinned implementation](https://github.com/gnina/gnina/tree/01cae6aa6e014e0bf11e511f1fe7a61b17975855). No native licensed source is vendored.
