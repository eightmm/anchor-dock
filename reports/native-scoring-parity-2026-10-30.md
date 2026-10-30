# Prepared full-ligand Vina/Vinardo scoring audit

This audit checks complete ligand scoring against the **native Vina 1.2.7 executable binding**, beyond the earlier one-pair checks. It does not measure docking accuracy.

## Inputs and alignment

- Seven seeded ETKDGv3 molecules cover ethanol, a secondary amine, an amide, pyridine, acetate, nitro and a branched flexible ligand. The bundled 10gs ligand supplies a larger eighth case.
- Three rigid placements per ligand and both Vina/Vinardo give **48 comparisons**. The receptor is one Meeko-prepared glycine held rigid as a controlled multi-atom field; it is not the biological 10gs receptor.
- Meeko 0.8.0 prepares explicit-H graphs. Its explicit `REMARK INDEX MAP` is converted to a serial-to-original-index dictionary. Both scorers consume the same PDBQT-rounded coordinates, prepared types and charges. Nonpolar H omission, polar H, branch mobility and intramolecular exclusions pass through `PreparedLigand`.
- The native call is the unrounded, private `Vina._vina.score(0.0)` binding, pinned to Vina 1.2.7. An explicit zero intramolecular reference isolates the total-score formula; native `score()` would instead use the current pose's intramolecular reference. Final native rigid-receptor scoring uses explicit atoms (`no_refine=False`); search maps provide its box-validity check.
- Vina's half-integral prepared endpoint rotor count is preserved: the 10gs ligand has **13.5 Vina num_tors versus TORSDOF 14**, and ethanol has 0.5 versus 1. These quantities must not be truncated or conflated.

## Results

Maximum absolute differences over the 48 cases:

| AnchorDock evaluation | Intermolecular | Intramolecular | Total, zero reference |
|---|---:|---:|---:|
| Continuous analytic equations | 0.584565 | 0.00265811 | 0.461546 |
| Analytic + native positive-energy cap | 0.165348 | 0.00262028 | 0.144276 |
| Explicit score-only native FAST convention + cap | **4.09829e-6** | **3.24820e-7** | **2.91349e-6** |

The last row emulates native final-score lookup: index `floor(32*r*r)`, midpoint energy from the two adjacent radial nodes. Together with the native cap (per ligand-atom receptor sum, per intramolecular pair), prepared mobility and half-integral rotor count, this explains the observed differences down to feature precision in these fixtures.

The discrete FAST table is available through an **explicit score-only API**, not a differentiable optimizer objective. Its piecewise-constant final energies and native smooth derivative tables should not be confused with exact differentiation of a continuous function. AnchorDock retains the continuous objective and exposes the positive-energy cap explicitly. Tight agreement of the score-only convention does not imply that default analytic scores are numerically identical to native final scores.

## Score-only API

`anchor_dock.core.native_vina.native_vina_components(scorer, coordinates)` returns final inter/intra/search energies; `native_vina_score` also applies the prepared half-integral rotor denominator and a declared intramolecular reference. The contract identifier is `vina-1.2.7-fast-r2-factor32-curl1000-prepared-v1`.

```python
from anchor_dock.core.native_vina import native_vina_score

# scorer must have authoritative PreparedLigand/receptor inputs, the prepared
# intramolecular mask and prepared vina_num_tors. Its optimization objective
# remains the continuous analytic function.
final = native_vina_score(
    scorer, optimized_coordinates.detach(), intramolecular_reference=0.0,
)
```

The API requires explicit detachment and refuses inferred types, metal/special types, altered official force-field features, custom intermolecular exclusions and inconsistent prepared masks/rotor counts. It always applies the authentic cap 1000 independently of a chosen continuous search cap. `intramolecular_reference=None` uses one common reference from the best final search pose in the supplied batch; a finite scalar supplies a reference from another selected pose. Native standalone `Vina.score()` instead uses each current pose's own reference. No native library is required at runtime. Validation and an unfused pair calculation are appropriate for final scoring; GPU throughput is unmeasured.

## Reproduction and regression

```bash
uv pip install meeko==0.8.0 vina==1.2.7 gemmi scipy
uv run --no-sync python examples/audit_native_scoring.py \
  --seed 713 \
  --output reports/native-scoring-parity-2026-10-30.json \
  --fixture-output tests/fixtures/native_scoring.json
uv run --frozen pytest -q tests/test_native_scoring_contract.py
```

The fixture stores full chemical graphs, PDBQT text, explicit index mappings and native expected energies for three ligands (including the 10gs ligand) and both scorers. The **24 regression/API tests** run without Meeko/Vina installed: they check the continuous-model error envelope, tightly matched public score-only energies, invalid contracts, explicit detachment, derived-cache mutation and batch-reference semantics. Source versions, seed, ligand hash and all 48 component results are in the [JSON evidence](native-scoring-parity-2026-10-30.json).

Source review used the pinned [Vina precalculate implementation](https://github.com/forlilab/AutoDock-Vina/blob/bb64e5e26b520569a86360d69248069b5e665b3f/src/lib/precalculate.h), [model pair construction](https://github.com/forlilab/AutoDock-Vina/blob/bb64e5e26b520569a86360d69248069b5e665b3f/src/lib/model.cpp), [rotor convention](https://github.com/forlilab/AutoDock-Vina/blob/bb64e5e26b520569a86360d69248069b5e665b3f/src/lib/conf_independent.cpp) and [score reporting](https://github.com/forlilab/AutoDock-Vina/blob/bb64e5e26b520569a86360d69248069b5e665b3f/src/lib/vina.cpp). That source pin and the independently installed 1.2.7 executable are recorded separately.

## Boundaries

This is a CPU score-contract check with eight ordinary ligands and one controlled receptor. Native PDBQT bond reconstruction can disagree with a supplied chemical graph on other inputs. Macrocycles, flexible receptors and metal coordination are outside this audit. It establishes neither general chemical coverage, actual docking success, affinity-prediction accuracy nor GPU speed. No CUDA device was available.
