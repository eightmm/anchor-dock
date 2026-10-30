# Known-pocket local pose recovery: 2026-10-30

Three public crystal complexes provide a local-refinement sanity check: one crystal control plus four mild and four moderate seeded rigid/torsional perturbations each. This does not measure global docking, cross-docking or screening enrichment.

Each tested optimizer finishes with **24/24 perturbed starts within 2 Å and passing limited sanity checks**. **19/24 already passed initially: five are newly recovered and zero lost.** Adam/LBFGS already achieve this in the scientific baseline. Cartesian shows no general accuracy or speed improvement and remains opt-in.

## Protocol

- Vina-family analytic score, inferred-xs-v3 chemistry. This is not an authoritative prepared-receptor/native full-score parity check.
- CPU, one thread, seed 713, deterministic algorithms, 100 outer steps per pose. Crystal controls are excluded from recovery counts.
- Mild: Gaussian translation RMS .5 Å, rotation-vector RMS 10°, independent torsion SD 5°. Moderate: 1.5 Å, 30°, 15°. These are distribution scales, not exact displacement bounds.
- Adam learning rate .05 and batch9; LBFGS .5 and single-pose chunks; Cartesian 1.0, damping .001, actual displacement cap .25 Å, 12 backtracks and single-pose chunks. Parameters were fixed before evaluation.
- Heavy-atom symmetry RMSD uses RDKit CalcRMS in the fixed receptor frame, including conjugated terminal symmetry. **No ligand alignment.** A finite match cap fails closed.
- A threshold pass requires RMSD ≤2 Å, finite coordinates, bond-length drift ≤.002 Å and minimum receptor-heavy distance ≥1 Å. These are limited sanity checks, not a chemical-validity assessment.
- Reported scores share the crystal intramolecular reference across starts/optimizers. Raw inter/intra/search components remain separate.

## Results

Moderate starts are the larger perturbation group (four per complex). Three selected complexes do not estimate a population success rate.

| Complex | Optimizer | Moderate median RMSD, initial → final (Å) | Initial → final threshold passes | Newly recovered / lost | Crystal control final RMSD (Å) |
|---|---|---:|---:|---:|---:|
| 10gs | adam | 2.025 → 0.735 | 6/8 → 8/8 | 2 / 0 | 0.701 |
| 10gs | lbfgs | 2.025 → 0.705 | 6/8 → 8/8 | 2 / 0 | 0.717 |
| 10gs | cartesian | 2.025 → 0.822 | 6/8 → 8/8 | 2 / 0 | 0.600 |
| 1iep | adam | 2.320 → 0.237 | 5/8 → 8/8 | 3 / 0 | 0.250 |
| 1iep | lbfgs | 2.320 → 0.242 | 5/8 → 8/8 | 3 / 0 | 0.199 |
| 1iep | cartesian | 2.320 → 0.875 | 5/8 → 8/8 | 3 / 0 | 0.206 |
| 5x72 | adam | 1.600 → 0.445 | 8/8 → 8/8 | 0 / 0 | 0.482 |
| 5x72 | lbfgs | 1.600 → 0.444 | 8/8 → 8/8 | 0 / 0 | 0.437 |
| 5x72 | cartesian | 1.600 → 0.449 | 8/8 → 8/8 | 0 / 0 | 0.429 |

Crystal controls can move away from the experimental pose while lowering the approximate scoring objective. Score decrease alone does not establish a more accurate pose. At 100 steps Cartesian also does not consistently reach Adam/LBFGS objective values or RMSD; no superiority is claimed.

## Preservation and source identity

Baseline local commit `b22926437b635454874035d92aaa35f2340dbef6` corresponds to relevant scientific code in remote PR commit `957c75c39298ddc2bfd2cba746f515cfb13925a8`. The integrated source was copied to a separate immutable checkout before execution. All six benchmark processes report unchanged package and driver hashes during execution. The integrated checkout retains the baseline Git HEAD, so its source digest identifies tested code; Git HEAD alone does not.

- Baseline package digest: `93f230cb77c7e5e0951738ab324f083d59956d996da995819c0385351e223f31`.
- Integrated package digest: `3edf156991807ed3898ca102749464751efc908a2af9c6a74f415dd9e8bc876d`.
- Driver: `sha256:d3b03d30cbe543fae0ec6d7d59bee815a6c70e0e6b7d086a5c973465512760bb`.
- Input content and seeded initial-pose hashes agree between baseline/current for every complex.
- All recorded per-pose initial/final RMSD, search energy, reported score and inter/intra component scalars for Adam/LBFGS agree exactly with baseline in all three cases. Final coordinates were not exported for a separate coordinate-identity claim.

## Timing observations

One wall-clock CPU sample per optimizer includes optimization/framework work. Optimizers use different score/line-search evaluations per outer step; these are not throughput or speedup measurements. Dense Cartesian Jacobians are a correctness reference.

| Complex | Adam (s) | LBFGS (s) | Cartesian (s) |
|---|---:|---:|---:|
| 10gs | 5.07 | 50.99 | 55.51 |
| 1iep | 8.62 | 52.68 | 60.07 |
| 5x72 | 3.50 | 26.46 | 30.65 |

CUDA was unavailable. GPU execution, peak memory and throughput remain unmeasured.

## Input provenance

[Machine-readable evidence](redocking-2026-10-30.json) contains input hashes, pinned URLs, versions, warnings, individual poses and comparison checks. External structures are not vendored.

- 10gs uses bundled inputs at upstream AnchorDock `125a4f82ec94cc0ffeb71d44d92912f58ff9ba79` as supplied, including existing H and heteroatoms.
- Public 1iep/5x72 structures use official AutoDock-Vina `bb64e5e26b520569a86360d69248069b5e665b3f`. Retain complete ATOM residues when any atom is within 12 Å of any crystal SDF atom, including supplied H; preserve order/H, omit HETATM/TER, append END. Ligands are copied verbatim; ligand H are removed before optimization/RMSD.
- 5x72 conformation A is explicitly selected before parsing: drop B records and blank A labels. Original/prepared hashes and record counts are recorded. Prepared coordinates exactly match the previously implicit RDKit A coordinates; the benchmark does not rely on implicit selection.
- Receptor H remain available for typing; only heavy receptor atoms enter the minimum-distance sanity check. With pinned RDKit2025.9.5, unsanitized PDB parsing already retains H with removeHs=True. Explicit removeHs=False documents policy; these results do not demonstrate a prior numerical H-loss bug.

| Complex | Ligand heavy atoms | Receptor total / heavy atoms |
|---|---:|---:|
| 10gs | 33 | 1392 / 714 |
| 1iep | 37 | 2258 / 1115 |
| 5x72 | 24 | 1688 / 845 |

Pinned sources:

- [10gs: 10gs_ligand.sdf](https://github.com/eightmm/anchor-dock/blob/125a4f82ec94cc0ffeb71d44d92912f58ff9ba79/examples/10gs/10gs_ligand.sdf)
- [10gs: 10gs_pocket.pdb](https://github.com/eightmm/anchor-dock/blob/125a4f82ec94cc0ffeb71d44d92912f58ff9ba79/examples/10gs/10gs_pocket.pdb)
- [1iep: 1iep_ligand.sdf](https://github.com/forlilab/AutoDock-Vina/blob/bb64e5e26b520569a86360d69248069b5e665b3f/example/basic_docking/data/1iep_ligand.sdf)
- [1iep: 1iep_receptorH.pdb](https://github.com/forlilab/AutoDock-Vina/blob/bb64e5e26b520569a86360d69248069b5e665b3f/example/basic_docking/data/1iep_receptorH.pdb)
- [5x72: 5x72_ligand_p59.sdf](https://github.com/forlilab/AutoDock-Vina/blob/bb64e5e26b520569a86360d69248069b5e665b3f/example/mulitple_ligands_docking/data/5x72_ligand_p59.sdf)
- [5x72: 5x72_receptorH.pdb](https://github.com/forlilab/AutoDock-Vina/blob/bb64e5e26b520569a86360d69248069b5e665b3f/example/mulitple_ligands_docking/data/5x72_receptorH.pdb)

## Reproduce

Bundled case:

```bash
uv run --frozen python examples/benchmark_redocking.py --complex-id 10gs --optimizers adam lbfgs cartesian --seed 713 --steps 100 --poses-per-level 4 --threads 1 --output redocking-10gs.json
```

For external cases, fetch pinned source files, prepare pockets as specified above and supply `--ligand`, `--receptor`, `--complex-id` and repeated `--source-url` options. `--source-root /path/to/immutable-checkout/src` evaluates another implementation in a fresh process with the same driver.

Metric regression tests: `uv run --frozen pytest -q tests/test_benchmark_metrics.py` (8 passed). They cover fixed-frame translation/rotation, graph/carboxylate symmetry, match truncation, invalid coordinates and atom-map preservation.
