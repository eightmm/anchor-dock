# AD4 map and intramolecular score checks

The rigid-receptor reference backend combines AutoGrid ingestion, prepared intra contributors, torsion and explicit unbound reporting through AD4Scorer. See [usage and boundaries](../docs/PREPARED.md).

An external CPU check used Meeko 0.8.0 and native Vina 1.2.7 AD4 on four flexible ligands with 6, 9, 17 and 21 prepared intra contributors. Receptor fields are **synthetic affine AutoGrid-format maps**, isolating interpolation/wrapper behavior from receptor preparation.

| Component | Maximum absolute difference |
|---|---:|
| Intermolecular maps, matched Vina conventions | 3.55e-15 |
| Same-as-bound final score | 3.55e-15 |
| Continuous analytic intra | 9.12e-4 |
| Diagnostic native radial-table intra | 1.11e-16 |

Native factor-32 squared-distance FAST table averaging explains the intra residual. The backend retains a continuous forward and its exact derivative; native search gradient approximations/quantization are not copied. Same-as-bound reporting directly computes inter + torsion to avoid cancellation. Zero-distance electrostatics and total-per-pair curl have regressions.

Reproduce with optional audit dependencies from the native Vina report:

```bash
uv run --no-sync python examples/audit_ad4.py --output reports/ad4-native-scoring-2026-10-30.json
uv run --frozen pytest -q tests/test_ad4.py tests/test_autogrid.py tests/test_grid_scoring.py
```

See [numerical evidence](ad4-native-scoring-2026-10-30.json). Native AutoDock-GPU execution, real receptor AutoGrid generation and GPU correctness/throughput remain unmeasured. Map contents are authoritative; the loader cannot prove their association with a separate receptor graph. Maps remain preweighted when analytic intra/torsion weights change.
