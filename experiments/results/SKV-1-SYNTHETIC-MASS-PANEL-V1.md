# SKV-1 synthetic mass panel v1 — retained result

Status: DEVELOPMENT / synthetic falsification evidence only.

The frozen panel contains six deterministic score rows and four equal-density
survivor policies. Every arm keeps 2/8 items (25%) except that the exact policy
is simply the reference top-2 at the same density.

## Key observation

Top-k recall alone is insufficient to characterize the cost of a miss.

On the **dominant-head** row, two policies both have top-2 recall = 0.5:

| policy | retained softmax mass | missed top-k mass |
|---|---:|---:|
| drop strongest key, add rank-3 | 0.000638 | 0.998229 |
| keep strongest key, replace rank-2 with rank-3 | 0.998532 | 0.000335 |

The set-level top-k recall is identical, but the retained probability mass is
nearly opposite. This directly justifies keeping a mass-weighted miss metric in
the SLHAv2-facing evaluator.

The same top-k recall = 0.5 on the **near-tie-boundary** row loses much less
top-k probability mass (0.243121 when the strongest key is replaced) than the
dominant-head row (0.998229). The damage of one ranking error therefore depends
strongly on score geometry.

The tail matched-density control also demonstrates that positional recency is
workload-dependent: it retains only 0.000387 mass on the dominant-head row but
0.970895 on the late-relevance row.

## Interpretation

This result does **not** establish a production router, model-quality benefit,
latency improvement or memory saving. It establishes only that:

1. equal candidate density is not equal information retention;
2. equal top-k recall can hide radically different softmax-mass loss;
3. a mass-aware diagnostic is necessary before promoting sparse/routed KV into
   SLHAv2.

Machine-readable values: `SKV-1-SYNTHETIC-MASS-PANEL-V1.json`.
Reproduce with `python scripts/run_slha_synthetic_panel.py`.
