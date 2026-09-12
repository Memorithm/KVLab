# BKV-K1 — Signature and page-aggregation controls

Status: implementation/control protocol. No performance or quality result is claimed by this document.

## Question

Can a compact deterministic Boolean signature select numerical KV pages with a better recall-versus-density tradeoff than matched controls, before any SIMD/NUMA/GPU optimization is considered?

## Null and alternative

- **H0:** at matched candidate density, the tested K-derived Boolean signature does not improve target-page recall over the frozen random and structural controls.
- **H1:** at least one preregistered K-derived Boolean signature improves target-page recall at matched candidate density on confirmatory data.

This document does not set a universal effect-size threshold because the concrete model/workload/target definition belongs in each experiment manifest. That threshold must be frozen before confirmatory evaluation.

## Frozen controls

BKV-K1 implements the following deterministic controls:

1. `random_control_signature(bit_length, seed, identity)` — stable pseudo-random bits used only as a matched control; no semantic interpretation.
2. `positional_control_signature(...)` — stable structure-only signature derived from logical page and token range; it contains no K/V values.
3. `sign_projection(values, threshold)` — one bit per finite numeric component using the explicit predicate `value >= threshold`.
4. Page aggregation over equal-width token signatures with exactly three rules: `or`, `and`, and strict `majority`. For an even majority tie, the output bit is `false`.

XNOR/popcount comparison is inherited from the BKV-K0 canonical packed representation.

BooleanLab-derived equations are excluded from this control PR. They require their own preregistration and equivalence/cost evidence before entering confirmatory KVLab runs.

## Target definition

Each concrete experiment must declare how `dense_target_pages` is obtained from the numerical oracle (for example top-k exact attention mass or a frozen cumulative-attention threshold). The target must be computed independently of Boolean candidate tuning. Empty targets are invalid.

## Primary metrics

For `P` total pages, selected candidate set `S`, and dense target `D`:

```text
true_positives       = |S ∩ D|
false_negatives      = |D - S|
recall               = |S ∩ D| / |D|
false_negative_rate  = |D - S| / |D|
candidate_density    = |S| / P
```

Also record exact Boolean signature logical bits and physical bytes from BKV-K0 accounting.

## Comparison rule

Do not compare signatures at arbitrarily different candidate densities and call the denser method better. Confirmatory comparison must use a preregistered matched-density procedure or report the full recall-density frontier.

## Data separation

Thresholds, signature width, aggregation rule and any top-k/cutoff are chosen on calibration data only. Confirmatory/holdout data may not be used to change them after inspection.

## Exit gate

BKV-K1 is implementation-complete when:

- every control is deterministic and tested;
- malformed/non-finite input fails closed;
- page aggregation semantics are frozen;
- recall, false-negative rate and candidate density are computed by a tested evaluator;
- no observed result has yet been promoted to a performance or scientific claim without an experiment manifest and evidence pack.
