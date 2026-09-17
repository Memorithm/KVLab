# BKV-K6 paired descriptive campaign summary

`kvlab.bikv-target-paired-summary.v1` is the first analysis surface over a complete `kvlab.bikv-target-campaign.v1` manifest. It exists to make baseline/candidate comparison replayable without silently dropping failed or unavailable measurements.

The builder first re-verifies the frozen protocol, campaign manifest and complete retained run payloads. It then pairs `baseline` and `candidate` by the exact `(seed, repetition_index)` slot. Every required BIKV metric retains the baseline/candidate run status, metric status, raw value/unit/reason and, when both sides are measured, a descriptive candidate-minus-baseline delta. Boolean correctness is compared as exact equality rather than coerced to a number.

For numeric metrics the record reports the measured-pair count, incomplete-pair count, median paired delta and observed min/max paired delta. Unit drift fails closed. `not_exposed`, `failed` and failed-run states remain visible in the point records and reduce completeness instead of being filtered away.

This contract is deliberately non-inferential. It does not synthesize a p-value, confidence interval, effect threshold, winner, promotion disposition, BKV-K9/NBKV authorization, hardware claim or protected-holdout opening. Those decisions require a separately frozen analysis/decision rule before outcome inspection. The target protocol's H0/H1, quality rule and holdout policy remain authoritative declarations; this summary does not reinterpret them.

A target-host/model campaign is still required. Synthetic fixtures and unit tests qualify only the replay/completeness semantics of this analysis surface.
