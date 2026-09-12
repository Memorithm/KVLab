# K9 — Cross-model KV transfer preregistration

Primary source: Taekyung Heo, Rasoul Shafipour, Ritchie Zhao, Maximilian Golub, Mohammad Mahdi Kamani, Ritika Borkar, Makesh Tarun Chandran, Pantea Zardoshti, Bita Darvish Rouhani, **Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse**, arXiv:2608.03893v1, 2026-08-04.

This document freezes the first KVLab K9 protocol before any final-holdout execution. It is a reproduction protocol, not a claim that the reported NVIDIA results reproduce in KVLab.

## H0 / H1

- **H0:** for a preregistered compatible source→target pair, the transferred KV cache does not meet the preregistered quality-retention and reconstruction criteria relative to the target model's native-prefill oracle.
- **H1:** for that same pair and frozen mapper configuration, the transferred KV cache meets those criteria on the untouched final holdout.

Numerical acceptance thresholds are intentionally not invented here. They must be committed with the concrete model pair, workload, quality metric and hardware manifest before the final holdout is run.

## Applicability gate

The initial reproduction is schedulable only when all of the following are explicit and pinned:

1. source and target model IDs and immutable revisions;
2. tokenizer revisions;
3. equal KV-head count and equal per-head KV dimension for the pair;
4. runtime/backend and precision;
5. calibration identities and final-holdout identities, with no overlap;
6. source-layer count selected per target layer;
7. ridge regularization parameter;
8. explicit RoPE removal for key mapping when following the paper's ridge baseline.

If the architecture facts are unavailable or the KV geometry is incompatible, record `NOT_APPLICABLE`; do not simulate compatibility.

## Frozen baseline

The oracle is the target model's unmodified native prefill. The mandatory first mapping baseline is closed-form ridge/linear mapping. Calibration may choose only parameters declared in the preregistered calibration procedure; the final holdout is never used for mapper selection, layer selection, regularization tuning or stopping decisions.

## Measurements

Record, where actually instrumentable:

- K/V reconstruction error separately;
- target native-prefill versus transferred-KV downstream quality;
- prefill and transfer/reconstruction latency separately;
- TTFT and decode metrics separately;
- logical KV bytes and actually resident HBM/RAM only when exposed by instrumentation;
- transfer bytes and host/device traffic only when measured;
- repetitions, dispersion/uncertainty and all failures.

Any non-instrumented quantity must be labelled as an estimate. No result may be generalized beyond the exact tested pair/revisions.

## Failure preservation

Pairs that degrade, fail numerical checks, violate geometry prerequisites, or lose downstream quality remain in the result registry. Negative outcomes must not be removed from reports or replaced by a different pair after seeing holdout results.

## Next implementation slice

After this protocol gate is green in CI, add an oracle-backed capture/replay adapter for one pinned compatible model-family pair, then implement RoPE key normalization and ridge fitting against calibration data. Final holdout execution remains blocked until the concrete thresholds and dataset identities are committed.
