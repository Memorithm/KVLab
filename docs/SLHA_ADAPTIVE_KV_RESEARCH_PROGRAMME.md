# SLHA Adaptive-KV Research Programme

Status: preregistered programme umbrella, 2026-09-26.

## Motivation

SLHAv2's current direct real-model score replacement is a quality NO-GO. Its
diagnostic ranking intervention indicates that key ordering, especially the
top set, is a major source of loss. This programme asks what KV information
must be retained and how cheaply it can be represented or routed.

The umbrella programme does not replace the narrower BKV and elastic-width
preregistrations. It composes them under one SLHAv2-facing evidence boundary.

## Primary outcome vector

Do not collapse these into one scalar:

```text
Q = (
  model_quality,
  top_k_recall,
  boundary_pair_accuracy,
  retained_softmax_mass,
  selected_density,
  logical_kv_bytes_touched,
  physical_bytes_or_traffic_if_measured,
  TTFT,
  TPOT,
  tokens_per_second,
  controller_transition_cost
)
```

The first three mechanistic metrics do not substitute for model quality.
Logical byte counts do not substitute for physical traffic.

## Frozen initial k values

Development panels report:

```text
K = {1, 4, 8, 16, 32, 64}
```

A confirmatory experiment may freeze a subset before execution, but must not
select k after observing the confirmatory population.

## Softmax-mass definition

For reference scores s over the causally visible universe U:

```text
p_i = exp(s_i - max(s)) / sum_j exp(s_j - max(s))
retained_mass(S) = sum_{i in S} p_i
omitted_mass(S) = 1 - retained_mass(S)
```

This is an evaluation oracle when exact dense scores are not available to a
deployable candidate.

## Ranking-boundary definition

For reference top-k set T and its complement N, boundary-pair accuracy compares
the candidate ordering for every pair (t,n) in T x N against the reference
ordering. Ties use the canonical stable item id rule.

This metric distinguishes a candidate that contains many correct top-k items
but places them below dangerous negatives.

## Initial candidate families

1. exact reference top-k/mass oracle — non-deployable upper bound;
2. random matched-density;
3. recent-tail matched-density;
4. sign/Hamming signatures;
5. sign/Hamming + norm;
6. age/position/frequency;
7. residual-energy / SLHA state;
8. Boolean combinations of qualified predicates;
9. adaptive widening/repair;
10. static and elastic widths 64→2048 bits.

## Literature-driven comparison requirements

The 2025 state of the art motivates explicit comparison classes:

- low-rank KV projection;
- adaptive/layer-wise mixed precision;
- vector/codebook quantization;
- two-stage coarse eviction + sparse top-k attention;
- query-agnostic importance/reconstruction.

These are baseline families to reproduce or approximate fairly where practical,
not evidence for Memorithm mechanisms.

## Data separation

Each real-model campaign must declare distinct identities for:

- calibration/training;
- development/validation;
- diagnostic;
- protected/final where applicable.

No protected result may select thresholds, widths, layers, heads, predicates
or repair tiers.

## SLHAv2 handoff

A canonical export bundle should eventually bind:

- schema version;
- KVLab commit;
- candidate id/config;
- model/tokenizer/dataset identities;
- selected universe and survivor IDs or reproducible policy;
- all quality metrics;
- all controls;
- execution environment;
- limitations.

SLHAv2 imports the bundle as evidence only. It owns implementation, destination
tests and promotion.
