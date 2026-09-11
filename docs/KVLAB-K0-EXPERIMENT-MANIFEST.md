# KVLab K0 experiment manifest contract

KVLab treats each KV mechanism as a resource/quality trade-off experiment, not as a generic "compression" result. Before execution, an experiment must record a versioned manifest accepted by `kvlab.manifest.KVExperimentManifest`.

## Required distinctions

The manifest separates:

- taxonomy axes A–J from benchmark mechanism families 1–13;
- `applicable` from `not_applicable` for architecture-specific mechanisms;
- logical KV size from GPU residency, host RAM, storage, transfers, reads/writes, fragmentation, duplication, compute, and recomputation;
- `measured`, `estimated`, and `not_exposed` quantities;
- the full-cache/native-prefill oracle from the candidate policy;
- H0/H1, success/failure rules, holdout policy, repetitions and seeds;
- exact KVLab commit from exact immutable inter-repository dependency revisions.

Sparse reads, paging, prefix reuse and offload must not be described as reducing logical cache size unless the experiment actually measures such a reduction. A candidate can instead reduce reads, residency, duplication, fragmentation, transfer, compute, or another declared resource.

A manifest validates experimental intent and provenance. It contains no result and cannot by itself establish a performance or scientific claim.

## Cross-model baseline

Family 13 is anchored to the primary NVIDIA study:

Taekyung Heo et al., **Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse**, arXiv:2608.03893v1, 4 August 2026.

The paper studies within-family source→target transfer, defines matched-KV pairs as sharing KV-head count and per-head dimension, and uses per-head ridge regression with cross-layer source selection and RoPE-stripped key mapping. KVLab must first reproduce a compatible baseline rather than generalize the result to arbitrary model pairs.

Primary source: https://arxiv.org/abs/2608.03893

For cross-model experiments, native target prefill remains the oracle baseline. Failed transfer pairs are retained as first-class results. Reconstruction error alone is insufficient: downstream behavior/quality and prefill/decode costs must also be reported when exposed.

## Holdout rule

Calibration/development data may choose mapper hyperparameters only under the preregistered development protocol. Protected final holdouts must never be used to select layer count, source layers, regularization, quantization, eviction, tiering or composition policy.

## K0 exit condition

K0 is complete only when manifests can be machine-validated before execution and the inventory records exact revisions for KVLab plus any participating FLAT-ATTENTION, NNIS, ElasticXxx, SciRust or SLHAv2/TurboQuant component. K0 does not require a GPU result and must not fabricate one.
