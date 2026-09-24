# DeepSeek-V4.1 KV reuse/replay research roadmap

Status: planned, future-only, evidence-gated.

Research input reviewed 2026-09-24: DeepSeek-AI, *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression*.

This document creates a new KVLab campaign family. It does not amend any earlier frozen preregistration and does not import DeepSeek's compression, quality, latency, or throughput results as Memorithm evidence.

## Questions

1. When does explicit cross-layer reuse of K/V or sparse-selection state preserve downstream quality?
2. Can a first broad selector construct a bounded candidate pool that later selectors reuse without unacceptable false negatives?
3. When is it better to drop short-horizon state and replay a bounded recent window than to keep or offload it?
4. How does a portable FP4 KV representation compare with current INT4/NF4/TQ3/MIXED baselines under exact byte accounting?
5. How do Boolean page selection, precision, reuse and replay interact rather than merely add?

## DSV41-KV0 — representation accounting

Freeze an exact portable FP4 comparator based on E2M1 payload values plus explicit group scales before experiments. Record payload, scale, metadata, alignment and padding bytes. INT4 and FP4 remain distinct arms.

## DSV41-KV1 — cross-layer reuse

Freeze layer groups and three modes: Full, Reindex and Reuse. Measure K/V bytes resident, index state bytes, selector work, downstream quality and latency. Include no-reuse and matched-random controls. A layer may reuse state only when source layer, representation, epoch, geometry and causal domain identities match exactly.

## DSV41-KV2 — hierarchical candidate pools

The first selector may scan the complete declared domain and produce a larger candidate pool. Later selectors operate only inside that pool. Measure target recall/FNR, retained mass where an oracle exists, O/LSE error for attention consumers, pool bytes and selector cost. Do not infer physical bandwidth from candidate counts.

## DSV41-KV3 — bounded replay

Create paired full-state versus replay continuations. The replay arm may reconstruct only from explicitly retained immutable inputs/provenance and a frozen recent-window size. Exact and approximate reconstruction are separate protocols. Approximate replay requires a preregistered quality guard.

## DSV41-KV4 — persistent versus session-local lifetime

Test global/long-tail reusable state and short-lived active-session state with separate retention/TTL policies. Report hit rate, resident bytes, transferred bytes, replay frequency and quality. Do not assume the source report's 72-hour/minute lifetimes transfer to Memorithm workloads.

## DSV41-KV5 — factorial composition

Cross the independently qualified axes:
- Boolean page/candidate selection;
- cross-layer reuse;
- FP4 versus INT4/other declared precision;
- keep/compress/offload/drop-and-replay.

Use factorial or ablation designs sufficient to attribute effects. No combined arm may hide the cost of routing, materialization, scale metadata, replay, transfer or synchronization.

## Ownership

- KVLab owns experiment design, preregistration, controls, statistics and negative results.
- FLAT-ATTENTION owns sparse attention/candidate execution.
- SLHAv2 owns compressed-KV/replayable-state semantics.
- NNIS owns runtime execution and physical movement/measurement.
- SciRust may host reusable deterministic reference codecs/contracts.
- ElasticXxx may choose only among already-qualified actions and retains rollback responsibility.

## Promotion gate

No mechanism is promoted from this programme unless it passes its frozen quality/correctness guard and improves at least one declared systems objective with all overhead included. External DeepSeek results are prior evidence only and never satisfy this gate.
