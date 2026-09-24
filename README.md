# KVLab

Experimental laboratory for KV-state representation, manipulation, transfer and causal analysis.

The 223-byte README was lying by omission. The lab already has a Python package (`kvlab/`), tests, and preregistered KV studies. Do not start a second KV repository.

## What this repository is

- Execution surface for campaigns C1–C11 / C6 as preregistered in `docs/`
- Causal analysis of KV-cache mechanics used by FLAT / TurboQuant / ADA
- Scientific source of truth for the first-class **Boolean KV Cache** programme
- Evidence packs belong here, not in a new lab

## What this repository is not

- Not the FLAT kernel owner (`Memorithm/FLAT-ATTENTION`)
- Not the TurboQuant codec owner (`Memorithm/TurboQuant`)
- Not a training run farm
- Not a place to silently change a preregistration after seeing results

## Layout

```text
docs/     preregistrations, protocols and research roadmaps
kvlab/    library
tests/    executable checks
```

## Priority research programmes

- [Boolean KV Cache — first-class memory-tier roadmap](docs/BOOLEAN_KV_CACHE_ROADMAP.md)
- [K9 cross-model KV transfer — preregistered reproduction track](docs/K9_CROSS_MODEL_TRANSFER_PREREGISTRATION.md)
- [KV Tiering + Replay Study — Preregistration](docs/KV-TIERING-REPLAY-PREREGISTRATION.md)
- [Conjecture Research Programme](docs/CONJECTURE-RESEARCH-PROGRAMME.md)

### K9 cross-model transfer

The K9 reproduction track now has frozen compatibility/holdout gates, immutable capture and replay contracts, protocol binding, a deterministic calibration-only ridge baseline, and a fail-closed standard-RoPE removal oracle in `kvlab.k9_rope`. The RoPE oracle supports only explicitly declared interleaved-pair or half-split coordinate layouts with caller-pinned base, rotary dimension and token position; it does not infer model semantics or authorize final-holdout execution. A concrete source→target family pair still needs immutable model/tokenizer revisions, verified actual RoPE/scaling semantics, dataset identities and preregistered acceptance thresholds before any final result can exist.

### Boolean KV Cache

KVLab now distinguishes three architectures that must never be conflated:

1. **NKV** — numerical KV baseline;
2. **BIKV** — Boolean-indexed numerical KV, where Boolean state selects pages/blocks but exact numerical K/V remains authoritative;
3. **NBKV** — native Boolean KV for explicitly Boolean operations, research-only until separately qualified.

The dedicated roadmap covers packed Boolean page indexes, CPU/SIMD/multicore execution, dual-socket NUMA experiments, large-RAM qualification, portable GPU search, CPU+GPU cooperative execution, first-token readiness, and composition with the existing KVLab programme.

The first dedicated large-memory target is the verified Dell T430 host captured in the BKV evidence pack: 2 × Intel Xeon E5-2683 v4, each with 16 physical cores / 32 SMT threads, for 32 physical cores / 64 logical CPUs total, two NUMA nodes and approximately 125.8 GiB system RAM. Every run must continue to detect and record the exact CPU SKU, physical/logical core count, NUMA topology, cache hierarchy and available instruction set rather than infer them from the chassis name.

#### Current T430 qualification status

BKV-K4 has now qualified a **distinct socket-local shard** layout for the frozen 32,000,000-page × 256-bit T430 campaign at exact evaluated commit `7e02edc1e3ac3b53cfb352cfad560db439755a0f`. Two 16,000,000-page shards, each scanned by 16 physical workers on its local socket, produced exactly the same 2,283 global candidate page IDs as the 32-physical-core interleaved monolithic oracle. The measured median pair latency was 12.133153 ms versus 15.111998 ms for the monolithic comparator, i.e. 1.245513× for this frozen host/workload. The reported 84.396859 GB/s is derived logical packed-signature throughput, not a direct DRAM-bandwidth measurement. This result is host/workload-specific and does not establish an end-to-end attention speedup.

#### Current BIKV integration status

KVLab exposes a versioned structural handoff for ProspectEngine in `kvlab/prospect_handoff.py`. Schema `kvlab.prospect-bkv-handoff/v1` binds the exact packed query/page signatures, cache generation and Hamming threshold to the admitted logical page IDs. Packed `u64` words use fixed 16-digit lowercase hexadecimal JSON strings so the bit pattern survives consumers that cannot represent every 64-bit integer exactly. The decoder recomputes the Hamming selection and rejects candidate-set tampering. This handoff carries no latency, traffic, quality or end-to-end speedup claim; it is a reproducible bridge from KVLab's Boolean-KV oracle to downstream routing experiments.

KVLab also retains the downstream provenance of that structural bridge with `kvlab.bikv-consumer-receipt/v1`, merged at commit `a1884432cc2218e9e873582316aa6a6629cc96b4`. The receipt binds the exact canonical handoff by SHA-256 to producer and consumer repository/revision pairs plus the downstream evidence schema, and can reject a different handoff payload during replay. The frozen cross-project fixture records KVLab handoff revision `0fb5adc5babfaea9077342db881ce775eacc4442` and ProspectEngine consumer merge `b6d96469bcb92f50cb95e7512920cebea1c4dd41`. This is provenance evidence only, not a benchmark or correctness/quality claim for numerical attention.

KVLab now also exposes a separate replayable logical-eviction handoff in `kvlab/prospect_eviction_handoff.py`. Schema `kvlab.prospect-kv-eviction/v1` records the ordered input token IDs, `oldest_first` retention policy, retained/evicted token IDs and exact logical byte accounting, then recomputes the result through `apply_eviction` during decode. This contract deliberately stops at logical KV semantics: `logical_evicted_bytes` is not allocator release, freed HBM, transfer avoidance, latency reduction or downstream-quality evidence. Those effects require separate instrumentation or an explicit numerical evaluator in the consumer.

The first cross-project BIKV handoff is implemented in `Memorithm/FLAT-ATTENTION`. BKV-K6.2 at FLAT commit `4e686bcad4f73e79d3cbf4d84c92a6e9b4673383` compares Boolean-selected paged numerical decode against the existing M16 paged decode while preserving original logical token positions and exact full/selected/avoided logical numerical K/V byte accounting. Its all-accept path checks O/LSE parity against M16 and its sparse path checks a scalar oracle restricted to the selected original positions.

BKV-K6.3 is merged at FLAT commit `0598bca2d39ae6e321bd0e40b25157022f8873ee`. It adds a deterministic, machine-readable research evidence envelope that binds BIKV candidate and dense M16 records to the same exact commit, environment, attention problem and measurement protocol; preserves phase medians as diagnostics; and makes any promotion/fallback disposition from the measured end-to-end candidate median versus the dense median. The schema also records whether Q/KV are device-resident, whether a host Q mirror remains, whether GPU timestamps, physical DRAM traffic, model quality, or resident-only production are actually claimed, and rejects contradictory host-mirror plus resident-only claims.

BKV-K6.4 is merged at FLAT commit `7a4eec7dbb90627dde800bc1c3c90dbdd890d6d1`. It connects the K6.3 evidence envelope directly to the live K6.2/M16 qualification harness so the BIKV candidate and dense baseline records are emitted from the same measured execution, with exact M40 benchmark provenance and canonical schema-versioned JSON. The current harness still uses host-observed timing and retains a host Q mirror; K6.4 therefore strengthens evidence capture and reproducibility but does not establish GPU-resident execution, physical DRAM traffic reduction, model-quality preservation, or a representative end-to-end speedup.

FLAT M13B.4 trace evidence originates from the backend-neutral event contract merged at commit `ca164387143cd58b5349d1e52850a2907071ed4c`; its canonical machine-readable `flat.m13b4-trace.v1` envelope is merged at FLAT commit `29c18275b5687b71599ef11f5badcc4571a52a41`. KVLab now retains that exact canonical envelope through the fail-closed consumer merged at `184f78925e6089d1ac923824a38c5e7b3253f38e`, content-addressing the producer bytes while requiring complete synchronization evidence for every multi-dispatch candidate. This is evidence infrastructure only: neither the producer schema nor the KVLab consumer proves overlap, lower TTFT/TPOT, reduced physical traffic, resident-only execution, model quality, or speedup.

FLAT M13B.5 page-selection evidence is now canonicalized as `flat.boolean-kv-selection.v1` at FLAT merge `315716b1bf6c42cf5439b351aa9c1c3e7a1644ba`. KVLab consumes that exact producer encoding through the fail-closed BKV-K6 consumer merged at `689d5efc261c6a7cc52a43fde4de8472626f2a8a`, verifies the producer checksum and page/signature/accounting invariants, and retains a SHA-256 identity over the canonical producer bytes. The record preserves logical full/selected/avoided numerical K/V byte accounting only; it is not physical HBM/DRAM/cache/PCIe traffic, latency, model-quality, residency, or speedup evidence.

FLAT BKV-K6 declared-target selection-quality evidence is merged as `flat.bikv-selection-quality.v1` at FLAT commit `dfd5fb7a90242b5e95c3946186864f56c83930bd`. KVLab retains that exact canonical producer record through the fail-closed consumer merged at `65f8c9a405d1d5f9a542462645c3dc09e0aa25c6`, verifies the complete outer checksum, recomputes TP/FN/FP plus recall/FNR and candidate density from the retained selection and independently declared target, and content-addresses both the complete quality record and embedded selection. This is page-selection quality evidence only: the target remains an experimental input, recall is not downstream model quality, and no physical-traffic, latency, energy, speedup, adaptive-tiering, or promotion claim follows.

Those FLAT harnesses remain correctness/provenance and host-observed timing evidence for their declared synthetic fixtures, not a KVLab T430 BKV-K5 hardware result and not a representative-model performance claim. The current K6 path still retains a host Q mirror; no universal TTFT/TPOT, physical DRAM-bandwidth, model-quality, resident-only, or end-to-end attention speedup claim follows from K6.2–K6.4 or the M13B.4 trace schema.

The next KVLab Boolean-KV gate is therefore to retain/cross-link BIKV evidence packs under KVLab provenance and execute target-host/model measurements of Boolean overhead, numerical K/V bytes avoided, `numerical_KV_bytes_avoided / Boolean_KV_bytes_read`, first-token latency, TPOT, candidate recall/false negatives and downstream correctness before runtime promotion or adaptive placement.

The target-host/model campaign inputs are now frozen by `kvlab.bikv-target-protocol.v1` (`docs/BKV-K6-TARGET-PROTOCOL.md`): exact KVLab/FLAT/model/tokenizer/runtime/dataset revisions, hardware fingerprint, precision/context/batch, seeds/warmups/repetitions, full-cache/native-prefill baseline, holdout/tuning boundary, timing/byte evidence kind and the mandatory BKV-K5/K6 correctness, memory/traffic, transfer/synchronization, backpressure and performance metric surface is content-addressed before execution. This is preregistration infrastructure only; it does not satisfy the BKV-K9 evidence gate without actual retained measurements.

The corresponding per-attempt retention contract `kvlab.bikv-target-run.v1` (`docs/BKV-K6-TARGET-RUN.md`) binds baseline/candidate attempts to that exact protocol and requires every frozen metric obligation to be retained explicitly as measured, not exposed, or failed. Failed attempts are first-class evidence rather than silently discarded. This is still evidence infrastructure, not a target-host/model qualification result. The companion `kvlab.bikv-target-campaign.v1` manifest now fail-closes campaign completeness by requiring exactly one baseline and one candidate attempt for every frozen seed/repetition slot and content-addressing each retained run; it performs no statistical aggregation or promotion decision.

The companion `kvlab.bikv-target-paired-summary.v1` surface now verifies those retained payloads against the complete campaign manifest and emits exact seed/repetition paired descriptive deltas while retaining unavailable/failed metric states. It is intentionally non-inferential: no p-value, confidence interval, winner or BKV-K9/NBKV promotion is synthesized before a separate decision rule is frozen. See `docs/BKV-K6-PAIRED-DESCRIPTIVE-SUMMARY.md`.

The preregistration boundary is now explicit through `kvlab.bikv-target-analysis-plan.v1` (`docs/BKV-K6-ANALYSIS-PLAN.md`). It content-addresses the protocol-bound H0/H1, primary performance metric and direction, minimum effect, quality noninferiority guard, uncertainty method, complete-pair requirement, multiplicity policy and holdout policy before outcome inspection. The contract does not choose campaign thresholds or evaluate outcomes; BKV-K9 remains blocked until a concrete plan is frozen before target measurements and a later evaluator applies it without tuning.

The next fail-closed layer is now implemented as `kvlab.bikv-target-decision-plan.v2` plus `kvlab.bikv-target-decision.v1` (`docs/BKV-K6-DECISION-EVALUATOR.md`). It freezes the previously implicit paired effect statistic and interval convention, requires explicit candidate-side recall/FNR/O/LSE/reset guards, rebuilds the paired summary from retained runs, and records pass, negative, or incomplete-evidence dispositions without opening BKV-K9. No concrete target-host/model outcome is added by this evaluator infrastructure.


BKV-K8 first-token observations now have an explicit protocol-binding verifier: the canonical v2 record must match the retained BIKV evidence bundle and frozen target protocol on evidence identity, producer identity/frozen FLAT revision, hardware fingerprint, timing source and byte-evidence kind before it can be retained as campaign evidence. A second fail-closed verifier binds the same observation to one exact completed candidate target-run record and requires the shared first-token/Boolean-byte measurements to match exactly. These close provenance/replay gaps only; no first-token latency result, traffic reduction, model-quality result or BKV-K9/BKV-K11 promotion is added by the verifiers. A campaign-binding receipt additionally requires the complete retained run payload set to reproduce the exact campaign manifest before linking that candidate run and observation to the campaign; failed sibling attempts remain visible and no campaign outcome is synthesized.

## Canon

See `Memorithm/scirust-hub` `CATALOG.md` and ADR-0020. Existing preregistrations remain authoritative for their campaigns. New Boolean KV experiments must be preregistered under the dedicated roadmap before confirmatory runs; the new programme extends KVLab rather than redesigning or duplicating it.

## Licensing

KVLab-owned source code is available under the PolyForm Noncommercial License
1.0.0. See [LICENSE](LICENSE), [LICENSE.md](LICENSE.md), and
[LICENSING.md](LICENSING.md). Commercial use requires a separate written
commercial agreement from the copyright holder. Third-party dependencies,
datasets, models, papers, and artifacts retain their own applicable terms and
attributions.

### DeepSeek-V4.1 KV reuse/replay programme

The 2026-09-24 review of DeepSeek-AI's *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression* adds a new future-only KVLab programme covering cross-layer KV/index reuse, hierarchical candidate pools, bounded replay, persistent-versus-session cache separation, and an explicit FP4 comparator. See [DSV41 KV reuse/replay roadmap](docs/DEEPSEEK_V41_KV_REUSE_REPLAY_ROADMAP.md).

This programme does not alter earlier frozen preregistrations or retroactively reinterpret existing BKV/K9/tiering results. FP4 and INT4 are separate evidence families, and DeepSeek's reported compression ratios are not KVLab results.
