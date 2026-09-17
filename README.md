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
- [KV Tiering + Replay Study — Preregistration](docs/KV-TIERING-REPLAY-PREREGISTRATION.md)
- [Conjecture Research Programme](docs/CONJECTURE-RESEARCH-PROGRAMME.md)

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

Those FLAT harnesses remain correctness/provenance and host-observed timing evidence for their declared synthetic fixtures, not a KVLab T430 BKV-K5 hardware result and not a representative-model performance claim. The current K6 path still retains a host Q mirror; no universal TTFT/TPOT, physical DRAM-bandwidth, model-quality, resident-only, or end-to-end attention speedup claim follows from K6.2–K6.4 or the M13B.4 trace schema.

The next KVLab Boolean-KV gate is therefore to retain/cross-link BIKV evidence packs under KVLab provenance and execute target-host/model measurements of Boolean overhead, numerical K/V bytes avoided, `numerical_KV_bytes_avoided / Boolean_KV_bytes_read`, first-token latency, TPOT, candidate recall/false negatives and downstream correctness before runtime promotion or adaptive placement.

The target-host/model campaign inputs are now frozen by `kvlab.bikv-target-protocol.v1` (`docs/BKV-K6-TARGET-PROTOCOL.md`): exact KVLab/FLAT/model/tokenizer/runtime/dataset revisions, hardware fingerprint, precision/context/batch, seeds/warmups/repetitions, full-cache/native-prefill baseline, holdout/tuning boundary, timing/byte evidence kind and the mandatory BKV-K5/K6 correctness, memory/traffic, transfer/synchronization, backpressure and performance metric surface is content-addressed before execution. This is preregistration infrastructure only; it does not satisfy the BKV-K9 evidence gate without actual retained measurements.

The corresponding per-attempt retention contract `kvlab.bikv-target-run.v1` (`docs/BKV-K6-TARGET-RUN.md`) binds baseline/candidate attempts to that exact protocol and requires every frozen metric obligation to be retained explicitly as measured, not exposed, or failed. Failed attempts are first-class evidence rather than silently discarded. This is still evidence infrastructure, not a target-host/model qualification result.

## Canon

See `Memorithm/scirust-hub` `CATALOG.md` and ADR-0020. Existing preregistrations remain authoritative for their campaigns. New Boolean KV experiments must be preregistered under the dedicated roadmap before confirmatory runs; the new programme extends KVLab rather than redesigning or duplicating it.
