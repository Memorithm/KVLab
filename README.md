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

The first cross-project BIKV handoff is implemented in `Memorithm/FLAT-ATTENTION`. BKV-K6.2 at FLAT commit `4e686bcad4f73e79d3cbf4d84c92a6e9b4673383` compares Boolean-selected paged numerical decode against the existing M16 paged decode while preserving original logical token positions and exact full/selected/avoided logical numerical K/V byte accounting. Its all-accept path checks O/LSE parity against M16 and its sparse path checks a scalar oracle restricted to the selected original positions.

BKV-K6.3 is merged at FLAT commit `0598bca2d39ae6e321bd0e40b25157022f8873ee`. It adds a deterministic, machine-readable research evidence envelope that binds BIKV candidate and dense M16 records to the same exact commit, environment, attention problem and measurement protocol; preserves phase medians as diagnostics; and makes any promotion/fallback disposition from the measured end-to-end candidate median versus the dense median. The schema also records whether Q/KV are device-resident, whether a host Q mirror remains, whether GPU timestamps, physical DRAM traffic, model quality, or resident-only production are actually claimed, and rejects contradictory host-mirror plus resident-only claims.

BKV-K6.4 is merged at FLAT commit `7a4eec7dbb90627dde800bc1c3c90dbdd890d6d1`. It connects the K6.3 evidence envelope directly to the live K6.2/M16 qualification harness so the BIKV candidate and dense baseline records are emitted from the same measured execution, with exact M40 benchmark provenance and canonical schema-versioned JSON. The current harness still uses host-observed timing and retains a host Q mirror; K6.4 therefore strengthens evidence capture and reproducibility but does not establish GPU-resident execution, physical DRAM traffic reduction, model-quality preservation, or a representative end-to-end speedup.

FLAT M13B.4 trace evidence is separately merged at commit `ca164387143cd58b5349d1e52850a2907071ed4c`. It defines a backend-neutral event contract for prefill commit/readiness, Boolean routing, numerical work and synchronization, with explicit host-wall-clock versus device-timestamp provenance and serial/multi-dispatch/fused scheduling variants. KVLab may consume that trace contract when a future BIKV campaign measures first-token readiness or CPU/GPU cooperation, but the schema itself is not a BKV-K6 result and does not prove overlap, lower TTFT/TPOT, reduced physical traffic, or resident-only execution.

Those FLAT harnesses remain correctness/provenance and host-observed timing evidence for their declared synthetic fixtures, not a KVLab T430 BKV-K5 hardware result and not a representative-model performance claim. The current K6 path still retains a host Q mirror; no universal TTFT/TPOT, physical DRAM-bandwidth, model-quality, resident-only, or end-to-end attention speedup claim follows from K6.2–K6.4 or the M13B.4 trace schema.

The next KVLab Boolean-KV gate is therefore to retain/cross-link BIKV evidence packs under KVLab provenance and execute target-host/model measurements of Boolean overhead, numerical K/V bytes avoided, `numerical_KV_bytes_avoided / Boolean_KV_bytes_read`, first-token latency, TPOT, candidate recall/false negatives and downstream correctness before runtime promotion or adaptive placement.

## Canon

See `Memorithm/scirust-hub` `CATALOG.md` and ADR-0020. Existing preregistrations remain authoritative for their campaigns. New Boolean KV experiments must be preregistered under the dedicated roadmap before confirmatory runs; the new programme extends KVLab rather than redesigning or duplicating it.