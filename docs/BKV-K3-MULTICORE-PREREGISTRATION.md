# BKV-K3 Multicore Scaling — Preregistration

Status: preregistered correctness/orchestration slice; no performance result yet.

## Question

Can a multicore Boolean-KV page scan preserve the exact candidate set and accounting of the BKV-K2 packed single-process oracle while scaling over verified physical CPU resources?

## H0 / H1

- **H0:** sharding or merge semantics change the candidate set/accounting, or measured throughput does not improve after synchronization and memory costs are included.
- **H1:** candidate sets/accounting remain exactly equal to BKV-K2 and measured throughput improves for at least one preregistered worker count on the captured host topology.

A correctness failure rejects the performance comparison. Negative scaling regimes are retained.

## Frozen correctness contract

Pages are partitioned into contiguous, non-overlapping shards. The number of shards is `min(page_count, workers)`. Shard sizes differ by at most one; lower shard ids receive any remainder. Each shard runs the exact BKV-K2 packed Hamming oracle. Candidate ids are translated back to global page ids and merged in global page order.

Before any timing claim, the sharded candidate tuple, pages scanned, signature width, and bits compared must exactly equal the BKV-K2 reference for identical frozen inputs.

## Hardware evidence gate

Worker counts must not be described as physical cores until an evidence pack records the actual host topology. For the Dell T430 campaign this includes at minimum:

- `lscpu`
- `lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE`
- `numactl --hardware`
- relevant `/proc/cpuinfo` feature information
- `/proc/meminfo`

No CPU SKU, socket count, core count, SMT state, cache hierarchy, NUMA layout, ISA, or RAM capacity is inferred from the chassis name.

## Scaling measurements

Once a real parallel backend is implemented and the correctness gate passes, sweep only worker counts supported by captured topology. Report raw query latency/throughput plus:

`S(p) = T(1) / T(p)`

`E(p) = S(p) / p`

Include launch, synchronization, candidate merge, allocation, and memory-placement costs. Record local/remote/interleaved NUMA policy separately when BKV-K4 begins.

## Current implementation boundary

The initial K3 code intentionally executes deterministic shards serially. It freezes partition and merge semantics for later process/Rust multicore implementations. It is not evidence of multicore speedup.
