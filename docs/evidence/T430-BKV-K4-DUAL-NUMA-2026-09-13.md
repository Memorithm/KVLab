# T430 BKV-K4 dual-NUMA placement evidence — 2026-09-13

Status: verified hardware result; placement microbenchmark only.

## Provenance

- Git head: `46b1d479898eced73c0e26ed0bf6f949b6d81085`
- Tracked tree dirty: `false`
- Pre-existing untracked files: `84`
- Host: Dell T430, 2 × Intel Xeon E5-2683 v4, 16 physical cores/socket, SMT2, two NUMA nodes
- NUMA distance: local 10, remote 21
- Corpus: mirrored identical deterministic pages on both sockets
- Shard pages: 16,000,000 per socket
- Total logical pages: 32,000,000
- Signature width: 256 bits
- Hamming threshold: 96
- Workers/socket: 16 physical-core representatives, no SMT
- Warmup: 5
- Repetitions: 25
- Seed: 1

## Results

| placement | node0 median ns | node1 median ns | pair median ns | logical pages/s | logical GB/s | socket balance | selected pages/shard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| local | 10,843,512 | 11,655,531 | 11,655,531 | 2,745,477,662.064474 | 87.855285 | 0.930332 | 1,145 |
| interleave | 11,930,427 | 14,938,507 | 14,938,507 | 2,142,115,005.200988 | 68.547680 | 0.798636 | 1,145 |
| remote | 18,966,276 | 20,915,807 | 20,915,807 | 1,529,943,358.150130 | 48.958187 | 0.906791 | 1,145 |

Derived comparisons from pair medians:

- local vs interleave: 1.281667× faster; 21.98% lower pair latency; 28.17% higher logical throughput.
- local vs remote: 1.794496× faster; 44.27% lower pair latency; 79.45% higher logical throughput.

The selected-page count remains exactly 1,145 for both mirrored shards under every placement treatment, so the treatment changed placement/performance rather than candidate semantics.

`logical GB/s` is derived from packed signature bytes divided by elapsed time. It is **not** a measured DRAM-bandwidth counter.

## Decision

The next BKV-K4 gate is true socket-local sharding rather than interleaving:

1. replicate the tiny Boolean query to both sockets;
2. keep disjoint global page-ID ranges and packed signatures local to each NUMA node;
3. scan each shard with the physical cores local to that node;
4. merge candidate IDs deterministically;
5. require exact equality against a monolithic deterministic oracle before accepting any performance result.

This evidence supports socket-local placement on this T430 for the tested 1 GiB-class Boolean scan. It does not establish a universal NUMA policy for other hardware or working-set sizes.
