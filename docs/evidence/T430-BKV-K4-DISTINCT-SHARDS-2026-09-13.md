# T430 BKV-K4 distinct socket-local shard evidence — 2026-09-13

Status: **verified hardware evidence for this host and exact commit only**.

## Provenance

- Host: Dell PowerEdge T430
- CPU: 2 × Intel Xeon E5-2683 v4
- Physical cores: 16 per socket, 32 total
- Logical CPUs: 64 with SMT
- NUMA nodes: 2
- Verified NUMA distance: local 10, remote 21
- Git head: `7e02edc1e3ac3b53cfb352cfad560db439755a0f`
- Tracked tree dirty: `false`
- Pre-existing untracked files: 96
- UTC run time: `2026-09-13T07:58:28Z`
- Benchmark schema: `bkv-k4-distinct-dual-local-v1`

## Frozen campaign

- Corpus mode: distinct contiguous global page ranges
- Total pages: 32,000,000
- Pages per socket-local shard: 16,000,000
- Signature width: 256 bits
- Max Hamming distance: 96
- Workers per local shard: 16 physical cores
- Monolithic comparator workers: 32 physical cores
- Warmup: 5
- Repetitions: 25
- Seed: 1

The Boolean query is replicated to both NUMA-local shards. Shard 0 owns the first global page-ID range and shard 1 owns the second disjoint range. Independently allocated shards reconstruct the same deterministic page stream as the monolithic comparator. The merged candidate IDs are compared against the monolithic candidate IDs exactly.

## Result

| Metric | Value |
|---|---:|
| Monolithic interleaved median | 15,111,998 ns |
| Shard 0 local median | 10,831,660 ns |
| Shard 1 local median | 12,133,153 ns |
| Dual-local pair median | 12,133,153 ns |
| Speedup vs monolithic interleave | **1.245513×** |
| Pair latency reduction vs monolithic interleave | **19.71%** |
| Dual-local pages/s | 2,637,401,836.109707 |
| Dual-local logical packed-signature GB/s | 84.396859 |
| Socket balance | 0.892732 |
| Selected global pages | 2,283 |
| Candidate equality | **exact** |

`logical packed-signature GB/s` is a derived traffic metric from page count × packed signature bytes divided by elapsed time. It is **not** a direct DRAM-bandwidth measurement.

## Interpretation

For this T430, this campaign validates the BKV-K4 architecture:

`replicated Boolean query -> NUMA0-local BKV shard + NUMA1-local BKV shard -> deterministic candidate merge`

The critical correctness gate passed: the merged global candidate-ID list is exactly identical to the monolithic oracle, while the dual-local pair is 1.245513× faster than the 32-physical-core interleaved monolithic comparator for this 32M-page / 256-bit campaign.

This result is consistent with the immediately preceding mirrored-placement gate at commit `46b1d479898eced73c0e26ed0bf6f949b6d81085`, where local placement measured 11.655531 ms, interleave 14.938507 ms, and remote 20.915807 ms with identical candidate counts. The mirrored gate isolated placement; the present gate adds disjoint global page identities and exact merged-candidate equivalence.

## Promoted design rule — scoped

For large Boolean-KV scans on this dual-socket T430, prefer socket-local sharding and replicate the compact Boolean query rather than interleaving a monolithic BKV corpus across NUMA nodes. Candidate results must be merged deterministically before numerical KV consumption.

This rule is **host- and workload-qualified**, not a universal NUMA claim. It must be requalified for other CPUs, page sizes, signature widths, candidate densities, thresholds, and production attention workloads.

## Next gate

BKV-K5 / BIKV integration should preserve the exact candidate IDs produced by this Boolean plane and use them only to select authoritative numerical K/V pages. Required measurements include Boolean metadata bytes read, numerical KV bytes avoided, selection/merge latency, staging/synchronization cost, TTFT/TPOT, dense-recall/false-negative rate, and downstream numerical output/LSE parity or bounded error as appropriate.