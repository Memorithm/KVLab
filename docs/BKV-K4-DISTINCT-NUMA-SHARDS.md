# BKV-K4 distinct NUMA shard gate

This gate follows the mirrored-placement experiment. Its purpose is to test the actual Boolean-KV architecture rather than only memory placement.

## Contract

For a deterministic Boolean-KV corpus with global page IDs `0..N`:

- one tiny Boolean query is reproduced identically on both sockets;
- shard 0 owns global IDs `[0, N/2)` and is allocated/scanned on NUMA node 0;
- shard 1 owns global IDs `[N/2, N)` and is allocated/scanned on NUMA node 1;
- each shard uses the physical cores local to its socket;
- each shard validates its parallel candidate set against a scalar oracle over the same local data;
- the concatenated, globally ordered candidate IDs must be byte-for-byte identical to a monolithic oracle over the complete deterministic corpus.

Any candidate mismatch is a hard failure and invalidates the performance result.

## Deterministic stream identity

`bkv_distinct_shard_bench` generates the query first, then advances through the same deterministic page-signature stream used by the monolithic corpus. `page_offset` skips the exact number of packed signature words that precede the shard. This permits independent processes to allocate separate NUMA-local buffers while preserving the same global corpus and IDs.

## T430 treatment

Default campaign:

- total pages: 32,000,000
- signature width: 256 bits
- total logical signature payload: 1,024,000,000 bytes (~0.954 GiB)
- shard pages: 16,000,000 per NUMA node
- workers: 16 physical cores per socket, no SMT
- monolithic comparator: 32 physical-core representatives with `--interleave=0,1`
- threshold: Hamming distance <= 96
- seed: 1
- warmup: 5
- repetitions: 25

The runner fails closed if the verified two-socket/32-physical-core T430 topology is absent.

## Metrics

The summary reports monolithic latency, each shard latency, pair latency, logical pages/s, derived logical packed-signature GB/s, socket balance, speedup relative to the monolithic interleaved treatment, selected-page count, and exact candidate equality.

Logical GB/s is an algorithmic traffic rate, not a hardware DRAM-bandwidth counter.

## Promotion rule

Promote socket-local distinct sharding into the BIKV runtime design only if:

1. exact candidate equality holds;
2. repeated hardware runs remain stable enough to distinguish the treatment;
3. dual-local pair latency is non-inferior to the interleaved monolithic comparator for the relevant large working set;
4. the same ownership model remains compatible with deterministic candidate merge and downstream numerical-KV lookup.

SMT, huge pages, and adaptive shard sizing are later treatments and must not be folded into this gate without separate evidence.
