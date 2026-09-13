# BKV-K4 dual-NUMA placement gate

This experiment isolates NUMA placement cost on the verified Dell T430 before introducing semantically distinct Boolean-KV shard identities.

## Question

For the same deterministic Boolean query and the same deterministic packed page corpus, does keeping each socket's working set in its local NUMA node outperform interleaved or deliberately remote placement when both sockets scan concurrently?

## Hardware contract

The protocol is specific to the captured T430 topology:

- 2 × Intel Xeon E5-2683 v4;
- 16 physical cores per socket, 2 SMT threads per core;
- node 0 physical-thread representatives: `0,2,...,30`;
- node 1 physical-thread representatives: `1,3,...,31`;
- local NUMA distance 10, remote distance 21.

The runner must fail rather than silently reinterpret a different topology. Hardware evidence is captured with each campaign.

## Corpus control

This is a **placement microbenchmark**, not the final production sharding scheme.

Both socket processes intentionally receive the same query and same page corpus. The data are mirrored but physically allocated independently. This makes the three treatment arms differ only by NUMA placement:

1. `local`: socket 0 CPU + node 0 RAM; socket 1 CPU + node 1 RAM;
2. `interleave`: each process pinned to one socket, memory interleaved across nodes 0 and 1;
3. `remote`: socket 0 CPU + node 1 RAM; socket 1 CPU + node 0 RAM.

Each child benchmark independently validates its parallel candidate set against the scalar oracle before timing. The runner also requires identical selected-page counts for the mirrored shards.

Because the corpus is mirrored, `2 × shard_pages` is a logical throughput accounting convention for the concurrent pair; it is not a claim that the two shards contain distinct semantic page identities.

## Metrics

The runner records:

- median latency for each socket process;
- pair latency = `max(node0_median, node1_median)`;
- logical pages/s for the concurrent pair;
- logical packed-signature GB/s;
- socket balance = `min(median0, median1) / max(median0, median1)`;
- selected pages per mirrored shard;
- commit SHA, tracked-tree dirtiness, pre-existing untracked-file count, topology and meminfo.

Logical GB/s is derived from packed signature bytes and must not be described as measured DRAM bandwidth.

## Decision rule

The experiment establishes a NUMA-placement preference only if the ordering is stable across repeated campaigns and the magnitude exceeds run-to-run noise. A single faster median is insufficient.

If socket-local placement is consistently superior to interleave, the next implementation step is a true dual-shard layout with distinct global page identities, replicated query bits, exact deterministic candidate merge, and equality against a monolithic oracle. If not, retain interleave as a viable baseline and investigate worker synchronization and memory-controller saturation before adding shard complexity.

## Runner

```bash
bash scripts/run_t430_bkv_k4_dual_numa_placement.sh
```

The default uses 16 million mirrored pages per socket, 256-bit signatures, 16 physical-core workers per socket, five warmups and 25 measured queries.
