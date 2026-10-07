# BKV-K4 distinct-shard paired latency protocol

This protocol replaces the v1 estimate that used the maximum of independently
aggregated shard medians. It is the minimum evidence contract for any renewed
T430 performance statement. The existing v1 exact-candidate result remains a
valid correctness result.

## Measurement unit

Each repetition produces one monolithic observation and one concurrent
dual-shard observation for the same frozen corpus, query, threshold and seed.
The dual-shard wall clock starts before the first child process is launched and
ends only after both processes complete and their candidate IDs are merged and
compared with the frozen oracle. Therefore:

`pair_e2e_ns = pair_scan_wall_ns + merge_ns`

The child-reported shard scan times are diagnostic only. They are never used to
derive pair latency, throughput or speedup.

## Bias controls and fail-closed gates

- Run from a tracked-clean exact Git revision and write into a new output path.
- Use at least three repetitions.
- Alternate monolithic-first and dual-first execution every repetition.
- Alternate shard-0-first and shard-1-first launch order every repetition.
- Preserve every raw per-repetition benchmark CSV and candidate-ID file.
- Require exact monolithic and merged dual candidate equality in every sample.
- Reject missing, non-positive, reordered, non-contiguous or schema-incompatible
  samples before summarization.
- Retain a manifest, raw paired-sample CSV, summary and SHA-256 inventory.

The frozen oracle is generated once before the timed campaign and is excluded
from both latency distributions. The order alternation mitigates systematic
thermal, cache and first-run bias; it does not turn one host into a general
hardware claim.

## Summary semantics

`kvlab.t430_pair_summary` computes the monolithic and dual end-to-end medians
from the paired raw rows. `paired_speedup_vs_monolithic` is their ratio. Logical
pages/s and packed-signature GB/s are derived from dual end-to-end latency and
remain logical workload rates, not measured DRAM bandwidth.

A successful run qualifies only the exact recorded host, topology, Git revision
and workload. It does not establish attention latency, TTFT/TPOT, numerical-KV
traffic reduction, model quality, or production speedup.
