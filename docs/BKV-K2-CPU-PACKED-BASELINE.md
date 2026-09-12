# BKV-K2 — Packed CPU baseline

Status: executable systems baseline; no T430 result has been collected yet.

## Purpose

BKV-K2 establishes the exact single-process packed-bit scan that later Rust/SIMD, multicore, NUMA and GPU implementations must reproduce before they can be compared for performance.

The baseline scans every page, computes exact packed Hamming distance using native integer `bit_count`, and emits candidates in logical page order. It deliberately does not hide page-scan cost behind an index.

## Timing boundary

The benchmark includes:

- traversal of every packed page signature;
- XOR + population count for every packed word;
- threshold comparison;
- construction of the accepted page-id tuple.

It excludes:

- signature generation;
- loading signatures from disk;
- numerical K/V access;
- FLAT attention;
- inter-process/network transfer.

Those costs enter later BIKV end-to-end campaigns.

## Required metrics

The runner records:

- median/min/max query latency;
- pages scanned;
- signature bits per page;
- pages/s;
- bits compared/s;
- queries/s;
- accepted page ids;
- requested/applied CPU affinity.

Do not interpret Python baseline rates as the ceiling of the CPU architecture. BKV-K2 is the deterministic reference and measurement protocol; optimized Rust/SIMD paths must be compared separately.

## T430 first run

From a clean KVLab checkout on the server:

```bash
python tools/capture_bkv_host.py > t430-bkv-host.json
python tools/benchmark_bkv_cpu.py \
  --pages 100000 \
  --signature-bits 256 \
  --max-distance 96 \
  --warmup 5 \
  --queries 25 \
  --cpu 0 \
  > t430-bkv-k2-cpu0.json
```

The numerical parameters above are a smoke/baseline configuration, not a scientific optimum. Later campaign manifests must freeze their own signature width, density/threshold target and sweep sizes.

If `numactl` is absent, host capture records it as unavailable rather than guessing NUMA topology. Install/enable the normal system NUMA tooling before BKV-K4 qualification.

## Exit gate

BKV-K2 is complete when the packed scan and benchmark are tested, reproducible from frozen synthetic inputs, and at least one target host evidence pack exists. BKV-K3 may implement multicore scaling before the physical evidence pack is available, but no scaling claim may be made without it.
