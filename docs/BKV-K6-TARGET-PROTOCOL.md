# BKV-K6 target-host/model protocol

Status: preregistration/provenance infrastructure only.

`kvlab.bikv-target-protocol.v1` freezes the identities and measurement obligations
for a target-host/model BIKV campaign before any qualifying measurements are run.
A valid protocol is **not** a performance, quality, traffic or promotion result and
does not unblock BKV-K9 by itself.

The protocol binds the exact KVLab and FLAT commits, retained BIKV evidence bundle,
model/revision, tokenizer/revision, runtime/revision, hardware fingerprint,
precision, context, batch, dataset/revision/partition, seeds, warmups, repetitions,
Boolean policy and timing/byte evidence kinds. Model, tokenizer, runtime and dataset
revisions use full 40-hex immutable revisions rather than mutable tags.

The dense baseline is fixed to `full-cache/native-prefill`. The metric obligation is
also fixed and cannot be shortened by a campaign file. It covers the complete
roadmap surface needed by BKV-K5/K6 qualification:

- candidate density, recall, false-negative rate, O/LSE error, downstream quality,
  and reset/reuse correctness;
- Boolean bits/token and bits/page, total index bytes, metadata overhead,
  numerical K/V bytes touched/avoided, Boolean bytes read, host/device transfers,
  measurable NUMA traffic, fragmentation, and allocator overhead;
- BKV-K6 query-signature and candidate-bitmap transfer bytes, synchronization
  wait, dispatch count, and backpressure wait;
- Boolean search/front-end latency, first-token latency, TPOT, tokens/s, pages/s,
  bits compared/s, effective bandwidth, and scaling efficiency.

A retained result may explicitly report a metric as unavailable/not exposed where
its measurement surface truly does not exist, but the protocol cannot silently
remove that obligation. Energy remains separate and may be reported only when
measured inputs are known, as required by the roadmap.

Byte accounting carries one declared evidence kind. Logical packed-byte accounting,
host-observed transfers, device-observed transfers and physical DRAM counters remain
distinct surfaces. A logical-byte result must never be rewritten as physical traffic.

Protected holdout partitions cannot permit tuning. Confirmatory/final phases cannot
permit tuning regardless of partition name. Negative, no-effect and failed runs must
remain in the retained campaign evidence; this protocol does not define a filter that
can discard them.

`tools/verify_bikv_target_protocol.py` accepts only canonical JSON and prints the
content-derived protocol SHA-256. Per-attempt result retention is now defined
separately by `kvlab.bikv-target-run.v1`; campaign execution remains separate work.
BKV-K9 stays blocked until actual retained target-host/model evidence satisfies the
gate in issue #111.
