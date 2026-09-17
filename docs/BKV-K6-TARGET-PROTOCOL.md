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
also fixed and cannot be shortened by a campaign file:

- first-token latency;
- steady-state TPOT and tokens/s;
- Boolean-front-end time;
- numerical K/V bytes avoided;
- Boolean K/V bytes read;
- candidate recall and false-negative rate;
- O and LSE error;
- downstream quality/correctness under a preregistered metric/rule.

Byte accounting carries one declared evidence kind. Logical packed-byte accounting,
host-observed transfers, device-observed transfers and physical DRAM counters remain
distinct surfaces. A logical-byte result must never be rewritten as physical traffic.

Protected holdout partitions cannot permit tuning. Confirmatory/final phases cannot
permit tuning regardless of partition name. Negative, no-effect and failed runs must
remain in the retained campaign evidence; this protocol does not define a filter that
can discard them.

`tools/verify_bikv_target_protocol.py` accepts only canonical JSON and prints the
content-derived protocol SHA-256. Campaign execution and result schemas remain
separate work; BKV-K9 stays blocked until actual retained target-host/model evidence
satisfies the gate in issue #111.
