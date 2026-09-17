# BKV-K8 — first-token evidence record

Status: **measurement/provenance contract only; no performance result is recorded by this document.**

`kvlab.bkv-k8-first-token-observation.v1` is the KVLab-owned record for one
first-token readiness observation. It is intentionally downstream of the
retained BIKV evidence bundle and does not reimplement FLAT-ATTENTION scheduling,
page ownership, or M13B.4 trace semantics.

The record binds:

- exact BIKV evidence-bundle SHA-256;
- producer repository and exact Git commit;
- target-host hardware-fingerprint SHA-256;
- explicit timing source (`host_wall_clock` or `device_timestamp`);
- first-token latency and the raw steady-state latency sample vector;
- Boolean-front-end interval;
- logical numerical K/V bytes avoided and Boolean K/V bytes read;
- historical-signature-rebuild count;
- whether the harness observed first-token Boolean-route consumption.

When first-token Boolean-route consumption is asserted, historical signature
rebuilds must be zero. This is the BKV-K8 readiness boundary: metadata intended
to remove first-token history reconstruction cannot be described as ready while
that reconstruction is still present.

`numerical_to_boolean_bytes_ratio()` returns the exact rational
`numerical_KV_bytes_avoided / Boolean_KV_bytes_read` when the denominator is
non-zero. These are logical byte counters. The ratio is **not** a physical DRAM,
PCIe, intersocket, HBM, cache-line, energy, or bandwidth measurement.

This schema deliberately stores raw steady-state timing samples rather than
converting them into a preferred estimator. Experiment-specific warmup,
repetitions, uncertainty, baselines and decision rules remain in the frozen
campaign protocol.

## Protocol-bound verification

The v2 observation can now be verified fail-closed against both the retained
`kvlab.bikv-evidence-bundle.v1` and the frozen
`kvlab.bikv-target-protocol.v1`. The binding requires exact agreement on the
evidence-bundle SHA-256, target hardware fingerprint, timing source and both
byte-evidence kinds. The observation producer commit must also equal the frozen
FLAT revision and the same producer repository/commit pair must occur in the
retained evidence bundle. This prevents a syntactically valid first-token
observation from being replayed under a different target protocol, attached to
an unretained producer identity, or from silently mixing logical byte accounting
with a physical traffic counter.

`tools/verify_bikv_first_token_observation.py` performs this check on canonical
JSON inputs and emits only verified identities and recorded observation fields.
`tools/verify_bikv_first_token_target_run.py` additionally binds the observation
to one exact completed candidate `kvlab.bikv-target-run.v1`: first-token latency,
Boolean-front-end latency, numerical K/V bytes avoided and Boolean K/V bytes read
must all be explicitly measured with canonical units and match the observation
exactly. The verifier emits the content identities of the protocol, target run,
observation and evidence bundle; it does not infer TPOT from the raw steady-state
sample vector or calculate a winner. Neither verifier opens a holdout or promotes
BKV-K9/BKV-K11.

## Remaining qualification

Before BKV-K8 can support a promotion claim, a target-host/model campaign still
has to retain the exact upstream evidence bundle and hardware fingerprint and
measure, under one frozen protocol, at least first-token latency, steady-state
TPOT, Boolean overhead, numerical K/V bytes avoided, Boolean K/V bytes read,
candidate recall/false negatives and downstream numerical/model correctness.
Negative or no-effect outcomes remain valid evidence. BKV-K11 adaptive placement
is not authorized by this contract.
