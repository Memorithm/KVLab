# BKV-K6 target-run retention contract

Status: **evidence-retention infrastructure only; no target-host result is recorded by this document.**

`kvlab.bikv-target-run.v1` records one baseline or candidate attempt under an
exact `kvlab.bikv-target-protocol.v1` identity. It closes the result-retention
schema gap without implementing a model/runtime campaign or deciding BKV-K9
promotion.

Every run binds the content-derived protocol SHA-256, campaign identity, frozen
seed, repetition index and baseline/candidate role. Validation against the
protocol rejects foreign seeds, excess repetition indices, a mismatched campaign
identity or a mismatched protocol digest.

The result cannot shorten the preregistered metric surface. Its metric array must
contain every `REQUIRED_METRICS` entry in the exact frozen protocol order. Each
entry is explicitly one of:

- `measured`: carries one finite non-negative numeric or Boolean observation and
  an explicit unit;
- `not_exposed`: carries no value and a required reason;
- `failed`: carries no value and a required failure reason.

A completed attempt cannot hide a failed metric. A failed attempt must retain at
least one failed metric and a top-level failure reason. This preserves failed and
negative-capable evidence instead of permitting unsuccessful attempts to vanish
from the campaign record. `not_exposed` remains distinct from zero and from a
failed measurement.

`tools/verify_bikv_target_run.py PROTOCOL RUN` accepts only canonical JSON,
recomputes both identities, validates the run against the supplied frozen
protocol and reports only provenance/status counts plus the run SHA-256. It does
not convert missing metrics into measurements or infer a scientific verdict.

## Non-claims

This schema does not execute a model, expose unavailable telemetry, estimate
physical traffic, compare uncertainty intervals, apply the H0/H1 decision rule,
or qualify BIKV. In particular, a canonical run record is not evidence of lower
TTFT/TPOT, higher tokens/s, lower physical DRAM/HBM/NUMA traffic, preserved model
quality, or a favorable `numerical_KV_bytes_avoided / Boolean_KV_bytes_read`
ratio. Those claims require actual retained observations under the frozen
protocol and the BKV-K9 gate in issue #111 remains closed until that evidence
exists.

## Campaign completeness manifest

`kvlab.bikv-target-campaign.v1` binds the complete retained attempt set for one
frozen target protocol. A valid manifest requires exactly one `baseline` and one
`candidate` record for every frozen `(seed, repetition_index)` slot, assigns each
slot the exact `kvlab.bikv-target-run.v1` SHA-256, rejects duplicate attempt IDs
or reused run payloads, and canonicalizes ordering independently of filesystem or
execution order. Failed attempts remain valid *retained slots* rather than being
dropped; their failed status is preserved by the referenced run hash and does
not become a successful measurement.

The manifest is a completeness/provenance gate only. It computes no aggregate,
uncertainty interval, H0/H1 verdict, performance ratio, recall result, quality
result or promotion decision. Those interpretations remain blocked until the
referenced target-host/model measurements exist and satisfy their separately
preregistered decision rules.

`tools/verify_bikv_target_campaign.py PROTOCOL CAMPAIGN RUN...` verifies canonical
manifest encoding, exact protocol identity, every retained run hash and complete
baseline/candidate slot coverage. Its output is limited to provenance/status counts
and content identities; a failed retained attempt remains failed evidence.
