# BKV-K6 — FLAT M13B.4 trace evidence consumer

Status: **candidate infrastructure; no target-host result**.

KVLab consumes the producer-owned FLAT-ATTENTION schema `flat.m13b4-trace.v1` as opaque-but-structurally-validated timing evidence for the Boolean KV programme. The pinned producer reference is FLAT-ATTENTION merge `29c18275b5687b71599ef11f5badcc4571a52a41` (PR #249).

The KVLab consumer deliberately mirrors only the producer contract needed to retain trustworthy bytes:

- exact canonical JSON bytes are required, including producer field order and enum spellings;
- timing source, scheduling variant, trace scope, event identities and `u64` timestamps are validated fail-closed;
- duplicate events, backwards timestamps, missing mandatory events and incomplete synchronization evidence are rejected;
- multi-dispatch candidates require an explicit synchronization wait pair;
- prefill evidence requires numerical-KV commit and Boolean-signature commit to precede joint decode visibility;
- the exact canonical bytes receive a SHA-256 identity for later evidence-bundle binding.

The consumer does **not** infer overlap or concurrency from event names, does not convert timestamps into a speedup claim, does not infer physical K/V traffic from logical events and does not promote a trace into a scientific verdict. Cross-unit overlap still requires correlated traces in one declared timing domain. Performance/quality interpretation remains subject to the preregistered M13B.4/BKV-K6 protocol and target-host measurement obligations.

CLI validation:

```bash
python tools/verify_flat_m13b4_trace.py TRACE.json
```

A successful CLI result reports only structural/provenance fields and the canonical trace SHA-256. It is not a benchmark result.
