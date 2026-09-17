# BKV-K6 — FLAT M13B.4 trace evidence consumer

Status: **candidate infrastructure; no target-host result**.

KVLab consumes the producer-owned FLAT-ATTENTION schema `flat.m13b4-trace.v1` as opaque-but-structurally-validated timing evidence for the Boolean KV programme. The pinned producer reference is FLAT-ATTENTION merge `29c18275b5687b71599ef11f5badcc4571a52a41` (PR #249).

The KVLab consumer preserves the producer encoding and structural/order invariants needed to retain trustworthy bytes, then applies one deliberately stricter evidence-retention rule: synchronization evidence must be a complete pair whenever present, and every `multi_dispatch_overlap_candidate` must retain an explicit pair even for prefill scope. This extra consumer gate does not change FLAT's producer schema; it only prevents KVLab from content-addressing ambiguous synchronization evidence as admissible BKV evidence.

- exact canonical JSON bytes are required, including producer field order and enum spellings;
- timing source, scheduling variant, trace scope, event identities and `u64` timestamps are validated fail-closed;
- duplicate events, backwards timestamps, missing mandatory events and incomplete synchronization evidence are rejected;
- multi-dispatch candidates require an explicit synchronization wait pair in every scope;
- prefill evidence requires numerical-KV commit and Boolean-signature commit to precede joint decode visibility;
- decode synchronization waits must remain inside the declared query-to-output observation unit;
- the exact canonical bytes receive a SHA-256 identity for later evidence-bundle binding.

The consumer does **not** infer overlap or concurrency from event names, does not convert timestamps into a speedup claim, does not infer physical K/V traffic from logical events and does not promote a trace into a scientific verdict. Cross-unit overlap still requires correlated traces in one declared timing domain. Performance/quality interpretation remains subject to the preregistered M13B.4/BKV-K6 protocol and target-host measurement obligations.

CLI validation:

```bash
python tools/verify_flat_m13b4_trace.py TRACE.json
```

A successful CLI result reports only structural/provenance fields and the canonical trace SHA-256. It is not a benchmark result.
