# BIKV evidence receipt

`kvlab.bikv-evidence-receipt.v1` is a provenance-only contract for retaining or cross-linking an upstream BIKV JSON evidence artifact without importing its scientific interpretation into KVLab.

The receipt binds:

- the producer repository in `owner/name` form;
- the exact 40-hex producer commit;
- SHA-256 of the source bytes exactly as received;
- the exact source byte length.

The builder accepts only non-empty UTF-8 JSON with a top-level object. It parses the payload only to enforce that structural boundary. The source digest is computed over the original bytes, not over a reserialized JSON object, so whitespace or ordering changes are visible provenance changes even when the parsed JSON values are equivalent.

The receipt itself has deterministic canonical JSON bytes and a separate receipt SHA-256. Source-artifact identity and receipt identity must not be conflated.

## What this contract does not establish

A valid receipt does **not** validate or promote any upstream latency, bandwidth, numerical-KV traffic, model-quality, correctness, energy, resident-only execution, first-token, TPOT, or speedup claim. Those statements require their own declared evidence and qualification. The receipt only says which exact upstream bytes and producer commit KVLab retained or referenced.

This contract is intended to support the existing BKV-K6 evidence-retention gate. Target-host/model measurements and their preregistered decision rules remain separate work.
