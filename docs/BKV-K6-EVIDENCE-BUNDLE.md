# BKV-K6 evidence receipt bundle

Status: provenance infrastructure only.

`kvlab.bikv-evidence-bundle.v1` groups multiple existing
`kvlab.bikv-evidence-receipt.v1` objects under distinct declared roles. It is a
cross-link for evidence retention; it does not parse or reinterpret the upstream
scientific envelopes.

Each entry retains:

- a canonical role;
- the SHA-256 of the canonical receipt;
- producer repository and exact commit;
- source-artifact SHA-256 and byte length.

Bundles are non-empty, sorted by role, reject duplicate roles, reject reuse of one
receipt under multiple roles, and have their own canonical JSON SHA-256 identity.
Replay can verify a supplied receipt against the exact identity recorded in the
bundle before loading any upstream artifact.

This supports the current BKV-K6 gate of retaining and cross-linking FLAT BIKV
evidence (for example a K6.4 evidence envelope and an M13B.4 trace receipt) inside
KVLab provenance without turning those artifacts into KVLab measurements.

## Non-claims

A bundle proves only provenance linkage between exact receipts. It does **not**
validate Boolean overhead, numerical K/V bytes avoided, physical DRAM traffic,
`numerical_KV_bytes_avoided / Boolean_KV_bytes_read`, first-token latency, TPOT,
recall/false-negative rate, downstream O/LSE/model quality, energy, or speedup.
Those remain target-host/model measurement gates.
