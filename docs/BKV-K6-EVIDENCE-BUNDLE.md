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

`tools/verify_bikv_evidence_bundle.py` adds a fail-closed command-line boundary for
that replay step. The command requires one canonical receipt file for every role
in the canonical bundle, rejects missing/extra/duplicate roles, validates each
receipt's canonical encoding, and checks the exact receipt/provenance identity
against the bound entry. Its success output contains provenance fields only; it
never opens or reports the upstream scientific payload.

This supports the current BKV-K6 gate of retaining and cross-linking FLAT BIKV
evidence (for example a K6.4 evidence envelope and an M13B.4 trace receipt) inside
KVLab provenance without turning those artifacts into KVLab measurements.

## Non-claims

A bundle or a successful verifier run proves only provenance linkage between exact
receipts. It does **not** validate Boolean overhead, numerical K/V bytes avoided,
physical DRAM traffic, `numerical_KV_bytes_avoided / Boolean_KV_bytes_read`,
first-token latency, TPOT, recall/false-negative rate, downstream O/LSE/model
quality, energy, or speedup. Those remain target-host/model measurement gates.
