# BKV-K6 FLAT M13B.5 page-selection evidence

Status: consumer contract only; no benchmark or performance conclusion.

FLAT-ATTENTION owns the Boolean KV page router. KVLab owns the retained evidence
and experimental interpretation. The producer candidate currently emits
`flat.boolean-kv-selection.v1` from its transparent `BooleanIndexedKvSelection`
record.

KVLab accepts only the exact canonical byte encoding from the pinned producer
revision. It verifies the producer's FNV-1a checksum, strict logical-page order,
unique physical pages, Hamming/XNOR conservation, exact packed Boolean-byte
geometry, selected/live-token bounds, and numerical byte totals rederived from
the retained K+V bytes/token geometry. KVLab then derives a SHA-256 content
identity over the exact canonical producer bytes for bundle retention.

The FNV checksum is not cryptographic attestation. The producer record contains
logical/storage accounting only. Neither FLAT nor KVLab may reinterpret
`avoided_numerical_kv_bytes` as measured DRAM, cache-line, PCIe or HBM traffic,
and the record contains no latency, model-quality, residency or speedup claim.

Until FLAT PR #250 is exact-head green and merged, the KVLab consumer remains a
candidate pinned to its source head SHA and must not be promoted as final
cross-repository provenance. After merge, repin the consumer to the FLAT merge
SHA and requalify its own exact head.
