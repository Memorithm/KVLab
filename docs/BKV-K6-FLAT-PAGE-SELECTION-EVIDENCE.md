# BKV-K6 FLAT M13B.5 page-selection evidence

Status: consumer contract only; no benchmark or performance conclusion.

FLAT-ATTENTION owns the Boolean KV page router. KVLab owns the retained evidence
and experimental interpretation. The qualified FLAT #250 producer, merged as
`315716b1bf6c42cf5439b351aa9c1c3e7a1644ba`, emits `flat.boolean-kv-selection.v1`
from its transparent `BooleanIndexedKvSelection` record.

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

KVLab is pinned to the qualified FLAT #250 merge SHA above. This consumer still
requires its own exact-head KVLab qualification before promotion or merge; the
producer merge does not itself qualify this downstream implementation.
