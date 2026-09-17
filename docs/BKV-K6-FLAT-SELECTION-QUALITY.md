# BKV-K6 FLAT declared-target selection quality retention

Status: consumer pinned to the qualified FLAT-ATTENTION #252 merge; no scientific or runtime result.

FLAT-ATTENTION #252 merged as `dfd5fb7a90242b5e95c3946186864f56c83930bd` and provides `flat.bikv-selection-quality.v1`. The producer binds one validated Boolean page-selection decision to an independently declared non-empty dense-reference target page set and emits exact page-level true-positive, false-negative, false-positive, recall/FNR and candidate-density counts.

KVLab's `flat_bikv_selection_quality` consumer retains the exact canonical producer bytes, derives SHA-256 identities for the complete quality record and embedded selection, reuses the existing fail-closed FLAT selection validator, and recomputes every target-set metric instead of trusting producer aggregates. It also verifies the producer outer `quality_checksum`, which covers the embedded selection, declared target set, and all derived metrics. It rejects duplicate JSON keys, non-canonical bytes, malformed or empty target sets, invalid embedded selection evidence, booleans masquerading as integer metrics, out-of-range/duplicate/non-monotone targets and any scalar/fraction drift.

The FLAT producer revision above is the immutable qualified merge revision. This KVLab consumer still requires its own exact-head qualification before merge; producer qualification does not substitute for consumer qualification.

The retained target pages remain an experimental input owned by the preregistered KVLab protocol. This consumer does not decide how the dense target is created, does not choose an acceptance threshold and does not reinterpret target-page recall as downstream model quality. It also does not establish physical DRAM/HBM/cache/PCIe traffic reduction, latency/TTFT/TPOT improvement, energy reduction, speedup or readiness for adaptive BKV tiering.
