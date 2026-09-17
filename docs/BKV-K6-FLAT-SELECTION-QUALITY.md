# BKV-K6 FLAT declared-target selection quality retention

Status: candidate consumer for FLAT-ATTENTION #252; no scientific or runtime result.

FLAT-ATTENTION candidate head `148c73eaf21096a0b0cc539b9a474387aa6c7af0` introduces `flat.bikv-selection-quality.v1`. The producer binds one validated Boolean page-selection decision to an independently declared non-empty dense-reference target page set and emits exact page-level true-positive, false-negative, false-positive, recall/FNR and candidate-density counts.

KVLab's `flat_bikv_selection_quality` consumer retains the exact canonical producer bytes, derives SHA-256 identities for the complete quality record and embedded selection, reuses the existing fail-closed FLAT selection validator, and recomputes every target-set metric instead of trusting producer aggregates. It also verifies the producer outer `quality_checksum`, which covers the embedded selection, declared target set, and all derived metrics. It rejects duplicate JSON keys, non-canonical bytes, malformed or empty target sets, invalid embedded selection evidence, booleans masquerading as integer metrics, out-of-range/duplicate/non-monotone targets and any scalar/fraction drift.

The FLAT producer revision above is a **candidate**, not a source-of-truth merge revision. This KVLab branch must remain draft until FLAT #252 finishes exact-head qualification and merges. The consumer must then be pinned to that final FLAT merge SHA and requalified on its own exact head before merge.

The retained target pages remain an experimental input owned by the preregistered KVLab protocol. This consumer does not decide how the dense target is created, does not choose an acceptance threshold and does not reinterpret target-page recall as downstream model quality. It also does not establish physical DRAM/HBM/cache/PCIe traffic reduction, latency/TTFT/TPOT improvement, energy reduction, speedup or readiness for adaptive BKV tiering.
