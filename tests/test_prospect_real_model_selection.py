import json
import unittest

from kvlab.prospect_real_model_eviction import ObservedMetric
from kvlab.prospect_real_model_selection import (
    PROSPECT_KV_REAL_MODEL_SELECTION_SCHEMA_V1,
    ProspectKvRealModelSelectionError,
    ProspectKvRealModelSelectionEvidenceV1,
)
from kvlab.prospect_selection_handoff import ProspectKvSelectionHandoffV1


def _selection(*, policy: str, retained: tuple[int, ...]) -> ProspectKvSelectionHandoffV1:
    return ProspectKvSelectionHandoffV1.capture(
        token_ids=(10, 11, 12, 13, 14),
        bytes_per_token=64,
        policy=policy,
        retained_token_ids=retained,
    )


def _metrics(candidate_accuracy: float = 0.78) -> tuple[ObservedMetric, ...]:
    return (
        ObservedMetric.capture(
            name="token_accuracy",
            kind="quality",
            unit="ratio",
            preference="higher_is_better",
            baseline_value=0.8,
            candidate_value=candidate_accuracy,
        ),
        ObservedMetric.capture(
            name="logit_l2",
            kind="numerical",
            unit="l2",
            preference="lower_is_better",
            baseline_value=0.0,
            candidate_value=0.125,
        ),
    )


def _evidence(
    *,
    policy: str = "lru",
    retained: tuple[int, ...] = (10, 12, 14),
    candidate_hash: str = "3",
) -> ProspectKvRealModelSelectionEvidenceV1:
    return ProspectKvRealModelSelectionEvidenceV1.capture(
        experiment_id="real-model-selection-c1",
        run_repository_revision="a" * 40,
        model_id="example/model",
        model_revision="model-r1",
        tokenizer_revision="tok-r1",
        runtime_backend="fixture-runtime",
        runtime_revision="runtime-r1",
        evaluation_id="holdout-001",
        trace_sha256="1" * 64,
        seed=7,
        selection=_selection(policy=policy, retained=retained),
        baseline_output_sha256="2" * 64,
        candidate_output_sha256=candidate_hash * 64,
        metrics=_metrics(),
    )


class ProspectKvRealModelSelectionTests(unittest.TestCase):
    def test_capture_round_trips_observed_explicit_selection(self) -> None:
        evidence = _evidence()
        self.assertEqual(evidence.schema, PROSPECT_KV_REAL_MODEL_SELECTION_SCHEMA_V1)
        self.assertEqual(evidence.selection.policy, "lru")
        self.assertEqual(evidence.selection.retained_token_ids, (10, 12, 14))
        self.assertEqual(evidence.baseline_logical_kv_bytes, 320)
        self.assertEqual(evidence.candidate_logical_kv_bytes, 192)
        self.assertAlmostEqual(evidence.metrics[0].delta, -0.02)

        payload = evidence.canonical_json()
        self.assertEqual(
            ProspectKvRealModelSelectionEvidenceV1.from_canonical_json(payload), evidence
        )

    def test_same_budget_can_represent_distinct_policy_selections_without_ranking(self) -> None:
        lru = _evidence(policy="lru", retained=(10, 12, 14), candidate_hash="3")
        magnitude = _evidence(
            policy="magnitude",
            retained=(11, 13, 14),
            candidate_hash="4",
        )

        self.assertEqual(lru.candidate_logical_kv_bytes, 192)
        self.assertEqual(magnitude.candidate_logical_kv_bytes, 192)
        self.assertNotEqual(lru.selection.retained_token_ids, magnitude.selection.retained_token_ids)
        self.assertNotEqual(lru.selection.policy, magnitude.selection.policy)

    def test_decoder_rejects_tampered_metric_delta_and_logical_bytes(self) -> None:
        raw = json.loads(_evidence().canonical_json())
        raw["metrics"][0]["delta"] = 0.5
        tampered = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        with self.assertRaises(ProspectKvRealModelSelectionError):
            ProspectKvRealModelSelectionEvidenceV1.from_canonical_json(tampered)

        raw = json.loads(_evidence().canonical_json())
        raw["candidate_logical_kv_bytes"] = 64
        tampered = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        with self.assertRaises(ProspectKvRealModelSelectionError):
            ProspectKvRealModelSelectionEvidenceV1.from_canonical_json(tampered)

    def test_decoder_rejects_bad_provenance_and_noncanonical_json(self) -> None:
        raw = json.loads(_evidence().canonical_json())
        raw["run_repository_revision"] = "deadbeef"
        tampered = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        with self.assertRaises(ProspectKvRealModelSelectionError):
            ProspectKvRealModelSelectionEvidenceV1.from_canonical_json(tampered)

        raw = json.loads(_evidence().canonical_json())
        raw["seed"] = 2**64
        tampered = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        with self.assertRaises(ProspectKvRealModelSelectionError):
            ProspectKvRealModelSelectionEvidenceV1.from_canonical_json(tampered)

        noncanonical = json.dumps(
            json.loads(_evidence().canonical_json()), sort_keys=True, indent=2
        )
        with self.assertRaises(ProspectKvRealModelSelectionError):
            ProspectKvRealModelSelectionEvidenceV1.from_canonical_json(noncanonical)

    def test_policy_label_does_not_claim_heuristic_replay(self) -> None:
        evidence = _evidence(policy="learned-policy-v7", retained=(10, 13, 14))
        self.assertEqual(evidence.selection.policy, "learned-policy-v7")
        self.assertEqual(evidence.selection.retained_token_ids, (10, 13, 14))


if __name__ == "__main__":
    unittest.main()
