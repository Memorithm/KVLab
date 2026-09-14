import json
import math
import unittest

from kvlab.eviction import EvictionPolicy
from kvlab.prospect_eviction_handoff import ProspectKvEvictionHandoffV1
from kvlab.prospect_real_model_eviction import (
    ObservedMetric,
    ProspectKvRealModelEvidenceError,
    ProspectKvRealModelEvictionEvidenceV1,
)


class ProspectRealModelEvictionEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.eviction = ProspectKvEvictionHandoffV1.capture(
            token_ids=[10, 11, 12, 13, 14],
            bytes_per_token=64,
            policy=EvictionPolicy(max_tokens=3),
        )
        self.metrics = (
            ObservedMetric.capture(
                name="logit_l2",
                kind="numerical",
                unit="l2",
                preference="lower_is_better",
                baseline_value=0.0,
                candidate_value=0.125,
            ),
            ObservedMetric.capture(
                name="token_accuracy",
                kind="quality",
                unit="ratio",
                preference="higher_is_better",
                baseline_value=0.80,
                candidate_value=0.78,
            ),
        )

    def evidence(self) -> ProspectKvRealModelEvictionEvidenceV1:
        return ProspectKvRealModelEvictionEvidenceV1.capture(
            experiment_id="c1-real-model-fixture",
            run_repository_revision="a" * 40,
            model_id="example/model",
            model_revision="model-rev-1",
            tokenizer_revision="tokenizer-rev-1",
            runtime_backend="fixture-backend",
            runtime_revision="runtime-rev-1",
            evaluation_id="holdout-001",
            trace_sha256="1" * 64,
            seed=7,
            eviction=self.eviction,
            baseline_output_sha256="2" * 64,
            candidate_output_sha256="3" * 64,
            metrics=self.metrics,
        )

    def test_canonical_round_trip_preserves_observed_evidence(self) -> None:
        evidence = self.evidence()
        payload = evidence.canonical_json()
        replay = ProspectKvRealModelEvictionEvidenceV1.from_canonical_json(payload)

        self.assertEqual(replay, evidence)
        self.assertEqual(replay.baseline_logical_kv_bytes, 320)
        self.assertEqual(replay.candidate_logical_kv_bytes, 192)
        self.assertEqual(len(replay.metrics), 2)
        self.assertAlmostEqual(replay.metrics[1].delta, -0.02)

    def test_rejects_tampered_metric_delta(self) -> None:
        raw = json.loads(self.evidence().canonical_json())
        raw["metrics"][1]["delta"] = 0.5
        payload = json.dumps(raw, sort_keys=True, separators=(",", ":"))

        with self.assertRaisesRegex(
            ProspectKvRealModelEvidenceError,
            "delta",
        ):
            ProspectKvRealModelEvictionEvidenceV1.from_canonical_json(payload)

    def test_rejects_logical_byte_claim_that_disagrees_with_eviction(self) -> None:
        raw = json.loads(self.evidence().canonical_json())
        raw["candidate_logical_kv_bytes"] = 128
        payload = json.dumps(raw, sort_keys=True, separators=(",", ":"))

        with self.assertRaisesRegex(
            ProspectKvRealModelEvidenceError,
            "candidate logical KV bytes",
        ):
            ProspectKvRealModelEvictionEvidenceV1.from_canonical_json(payload)

    def test_rejects_duplicate_metric_names(self) -> None:
        duplicate = ObservedMetric.capture(
            name="logit_l2",
            kind="quality",
            unit="ratio",
            preference="higher_is_better",
            baseline_value=0.9,
            candidate_value=0.8,
        )
        with self.assertRaisesRegex(
            ProspectKvRealModelEvidenceError,
            "duplicate observed metric",
        ):
            ProspectKvRealModelEvictionEvidenceV1.capture(
                experiment_id="duplicate-metric",
                run_repository_revision="a" * 40,
                model_id="example/model",
                model_revision="model-rev-1",
                tokenizer_revision="tokenizer-rev-1",
                runtime_backend="fixture-backend",
                runtime_revision="runtime-rev-1",
                evaluation_id="holdout-001",
                trace_sha256="1" * 64,
                seed=7,
                eviction=self.eviction,
                baseline_output_sha256="2" * 64,
                candidate_output_sha256="3" * 64,
                metrics=(self.metrics[0], duplicate),
            )

    def test_rejects_noncanonical_or_unknown_fields(self) -> None:
        canonical = self.evidence().canonical_json()
        pretty = json.dumps(json.loads(canonical), indent=2, sort_keys=True)
        with self.assertRaisesRegex(
            ProspectKvRealModelEvidenceError,
            "not canonical",
        ):
            ProspectKvRealModelEvictionEvidenceV1.from_canonical_json(pretty)

        raw = json.loads(canonical)
        raw["unsupported"] = True
        payload = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        with self.assertRaisesRegex(
            ProspectKvRealModelEvidenceError,
            "fields do not match",
        ):
            ProspectKvRealModelEvictionEvidenceV1.from_canonical_json(payload)

    def test_rejects_nonfinite_metric_values(self) -> None:
        with self.assertRaisesRegex(
            ProspectKvRealModelEvidenceError,
            "finite",
        ):
            ObservedMetric.capture(
                name="bad",
                kind="numerical",
                unit="l2",
                preference="none",
                baseline_value=0.0,
                candidate_value=math.inf,
            )

    def test_rejects_invalid_output_digest(self) -> None:
        with self.assertRaisesRegex(
            ProspectKvRealModelEvidenceError,
            "baseline_output_sha256",
        ):
            ProspectKvRealModelEvictionEvidenceV1.capture(
                experiment_id="bad-digest",
                run_repository_revision="a" * 40,
                model_id="example/model",
                model_revision="model-rev-1",
                tokenizer_revision="tokenizer-rev-1",
                runtime_backend="fixture-backend",
                runtime_revision="runtime-rev-1",
                evaluation_id="holdout-001",
                trace_sha256="1" * 64,
                seed=7,
                eviction=self.eviction,
                baseline_output_sha256="not-a-digest",
                candidate_output_sha256="3" * 64,
                metrics=self.metrics,
            )


if __name__ == "__main__":
    unittest.main()
