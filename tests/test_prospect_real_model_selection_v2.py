import json
import unittest

from kvlab.prospect_real_model_eviction import ObservedMetric
from kvlab.prospect_real_model_selection_v2 import (
    ProspectKvRealModelSelectionEvidenceV2,
    ProspectKvRealModelSelectionV2Error,
)
from kvlab.prospect_selection_handoff_v2 import ProspectKvSelectionHandoffV2


def _selection():
    return ProspectKvSelectionHandoffV2.capture(
        token_ids=[7, 11, 7, 7, 19],
        bytes_per_token=64,
        policy="fixture",
        retained_positions=[0, 2, 4],
    )


def _metric():
    return ObservedMetric(
        name="token_accuracy",
        kind="quality",
        unit="ratio",
        preference="higher_is_better",
        baseline_value=1.0,
        candidate_value=0.75,
        delta=-0.25,
    )


def _evidence():
    return ProspectKvRealModelSelectionEvidenceV2.capture(
        experiment_id="position-campaign",
        run_repository_revision="a" * 40,
        model_id="example/model",
        model_revision="model-r1",
        tokenizer_revision="tok-r1",
        runtime_backend="nnis",
        runtime_revision="b" * 40,
        evaluation_id="eval-001",
        trace_sha256="1" * 64,
        seed=7,
        selection=_selection(),
        baseline_output_sha256="2" * 64,
        candidate_output_sha256="3" * 64,
        metrics=[_metric()],
    )


class ProspectKvRealModelSelectionV2Tests(unittest.TestCase):
    def test_position_evidence_round_trips_canonically_with_duplicate_tokens(self):
        evidence = _evidence()
        payload = evidence.canonical_json()
        replayed = ProspectKvRealModelSelectionEvidenceV2.from_canonical_json(payload)

        self.assertEqual(replayed, evidence)
        self.assertEqual(replayed.selection.retained_positions, (0, 2, 4))
        self.assertEqual(replayed.selection.retained_token_ids, (7, 7, 19))
        self.assertEqual(replayed.baseline_logical_kv_bytes, 320)
        self.assertEqual(replayed.candidate_logical_kv_bytes, 192)

    def test_position_evidence_rejects_logical_byte_tampering(self):
        raw = json.loads(_evidence().canonical_json())
        raw["candidate_logical_kv_bytes"] += 64
        payload = json.dumps(raw, sort_keys=True, separators=(",", ":"))

        with self.assertRaisesRegex(
            ProspectKvRealModelSelectionV2Error, "candidate logical KV bytes"
        ):
            ProspectKvRealModelSelectionEvidenceV2.from_canonical_json(payload)


if __name__ == "__main__":
    unittest.main()
