import json
import unittest

from kvlab.slha_evidence import (
    SLHA_QUALITY_EVIDENCE_SCHEMA_V1,
    SlhaEvidenceError,
    SlhaQualityEvidenceV1,
)


class SlhaEvidenceTests(unittest.TestCase):
    def evidence(self) -> SlhaQualityEvidenceV1:
        return SlhaQualityEvidenceV1.capture(
            experiment_id="skv0-test",
            query_id="q0",
            candidate_id="candidate-a",
            top_k=2,
            reference_scores=(4.0, 3.0, 2.0, 1.0),
            selected_ids=(0, 2),
            candidate_scores=(4.0, 1.0, 3.0, 2.0),
        )

    def test_capture_and_round_trip_recompute_metrics(self) -> None:
        evidence = self.evidence()
        self.assertEqual(evidence.schema, SLHA_QUALITY_EVIDENCE_SCHEMA_V1)
        payload = evidence.canonical_json()
        self.assertEqual(SlhaQualityEvidenceV1.from_canonical_json(payload), evidence)

    def test_tampered_derived_metric_is_rejected(self) -> None:
        decoded = json.loads(self.evidence().canonical_json())
        decoded["selection"]["top_k_recall"] = 1.0
        tampered = json.dumps(decoded, sort_keys=True, separators=(",", ":"))
        with self.assertRaisesRegex(SlhaEvidenceError, "inconsistent derived metrics"):
            SlhaQualityEvidenceV1.from_canonical_json(tampered)

    def test_noncanonical_json_is_rejected(self) -> None:
        pretty = json.dumps(json.loads(self.evidence().canonical_json()), indent=2, sort_keys=True)
        with self.assertRaisesRegex(SlhaEvidenceError, "not canonical"):
            SlhaQualityEvidenceV1.from_canonical_json(pretty)

    def test_missing_field_is_rejected(self) -> None:
        decoded = json.loads(self.evidence().canonical_json())
        decoded.pop("ranking")
        payload = json.dumps(decoded, sort_keys=True, separators=(",", ":"))
        with self.assertRaisesRegex(SlhaEvidenceError, "fields"):
            SlhaQualityEvidenceV1.from_canonical_json(payload)


if __name__ == "__main__":
    unittest.main()
