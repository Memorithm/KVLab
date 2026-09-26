import json
import unittest

from kvlab.bikv_mass_evidence import BikvMassEvidenceError, BikvMassEvidenceV1
from kvlab.boolean_kv import BooleanKvCache, PackedBits
from kvlab.prospect_handoff import ProspectBkvHandoffV1


def packed(value: int, bits: int = 4) -> PackedBits:
    return PackedBits(bit_length=bits, words=(value,))


class BikvMassEvidenceTests(unittest.TestCase):
    def handoff(self) -> ProspectBkvHandoffV1:
        cache = BooleanKvCache(4)
        for value in (0b1010, 0b1110, 0b0000, 0b1011):
            cache.append(packed(value))
        return ProspectBkvHandoffV1.capture(
            cache,
            packed(0b1010),
            max_distance=1,
        )

    def evidence(self) -> BikvMassEvidenceV1:
        return BikvMassEvidenceV1.capture(
            experiment_id="skv3-test",
            query_id="q0",
            score_semantics="max-token-logit-per-page-v1",
            handoff=self.handoff(),
            reference_page_scores=(8.0, 7.0, 0.0, 6.0),
            top_k=2,
        )

    def test_capture_binds_bikv_survivors_to_mass_metrics(self) -> None:
        evidence = self.evidence()
        self.assertEqual(evidence.selection.selected_count, 3)
        self.assertEqual(evidence.selection.top_k_recall, 1.0)
        self.assertGreater(evidence.selection.retained_softmax_mass, 0.99)
        self.assertTrue(evidence.evaluation_oracle_only)

    def test_round_trip_recomputes_handoff_and_metrics(self) -> None:
        evidence = self.evidence()
        payload = evidence.canonical_json()
        self.assertEqual(BikvMassEvidenceV1.from_canonical_json(payload), evidence)

    def test_tampered_mass_metric_is_rejected(self) -> None:
        decoded = json.loads(self.evidence().canonical_json())
        decoded["selection"]["retained_softmax_mass"] = 0.0
        payload = json.dumps(decoded, sort_keys=True, separators=(",", ":"))
        with self.assertRaisesRegex(BikvMassEvidenceError, "inconsistent derived metrics"):
            BikvMassEvidenceV1.from_canonical_json(payload)

    def test_dense_score_oracle_flag_cannot_be_disabled(self) -> None:
        decoded = json.loads(self.evidence().canonical_json())
        decoded["evaluation_oracle_only"] = False
        payload = json.dumps(decoded, sort_keys=True, separators=(",", ":"))
        with self.assertRaisesRegex(BikvMassEvidenceError, "evaluation-oracle-only"):
            BikvMassEvidenceV1.from_canonical_json(payload)

    def test_score_count_must_match_page_universe(self) -> None:
        with self.assertRaisesRegex(BikvMassEvidenceError, "length"):
            BikvMassEvidenceV1.capture(
                experiment_id="skv3-test",
                query_id="q0",
                score_semantics="test",
                handoff=self.handoff(),
                reference_page_scores=(1.0, 0.0),
                top_k=1,
            )


if __name__ == "__main__":
    unittest.main()
