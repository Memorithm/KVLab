from fractions import Fraction
import unittest

from kvlab.bikv_traffic import (
    LOGICAL_PACKED_PAYLOAD,
    PHYSICAL_DRAM_COUNTER,
    BikvTrafficEvidence,
    BikvTrafficEvidenceError,
)


class BikvTrafficEvidenceTests(unittest.TestCase):
    def test_logical_payload_ratio_is_exact_and_labelled(self) -> None:
        evidence = BikvTrafficEvidence(
            numerical_kv_bytes_avoided=4096,
            numerical_evidence_kind=LOGICAL_PACKED_PAYLOAD,
            boolean_kv_bytes_read=256,
            boolean_evidence_kind=LOGICAL_PACKED_PAYLOAD,
        )
        self.assertEqual(evidence.evidence_kind, LOGICAL_PACKED_PAYLOAD)
        self.assertEqual(
            evidence.numerical_bytes_avoided_per_boolean_byte,
            Fraction(16, 1),
        )

    def test_physical_counter_ratio_remains_separate(self) -> None:
        evidence = BikvTrafficEvidence(
            numerical_kv_bytes_avoided=600,
            numerical_evidence_kind=PHYSICAL_DRAM_COUNTER,
            boolean_kv_bytes_read=100,
            boolean_evidence_kind=PHYSICAL_DRAM_COUNTER,
        )
        self.assertEqual(evidence.evidence_kind, PHYSICAL_DRAM_COUNTER)
        self.assertEqual(evidence.numerical_bytes_avoided_per_boolean_byte, Fraction(6, 1))

    def test_mixed_logical_and_physical_evidence_fails_closed(self) -> None:
        with self.assertRaisesRegex(BikvTrafficEvidenceError, "same evidence kind"):
            BikvTrafficEvidence(
                numerical_kv_bytes_avoided=4096,
                numerical_evidence_kind=LOGICAL_PACKED_PAYLOAD,
                boolean_kv_bytes_read=256,
                boolean_evidence_kind=PHYSICAL_DRAM_COUNTER,
            ).validate()

    def test_unknown_kind_and_bool_alias_fail_closed(self) -> None:
        with self.assertRaises(BikvTrafficEvidenceError):
            BikvTrafficEvidence(
                numerical_kv_bytes_avoided=1,
                numerical_evidence_kind="estimated_bandwidth",
                boolean_kv_bytes_read=1,
                boolean_evidence_kind="estimated_bandwidth",
            ).validate()
        with self.assertRaises(BikvTrafficEvidenceError):
            BikvTrafficEvidence(
                numerical_kv_bytes_avoided=True,
                numerical_evidence_kind=LOGICAL_PACKED_PAYLOAD,
                boolean_kv_bytes_read=1,
                boolean_evidence_kind=LOGICAL_PACKED_PAYLOAD,
            ).validate()

    def test_zero_counters_do_not_fabricate_ratio(self) -> None:
        evidence = BikvTrafficEvidence(
            numerical_kv_bytes_avoided=0,
            numerical_evidence_kind=LOGICAL_PACKED_PAYLOAD,
            boolean_kv_bytes_read=0,
            boolean_evidence_kind=LOGICAL_PACKED_PAYLOAD,
        )
        self.assertIsNone(evidence.numerical_bytes_avoided_per_boolean_byte)


if __name__ == "__main__":
    unittest.main()
