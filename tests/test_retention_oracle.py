import unittest

from kvlab.oracle import KVLayerRecord, capture_full_cache
from kvlab.retention import SlidingWindowPolicy, apply_sliding_window
from kvlab.retention_oracle import compare_retention_to_oracle


class RetentionOracleComparisonTests(unittest.TestCase):
    def _oracle(self):
        return capture_full_cache(
            model_revision="model@abc123",
            tokenizer_revision="tokenizer@def456",
            sequence_length=4,
            records=(
                KVLayerRecord(
                    layer=0,
                    dtype="fp16",
                    key_shape=(4, 1),
                    value_shape=(4, 1),
                    key_bytes=b"abcdefgh",
                    value_bytes=b"ijklmnop",
                ),
            ),
        )

    def test_compares_sliding_window_against_exact_oracle(self):
        oracle = self._oracle()
        candidate = apply_sliding_window(
            input_tokens=4,
            bytes_per_token=4,
            policy=SlidingWindowPolicy(window_tokens=2),
        )

        comparison = compare_retention_to_oracle(
            oracle=oracle,
            candidate_input_tokens=candidate.input_tokens,
            candidate_logical_input_bytes=candidate.logical_input_bytes,
            retained_tokens=candidate.retained_tokens,
            retained_logical_bytes=candidate.logical_retained_bytes,
        )

        self.assertEqual(comparison.oracle_sequence_length, 4)
        self.assertEqual(comparison.oracle_logical_bytes, 16)
        self.assertEqual(comparison.logical_reduction_tokens, 2)
        self.assertEqual(comparison.logical_reduction_bytes, 8)
        self.assertEqual(comparison.logical_retention_ratio, 0.5)
        self.assertFalse(hasattr(comparison, "gpu_resident_bytes"))

    def test_rejects_different_context_length(self):
        oracle = self._oracle()
        with self.assertRaisesRegex(ValueError, "token count"):
            compare_retention_to_oracle(
                oracle=oracle,
                candidate_input_tokens=3,
                candidate_logical_input_bytes=16,
                retained_tokens=2,
                retained_logical_bytes=8,
            )

    def test_rejects_different_logical_representation(self):
        oracle = self._oracle()
        with self.assertRaisesRegex(ValueError, "logical input bytes"):
            compare_retention_to_oracle(
                oracle=oracle,
                candidate_input_tokens=4,
                candidate_logical_input_bytes=8,
                retained_tokens=2,
                retained_logical_bytes=4,
            )

    def test_rejects_impossible_retained_accounting(self):
        oracle = self._oracle()
        with self.assertRaisesRegex(ValueError, "zero retained tokens"):
            compare_retention_to_oracle(
                oracle=oracle,
                candidate_input_tokens=4,
                candidate_logical_input_bytes=16,
                retained_tokens=0,
                retained_logical_bytes=4,
            )


if __name__ == "__main__":
    unittest.main()
