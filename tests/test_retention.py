import unittest

from kvlab.retention import SlidingWindowPolicy, apply_sliding_window


class SlidingWindowRetentionTests(unittest.TestCase):
    def test_window_evicts_only_oldest_logical_entries(self) -> None:
        accounting = apply_sliding_window(
            input_tokens=10,
            bytes_per_token=256,
            policy=SlidingWindowPolicy(window_tokens=4),
        )

        self.assertEqual(accounting.retained_tokens, 4)
        self.assertEqual(accounting.evicted_tokens, 6)
        self.assertEqual(accounting.logical_input_bytes, 2560)
        self.assertEqual(accounting.logical_retained_bytes, 1024)
        self.assertEqual(accounting.logical_evicted_bytes, 1536)
        self.assertAlmostEqual(accounting.logical_retention_ratio, 0.4)

    def test_short_sequence_is_unchanged(self) -> None:
        accounting = apply_sliding_window(
            input_tokens=3,
            bytes_per_token=128,
            policy=SlidingWindowPolicy(window_tokens=8),
        )

        self.assertEqual(accounting.retained_tokens, 3)
        self.assertEqual(accounting.evicted_tokens, 0)
        self.assertEqual(accounting.logical_retained_bytes, 384)
        self.assertEqual(accounting.logical_retention_ratio, 1.0)

    def test_empty_sequence_has_full_retention_ratio(self) -> None:
        accounting = apply_sliding_window(
            input_tokens=0,
            bytes_per_token=64,
            policy=SlidingWindowPolicy(window_tokens=4),
        )
        self.assertEqual(accounting.logical_retention_ratio, 1.0)

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            SlidingWindowPolicy(window_tokens=0)
        with self.assertRaises(ValueError):
            apply_sliding_window(
                input_tokens=-1,
                bytes_per_token=64,
                policy=SlidingWindowPolicy(window_tokens=4),
            )
        with self.assertRaises(ValueError):
            apply_sliding_window(
                input_tokens=1,
                bytes_per_token=0,
                policy=SlidingWindowPolicy(window_tokens=4),
            )


if __name__ == "__main__":
    unittest.main()
