import unittest

from kvlab.eviction import EvictionPolicy, apply_eviction


class EvictionTests(unittest.TestCase):
    def test_oldest_first_eviction_is_deterministic(self) -> None:
        result = apply_eviction(
            token_ids=[10, 11, 12, 13, 14],
            bytes_per_token=64,
            policy=EvictionPolicy(max_tokens=3),
        )

        self.assertEqual(result.evicted_token_ids, (10, 11))
        self.assertEqual(result.retained_token_ids, (12, 13, 14))
        self.assertEqual(result.logical_input_bytes, 320)
        self.assertEqual(result.logical_retained_bytes, 192)
        self.assertEqual(result.logical_evicted_bytes, 128)

    def test_short_sequence_is_unchanged(self) -> None:
        result = apply_eviction(
            token_ids=[1, 2],
            bytes_per_token=32,
            policy=EvictionPolicy(max_tokens=4),
        )

        self.assertEqual(result.evicted_token_ids, ())
        self.assertEqual(result.retained_token_ids, (1, 2))
        self.assertEqual(result.logical_evicted_bytes, 0)

    def test_duplicate_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            apply_eviction(
                token_ids=[1, 1],
                bytes_per_token=16,
                policy=EvictionPolicy(max_tokens=1),
            )

    def test_invalid_sizes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EvictionPolicy(max_tokens=0)
        with self.assertRaises(ValueError):
            apply_eviction(
                token_ids=[1],
                bytes_per_token=0,
                policy=EvictionPolicy(max_tokens=1),
            )


if __name__ == "__main__":
    unittest.main()
