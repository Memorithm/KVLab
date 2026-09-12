import unittest

from kvlab.prefix_cache import PrefixLookup, summarize_prefix_cache


class PrefixCacheAccountingTests(unittest.TestCase):
    def test_summary_keeps_logical_kv_reduction_zero(self) -> None:
        summary = summarize_prefix_cache(
            [
                PrefixLookup("a", prefix_tokens=128, shared_full_pages=4, hit=True),
                PrefixLookup("b", prefix_tokens=64, shared_full_pages=0, hit=False),
                PrefixLookup("c", prefix_tokens=128, shared_full_pages=4, hit=True),
            ]
        )
        self.assertEqual(summary.requests, 3)
        self.assertEqual(summary.hits, 2)
        self.assertEqual(summary.misses, 1)
        self.assertAlmostEqual(summary.hit_rate, 2 / 3)
        self.assertEqual(summary.prefix_tokens_requested, 320)
        self.assertEqual(summary.prefix_tokens_hit, 256)
        self.assertEqual(summary.shared_full_pages, 8)
        self.assertEqual(summary.logical_kv_reduction_tokens, 0)

    def test_miss_cannot_claim_shared_pages(self) -> None:
        with self.assertRaisesRegex(ValueError, "miss cannot report shared pages"):
            PrefixLookup("miss", prefix_tokens=32, shared_full_pages=1, hit=False)

    def test_empty_observation_set_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one prefix lookup"):
            summarize_prefix_cache([])

    def test_negative_counts_and_empty_identity_are_rejected(self) -> None:
        invalid = [
            ("", 0, 0, False),
            ("x", -1, 0, False),
            ("x", 1, -1, True),
        ]
        for request_id, tokens, pages, hit in invalid:
            with self.subTest(request_id=request_id, tokens=tokens, pages=pages):
                with self.assertRaises(ValueError):
                    PrefixLookup(request_id, tokens, pages, hit)


if __name__ == "__main__":
    unittest.main()
