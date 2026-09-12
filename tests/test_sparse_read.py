import unittest

from kvlab.sparse_read import (
    PageScore,
    QueryAwarePagePolicy,
    select_query_aware_pages,
)


class QueryAwareSparseReadTests(unittest.TestCase):
    def test_highest_scores_are_selected_with_stable_tie_break(self) -> None:
        accounting = select_query_aware_pages(
            scores=(
                PageScore(page_id=4, score=0.5),
                PageScore(page_id=1, score=0.9),
                PageScore(page_id=3, score=0.9),
                PageScore(page_id=2, score=0.1),
            ),
            page_bytes=4096,
            policy=QueryAwarePagePolicy(max_pages=2),
        )

        self.assertEqual(accounting.selected_pages, (1, 3))
        self.assertEqual(accounting.skipped_pages, (2, 4))
        self.assertEqual(accounting.logical_cache_bytes, 4 * 4096)
        self.assertEqual(accounting.selected_read_bytes, 2 * 4096)
        self.assertAlmostEqual(accounting.selected_fraction, 0.5)

    def test_sparse_read_does_not_claim_logical_cache_reduction(self) -> None:
        accounting = select_query_aware_pages(
            scores=(
                PageScore(page_id=0, score=2.0),
                PageScore(page_id=1, score=1.0),
                PageScore(page_id=2, score=0.0),
            ),
            page_bytes=1024,
            policy=QueryAwarePagePolicy(max_pages=1),
        )

        self.assertEqual(accounting.logical_cache_bytes, 3072)
        self.assertEqual(accounting.selected_read_bytes, 1024)
        self.assertEqual(accounting.total_pages, 3)

    def test_zero_page_budget_is_explicit_and_deterministic(self) -> None:
        accounting = select_query_aware_pages(
            scores=(PageScore(page_id=7, score=1.0),),
            page_bytes=512,
            policy=QueryAwarePagePolicy(max_pages=0),
        )

        self.assertEqual(accounting.selected_pages, ())
        self.assertEqual(accounting.skipped_pages, (7,))
        self.assertEqual(accounting.logical_cache_bytes, 512)
        self.assertEqual(accounting.selected_read_bytes, 0)

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            QueryAwarePagePolicy(max_pages=-1)
        with self.assertRaises(ValueError):
            PageScore(page_id=-1, score=0.0)
        with self.assertRaises(ValueError):
            PageScore(page_id=0, score=float("nan"))
        with self.assertRaises(ValueError):
            select_query_aware_pages(
                scores=(PageScore(page_id=0, score=1.0),),
                page_bytes=0,
                policy=QueryAwarePagePolicy(max_pages=1),
            )
        with self.assertRaises(ValueError):
            select_query_aware_pages(
                scores=(
                    PageScore(page_id=0, score=1.0),
                    PageScore(page_id=0, score=0.5),
                ),
                page_bytes=1024,
                policy=QueryAwarePagePolicy(max_pages=1),
            )


if __name__ == "__main__":
    unittest.main()
