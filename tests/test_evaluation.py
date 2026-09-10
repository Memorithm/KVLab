import unittest

from kvlab.baselines import BudgetSelection, select_lru_baseline, select_random_baseline
from kvlab.evaluation import evaluate_selection
from kvlab.synthetic_trace import KvRegion, SyntheticKvTrace


class SelectionEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = SyntheticKvTrace(
            "fixture",
            (
                KvRegion("a", 4, (1.0, 2.0)),
                KvRegion("b", 6, (3.0, -1.0)),
                KvRegion("c", 4, (-2.0, 4.0)),
            ),
        )

    def test_lru_selection_is_scored_against_full_cache(self) -> None:
        result = evaluate_selection(self.trace, select_lru_baseline(self.trace, 10))
        self.assertEqual(result.retained_region_ids, ("b", "c"))
        self.assertEqual(result.full_cache_output, (2.0, 5.0))
        self.assertEqual(result.selected_output, (1.0, 3.0))
        self.assertAlmostEqual(result.output_l2_delta, 5**0.5)
        self.assertEqual(result.retained_bytes, 10)
        self.assertEqual(result.unused_bytes, 0)

    def test_random_selection_evaluation_is_reproducible(self) -> None:
        first = evaluate_selection(self.trace, select_random_baseline(self.trace, 8, seed=7))
        second = evaluate_selection(self.trace, select_random_baseline(self.trace, 8, seed=7))
        self.assertEqual(first, second)

    def test_unknown_region_fails_closed(self) -> None:
        selection = BudgetSelection("invalid", 4, ("missing",), 4)
        with self.assertRaisesRegex(ValueError, "unknown region_id"):
            evaluate_selection(self.trace, selection)

    def test_inconsistent_retained_bytes_fails_closed(self) -> None:
        selection = BudgetSelection("invalid", 4, ("a",), 3)
        with self.assertRaisesRegex(ValueError, "retained_bytes"):
            evaluate_selection(self.trace, selection)

    def test_duplicate_region_fails_closed(self) -> None:
        selection = BudgetSelection("invalid", 8, ("a", "a"), 8)
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate_selection(self.trace, selection)


if __name__ == "__main__":
    unittest.main()
