import unittest

from kvlab.baselines import (
    BudgetSelection,
    select_lru_baseline,
    select_magnitude_baseline,
    select_random_baseline,
    select_sensitivity_per_byte_baseline,
)
from kvlab.comparison import compare_budget_selections
from kvlab.synthetic_trace import KvRegion, SyntheticKvTrace


class BudgetComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = SyntheticKvTrace(
            trace_id="comparison-fixture",
            regions=(
                KvRegion("old", 4, (1.0, 0.0)),
                KvRegion("middle", 6, (0.0, 2.0)),
                KvRegion("new", 4, (2.0, 1.0)),
            ),
        )

    def test_compares_all_current_baselines_under_one_budget(self) -> None:
        budget = 8
        comparison = compare_budget_selections(
            self.trace,
            (
                select_lru_baseline(self.trace, budget),
                select_random_baseline(self.trace, budget, seed=7),
                select_magnitude_baseline(self.trace, budget),
                select_sensitivity_per_byte_baseline(self.trace, budget),
            ),
        )

        self.assertEqual(comparison.trace_id, self.trace.trace_id)
        self.assertEqual(comparison.budget_bytes, budget)
        by_policy = comparison.by_policy()
        self.assertEqual(
            set(by_policy),
            {"lru", "random", "magnitude", "synthetic_sensitivity_per_byte"},
        )
        for evaluation in comparison.evaluations:
            self.assertEqual(evaluation.budget_bytes, budget)
            self.assertLessEqual(evaluation.retained_bytes, budget)
            self.assertGreaterEqual(evaluation.output_l2_delta, 0.0)

    def test_reports_budget_matched_regret_relative_to_best_policy(self) -> None:
        budget = 8
        comparison = compare_budget_selections(
            self.trace,
            (
                select_lru_baseline(self.trace, budget),
                select_magnitude_baseline(self.trace, budget),
                select_sensitivity_per_byte_baseline(self.trace, budget),
            ),
        )

        regrets = comparison.regret_by_policy()
        self.assertEqual(set(regrets), set(comparison.by_policy()))
        self.assertAlmostEqual(min(regrets.values()), 0.0)
        self.assertAlmostEqual(
            comparison.best_output_l2_delta(),
            min(evaluation.output_l2_delta for evaluation in comparison.evaluations),
        )
        for policy, regret in regrets.items():
            self.assertAlmostEqual(
                regret,
                comparison.by_policy()[policy].output_l2_delta
                - comparison.best_output_l2_delta(),
            )
            self.assertGreaterEqual(regret, 0.0)

    def test_rejects_mixed_budgets(self) -> None:
        with self.assertRaises(ValueError):
            compare_budget_selections(
                self.trace,
                (
                    select_lru_baseline(self.trace, 8),
                    select_magnitude_baseline(self.trace, 10),
                ),
            )

    def test_rejects_duplicate_policy_names(self) -> None:
        first = BudgetSelection("same", 8, ("old",), 4)
        second = BudgetSelection("same", 8, ("new",), 4)
        with self.assertRaises(ValueError):
            compare_budget_selections(self.trace, (first, second))

    def test_rejects_empty_comparison(self) -> None:
        with self.assertRaises(ValueError):
            compare_budget_selections(self.trace, ())


if __name__ == "__main__":
    unittest.main()
