import unittest

from kvlab.temporal import (
    TemporalKvRegion,
    TemporalKvTrace,
    TemporalSelection,
    select_exact_future_oracle,
    select_observed_history_per_byte,
)
from kvlab.temporal_comparison import compare_temporal_selections


class TemporalComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = TemporalKvTrace(
            "comparison",
            (
                TemporalKvRegion("visible-now", 4, ((8.0,), (0.0,), (0.0,))),
                TemporalKvRegion("useful-later", 4, ((1.0,), (6.0,), (6.0,))),
            ),
        )

    def test_shared_budget_table_preserves_online_regret(self) -> None:
        online = select_observed_history_per_byte(self.trace, 0, 4)
        oracle = select_exact_future_oracle(self.trace, 0, 4)

        results = compare_temporal_selections(self.trace, (online, oracle))

        self.assertEqual([result.policy_name for result in results], ["observed_history_per_byte", "exact_future_oracle"])
        self.assertEqual(results[0].policy_future_utility, 0.0)
        self.assertEqual(results[0].oracle_future_utility, 12.0)
        self.assertEqual(results[0].regret, 12.0)
        self.assertEqual(results[1].regret, 0.0)

    def test_mismatched_budgets_fail_closed(self) -> None:
        online = select_observed_history_per_byte(self.trace, 0, 4)
        incompatible = TemporalSelection("other", 0, 8, ("visible-now",), 4)
        with self.assertRaises(ValueError):
            compare_temporal_selections(self.trace, (online, incompatible))

    def test_mismatched_decision_steps_fail_closed(self) -> None:
        first = select_observed_history_per_byte(self.trace, 0, 4)
        second = select_observed_history_per_byte(self.trace, 1, 4)
        with self.assertRaises(ValueError):
            compare_temporal_selections(self.trace, (first, second))

    def test_duplicate_policy_names_fail_closed(self) -> None:
        first = select_observed_history_per_byte(self.trace, 0, 4)
        duplicate = TemporalSelection(first.policy_name, 0, 4, ("useful-later",), 4)
        with self.assertRaises(ValueError):
            compare_temporal_selections(self.trace, (first, duplicate))

    def test_empty_comparison_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            compare_temporal_selections(self.trace, ())


if __name__ == "__main__":
    unittest.main()
