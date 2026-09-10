import unittest

from kvlab.baselines import (
    select_lru_baseline,
    select_magnitude_baseline,
    select_random_baseline,
    select_sensitivity_per_byte_baseline,
)
from kvlab.synthetic_trace import KvRegion, SyntheticKvTrace


class BudgetBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = SyntheticKvTrace(
            trace_id="budget-fixture",
            regions=(
                KvRegion("old", 4, (1.0, 0.0)),
                KvRegion("middle", 6, (0.0, 1.0)),
                KvRegion("new", 4, (1.0, 1.0)),
            ),
        )

    def test_lru_keeps_newest_regions_that_fit(self) -> None:
        result = select_lru_baseline(self.trace, 10)
        self.assertEqual(result.retained_region_ids, ("middle", "new"))
        self.assertEqual(result.retained_bytes, 10)
        self.assertEqual(result.unused_bytes, 0)

    def test_random_is_reproducible_for_a_fixed_seed(self) -> None:
        first = select_random_baseline(self.trace, 8, seed=7)
        second = select_random_baseline(self.trace, 8, seed=7)
        self.assertEqual(first, second)
        self.assertLessEqual(first.retained_bytes, 8)

    def test_magnitude_prefers_larger_contribution_norm(self) -> None:
        result = select_magnitude_baseline(self.trace, 8)
        self.assertEqual(result.retained_region_ids, ("old", "new"))
        self.assertEqual(result.retained_bytes, 8)
        self.assertEqual(result.policy, "magnitude")

    def test_sensitivity_per_byte_accounts_for_storage_cost(self) -> None:
        trace = SyntheticKvTrace(
            trace_id="sensitivity-fixture",
            regions=(
                KvRegion("large", 8, (3.0, 0.0)),
                KvRegion("efficient", 4, (2.0, 0.0)),
                KvRegion("small", 4, (1.0, 0.0)),
            ),
        )
        result = select_sensitivity_per_byte_baseline(trace, 8)
        self.assertEqual(result.retained_region_ids, ("efficient", "small"))
        self.assertEqual(result.retained_bytes, 8)
        self.assertEqual(result.policy, "synthetic_sensitivity_per_byte")

    def test_ranked_ties_preserve_trace_order(self) -> None:
        trace = SyntheticKvTrace(
            trace_id="tie-fixture",
            regions=(
                KvRegion("first", 4, (1.0, 0.0)),
                KvRegion("second", 4, (0.0, 1.0)),
            ),
        )
        result = select_magnitude_baseline(trace, 4)
        self.assertEqual(result.retained_region_ids, ("first",))

    def test_budget_above_full_cache_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_lru_baseline(self.trace, self.trace.total_storage_bytes + 1)

    def test_negative_budget_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_random_baseline(self.trace, -1, seed=0)


if __name__ == "__main__":
    unittest.main()
