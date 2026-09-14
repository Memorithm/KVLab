import unittest

from kvlab.prospect_position_baselines import (
    ProspectPositionBaselineError,
    budget_matched_lru_random,
    select_lru_positions,
    select_seeded_random_positions,
)


class ProspectPositionBaselineTests(unittest.TestCase):
    def test_lru_keeps_newest_positions_at_exact_budget(self):
        selection = select_lru_positions(
            input_position_count=10, retained_position_count=4
        )
        self.assertEqual(selection.policy, "lru")
        self.assertEqual(selection.retained_positions, (6, 7, 8, 9))
        self.assertEqual(selection.retained_position_count, 4)

    def test_seeded_random_is_reproducible_sorted_and_budget_matched(self):
        first = select_seeded_random_positions(
            input_position_count=10, retained_position_count=4, seed=7
        )
        second = select_seeded_random_positions(
            input_position_count=10, retained_position_count=4, seed=7
        )
        self.assertEqual(first, second)
        self.assertEqual(first.policy, "random_seeded")
        self.assertEqual(tuple(sorted(first.retained_positions)), first.retained_positions)
        self.assertEqual(len(set(first.retained_positions)), 4)
        self.assertTrue(all(0 <= position < 10 for position in first.retained_positions))

    def test_pair_has_identical_retained_row_budget(self):
        lru, random = budget_matched_lru_random(
            input_position_count=27, retained_position_count=14, seed=11
        )
        self.assertEqual(lru.retained_position_count, random.retained_position_count)
        self.assertEqual(len(lru.retained_positions), len(random.retained_positions))
        self.assertNotEqual(lru.retained_positions, random.retained_positions)

    def test_full_cache_zero_and_out_of_range_budgets_fail_closed(self):
        for retained in (0, 10, 11):
            with self.assertRaises(ProspectPositionBaselineError):
                select_lru_positions(
                    input_position_count=10, retained_position_count=retained
                )
        with self.assertRaises(ProspectPositionBaselineError):
            select_seeded_random_positions(
                input_position_count=10, retained_position_count=4, seed=-1
            )


if __name__ == "__main__":
    unittest.main()
