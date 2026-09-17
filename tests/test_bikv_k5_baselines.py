import unittest
from unittest.mock import patch
from fractions import Fraction

from kvlab.bikv_k5_baselines import (
    BikvK5BaselineError,
    BikvK5MatchedDensityBaselines,
)


class BikvK5MatchedDensityBaselineTests(unittest.TestCase):
    def test_controls_match_boolean_density_exactly(self) -> None:
        plan = BikvK5MatchedDensityBaselines(
            total_pages=10,
            boolean_selected_pages=(1, 4, 9),
            seed=0xB1C5,
        )

        random_pages = plan.random_matched_pages()
        positional_pages = plan.positional_matched_pages()

        self.assertEqual(plan.selected_count, 3)
        self.assertEqual(plan.candidate_density, Fraction(3, 10))
        self.assertEqual(len(random_pages), 3)
        self.assertEqual(len(set(random_pages)), 3)
        self.assertEqual(tuple(sorted(random_pages)), random_pages)
        self.assertEqual(positional_pages, (7, 8, 9))
        self.assertEqual(plan.full_pages(), tuple(range(10)))

    def test_random_control_is_reproducible_and_seed_bound(self) -> None:
        left = BikvK5MatchedDensityBaselines(32, (0, 4, 8, 12, 16, 20), 7)
        same = BikvK5MatchedDensityBaselines(32, (1, 5, 9, 13, 17, 21), 7)
        other_seed = BikvK5MatchedDensityBaselines(32, (0, 4, 8, 12, 16, 20), 8)

        self.assertEqual(left.random_matched_pages(), same.random_matched_pages())
        self.assertNotEqual(left.random_matched_pages(), other_seed.random_matched_pages())
        self.assertEqual(left.random_algorithm, "splitmix64-page-ranking-v1")
        self.assertEqual(left.positional_algorithm, "tail-window-v1")

    def test_cross_repository_reference_vectors_are_stable(self) -> None:
        seed_7 = BikvK5MatchedDensityBaselines(32, (0, 4, 8, 12, 16, 20), 7)
        seed_8 = BikvK5MatchedDensityBaselines(32, (0, 4, 8, 12, 16, 20), 8)
        seed_b1c5 = BikvK5MatchedDensityBaselines(10, (1, 4, 9), 0xB1C5)

        self.assertEqual(seed_7.random_matched_pages(), (4, 12, 13, 18, 19, 21))
        self.assertEqual(seed_8.random_matched_pages(), (2, 3, 11, 26, 28, 29))
        self.assertEqual(seed_7.positional_matched_pages(), (26, 27, 28, 29, 30, 31))
        self.assertEqual(seed_b1c5.random_matched_pages(), (0, 2, 3))

    def test_full_density_shortcuts_ranking(self) -> None:
        plan = BikvK5MatchedDensityBaselines(4, (0, 1, 2, 3), 17)
        with patch("kvlab.bikv_k5_baselines._splitmix64", side_effect=AssertionError):
            self.assertEqual(plan.random_matched_pages(), (0, 1, 2, 3))

    def test_zero_density_and_empty_cache_are_explicit(self) -> None:
        empty_selection = BikvK5MatchedDensityBaselines(5, (), 0)
        empty_cache = BikvK5MatchedDensityBaselines(0, (), 0)

        self.assertEqual(empty_selection.candidate_density, Fraction(0, 1))
        self.assertEqual(empty_selection.random_matched_pages(), ())
        self.assertEqual(empty_selection.positional_matched_pages(), ())
        self.assertEqual(empty_cache.candidate_density, Fraction(0, 1))
        self.assertEqual(empty_cache.full_pages(), ())

    def test_invalid_plans_fail_closed(self) -> None:
        invalid_plans = (
            BikvK5MatchedDensityBaselines(-1, (), 0),
            BikvK5MatchedDensityBaselines(3, (0, 0), 0),
            BikvK5MatchedDensityBaselines(3, (2, 1), 0),
            BikvK5MatchedDensityBaselines(3, (3,), 0),
            BikvK5MatchedDensityBaselines(3, (True,), 0),
            BikvK5MatchedDensityBaselines(3, (), -1),
            BikvK5MatchedDensityBaselines(3, (), 1 << 64),
        )
        for plan in invalid_plans:
            with self.subTest(plan=plan):
                with self.assertRaises(BikvK5BaselineError):
                    plan.validate()


if __name__ == "__main__":
    unittest.main()
