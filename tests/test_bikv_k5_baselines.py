from fractions import Fraction

import pytest

from kvlab.bikv_k5_baselines import (
    BikvK5BaselineError,
    BikvK5MatchedDensityBaselines,
)


def test_controls_match_boolean_density_exactly() -> None:
    plan = BikvK5MatchedDensityBaselines(
        total_pages=10,
        boolean_selected_pages=(1, 4, 9),
        seed=0xB1C5,
    )

    random_pages = plan.random_matched_pages()
    positional_pages = plan.positional_matched_pages()

    assert plan.selected_count == 3
    assert plan.candidate_density == Fraction(3, 10)
    assert len(random_pages) == 3
    assert len(set(random_pages)) == 3
    assert tuple(sorted(random_pages)) == random_pages
    assert positional_pages == (7, 8, 9)
    assert plan.full_pages() == tuple(range(10))


def test_random_control_is_reproducible_and_seed_bound() -> None:
    left = BikvK5MatchedDensityBaselines(32, (0, 4, 8, 12, 16, 20), 7)
    same = BikvK5MatchedDensityBaselines(32, (1, 5, 9, 13, 17, 21), 7)
    other_seed = BikvK5MatchedDensityBaselines(32, (0, 4, 8, 12, 16, 20), 8)

    assert left.random_matched_pages() == same.random_matched_pages()
    assert left.random_matched_pages() != other_seed.random_matched_pages()
    assert left.random_algorithm == "splitmix64-page-ranking-v1"
    assert left.positional_algorithm == "tail-window-v1"


def test_zero_density_and_empty_cache_are_explicit() -> None:
    empty_selection = BikvK5MatchedDensityBaselines(5, (), 0)
    empty_cache = BikvK5MatchedDensityBaselines(0, (), 0)

    assert empty_selection.candidate_density == Fraction(0, 1)
    assert empty_selection.random_matched_pages() == ()
    assert empty_selection.positional_matched_pages() == ()
    assert empty_cache.candidate_density == Fraction(0, 1)
    assert empty_cache.full_pages() == ()


@pytest.mark.parametrize(
    "plan",
    [
        BikvK5MatchedDensityBaselines(-1, (), 0),
        BikvK5MatchedDensityBaselines(3, (0, 0), 0),
        BikvK5MatchedDensityBaselines(3, (2, 1), 0),
        BikvK5MatchedDensityBaselines(3, (3,), 0),
        BikvK5MatchedDensityBaselines(3, (True,), 0),
        BikvK5MatchedDensityBaselines(3, (), -1),
        BikvK5MatchedDensityBaselines(3, (), 1 << 64),
    ],
)
def test_invalid_plans_fail_closed(plan: BikvK5MatchedDensityBaselines) -> None:
    with pytest.raises(BikvK5BaselineError):
        plan.validate()
