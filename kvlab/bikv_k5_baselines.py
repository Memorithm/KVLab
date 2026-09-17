"""Deterministic BKV-K5 matched-density selection baselines.

BKV-K5 compares Boolean-indexed numerical K/V against baselines with the same
selection density.  This module defines the page-id selection plans only.  It
intentionally does not execute attention, infer numerical K/V traffic, or turn
candidate density into a performance or quality claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from heapq import nsmallest


_MASK64 = (1 << 64) - 1
_BASELINE_ALGORITHM = "splitmix64-page-ranking-v1"
_POSITIONAL_ALGORITHM = "tail-window-v1"


class BikvK5BaselineError(ValueError):
    """Raised when a BKV-K5 matched-density baseline plan is invalid."""


def _require_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BikvK5BaselineError(f"{name} must be a non-negative integer")
    return value


def _splitmix64(value: int) -> int:
    """Return one fully specified SplitMix64 output for ``value``."""

    z = (value + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def _validate_selected_pages(
    *, total_pages: int, selected_pages: tuple[int, ...]
) -> tuple[int, ...]:
    total = _require_nonnegative_int("total_pages", total_pages)
    if not isinstance(selected_pages, tuple):
        raise BikvK5BaselineError("selected_pages must be a tuple")
    previous = -1
    for page in selected_pages:
        page_id = _require_nonnegative_int("selected page id", page)
        if page_id >= total:
            raise BikvK5BaselineError("selected page id is outside total_pages")
        if page_id <= previous:
            raise BikvK5BaselineError(
                "selected_pages must be strictly increasing and duplicate-free"
            )
        previous = page_id
    return selected_pages


@dataclass(frozen=True, slots=True)
class BikvK5MatchedDensityBaselines:
    """Selection-only baselines matched to one Boolean candidate set.

    ``boolean_selected_pages`` is the exact page-id set produced by the Boolean
    router for one frozen query/policy.  The random-like control ranks every
    logical page by a versioned SplitMix64 score and takes exactly the same
    number of pages.  The positional control takes the most recent logical
    pages.  Both controls therefore match Boolean candidate density exactly.

    The returned page ids are logical selection plans.  They are not evidence
    of bytes transferred, memory residency, latency, quality, or speedup.
    """

    total_pages: int
    boolean_selected_pages: tuple[int, ...]
    seed: int

    def validate(self) -> None:
        total = _require_nonnegative_int("total_pages", self.total_pages)
        seed = _require_nonnegative_int("seed", self.seed)
        if seed > _MASK64:
            raise BikvK5BaselineError("seed must fit in an unsigned 64-bit integer")
        _validate_selected_pages(
            total_pages=total, selected_pages=self.boolean_selected_pages
        )

    @property
    def selected_count(self) -> int:
        self.validate()
        return len(self.boolean_selected_pages)

    @property
    def candidate_density(self) -> Fraction:
        """Return exact selected-page density; zero pages yields 0/1."""

        self.validate()
        if self.total_pages == 0:
            return Fraction(0, 1)
        return Fraction(self.selected_count, self.total_pages)

    @property
    def random_algorithm(self) -> str:
        return _BASELINE_ALGORITHM

    @property
    def positional_algorithm(self) -> str:
        return _POSITIONAL_ALGORITHM

    def random_matched_pages(self) -> tuple[int, ...]:
        """Return a deterministic seed-bound matched-density control."""

        self.validate()
        count = self.selected_count
        if count == 0:
            return ()
        if count == self.total_pages:
            return self.full_pages()
        ranked = nsmallest(
            count,
            range(self.total_pages),
            key=lambda page: (_splitmix64(self.seed ^ page), page),
        )
        return tuple(sorted(ranked))

    def positional_matched_pages(self) -> tuple[int, ...]:
        """Return the latest-page tail window at exactly the matched density."""

        self.validate()
        count = self.selected_count
        if count == 0:
            return ()
        return tuple(range(self.total_pages - count, self.total_pages))

    def full_pages(self) -> tuple[int, ...]:
        """Return all logical pages for full/paged numerical controls."""

        self.validate()
        return tuple(range(self.total_pages))
