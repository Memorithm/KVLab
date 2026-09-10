"""Deterministic non-model budget baselines for C1 calibration.

The policies in this module operate only on :class:`SyntheticKvTrace` metadata.
They do not implement FLAT-ATTENTION cache semantics, model attention scores, or
NNIS placement. Their purpose is to provide reproducible budget-matched controls
before any real-model C1 experiment is authorized.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .synthetic_trace import SyntheticKvTrace


@dataclass(frozen=True)
class BudgetSelection:
    """One deterministic policy result under an explicit byte budget."""

    policy: str
    budget_bytes: int
    retained_region_ids: tuple[str, ...]
    retained_bytes: int

    @property
    def unused_bytes(self) -> int:
        """Return budget bytes left unused by the greedy packing rule."""

        return self.budget_bytes - self.retained_bytes


def _validate_budget(trace: SyntheticKvTrace, budget_bytes: int) -> None:
    if budget_bytes < 0:
        raise ValueError("budget_bytes must be non-negative")
    if budget_bytes > trace.total_storage_bytes:
        raise ValueError("budget_bytes cannot exceed full-cache storage")


def select_lru_baseline(trace: SyntheticKvTrace, budget_bytes: int) -> BudgetSelection:
    """Keep newest synthetic regions first, treating tuple order as recency.

    The trace ordering convention is explicit: earlier tuple entries are older
    and later entries are newer. Regions are indivisible; a region is retained
    only when its complete storage cost fits in the remaining budget.
    """

    _validate_budget(trace, budget_bytes)
    selected: list[str] = []
    retained_bytes = 0
    for region in reversed(trace.regions):
        if retained_bytes + region.storage_bytes <= budget_bytes:
            selected.append(region.region_id)
            retained_bytes += region.storage_bytes
    selected.reverse()
    return BudgetSelection("lru", budget_bytes, tuple(selected), retained_bytes)


def select_random_baseline(
    trace: SyntheticKvTrace,
    budget_bytes: int,
    *,
    seed: int,
) -> BudgetSelection:
    """Select indivisible regions in a seeded random order under the byte budget."""

    _validate_budget(trace, budget_bytes)
    indices = list(range(len(trace.regions)))
    Random(seed).shuffle(indices)
    retained: set[int] = set()
    retained_bytes = 0
    for index in indices:
        region = trace.regions[index]
        if retained_bytes + region.storage_bytes <= budget_bytes:
            retained.add(index)
            retained_bytes += region.storage_bytes
    retained_region_ids = tuple(
        region.region_id for index, region in enumerate(trace.regions) if index in retained
    )
    return BudgetSelection("random", budget_bytes, retained_region_ids, retained_bytes)
