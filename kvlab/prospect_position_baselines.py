"""Deterministic position-native baselines for real-model KV campaigns.

These selectors operate only on sequence positions and an explicit retained-row
budget. They intentionally do not reuse the synthetic magnitude/sensitivity
baselines, which depend on synthetic contribution vectors and are not
real-model signatures.

The returned positions describe logical KV rows only. They are not evidence of
quality, allocator release, HBM savings, traffic reduction, latency, or
throughput.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random


class ProspectPositionBaselineError(ValueError):
    """Raised when a position-baseline request is structurally invalid."""


@dataclass(frozen=True, slots=True)
class PositionBaselineSelection:
    policy: str
    input_position_count: int
    retained_position_count: int
    retained_positions: tuple[int, ...]


def _validate_budget(input_position_count: int, retained_position_count: int) -> None:
    if input_position_count <= 0:
        raise ProspectPositionBaselineError("input_position_count must be positive")
    if retained_position_count <= 0:
        raise ProspectPositionBaselineError("retained_position_count must be positive")
    if retained_position_count >= input_position_count:
        raise ProspectPositionBaselineError(
            "candidate retained_position_count must be smaller than the full-cache baseline"
        )


def select_lru_positions(
    *, input_position_count: int, retained_position_count: int
) -> PositionBaselineSelection:
    """Keep the newest positions under an exact retained-row budget."""

    _validate_budget(input_position_count, retained_position_count)
    start = input_position_count - retained_position_count
    return PositionBaselineSelection(
        policy="lru",
        input_position_count=input_position_count,
        retained_position_count=retained_position_count,
        retained_positions=tuple(range(start, input_position_count)),
    )


def select_seeded_random_positions(
    *, input_position_count: int, retained_position_count: int, seed: int
) -> PositionBaselineSelection:
    """Select one reproducible random subset, returned in canonical position order."""

    _validate_budget(input_position_count, retained_position_count)
    if not 0 <= seed <= (2**64 - 1):
        raise ProspectPositionBaselineError("seed must be an unsigned 64-bit integer")
    retained = sorted(
        Random(seed).sample(range(input_position_count), retained_position_count)
    )
    return PositionBaselineSelection(
        policy="random_seeded",
        input_position_count=input_position_count,
        retained_position_count=retained_position_count,
        retained_positions=tuple(retained),
    )


def budget_matched_lru_random(
    *, input_position_count: int, retained_position_count: int, seed: int
) -> tuple[PositionBaselineSelection, PositionBaselineSelection]:
    """Return LRU and seeded-random controls at exactly one row budget."""

    return (
        select_lru_positions(
            input_position_count=input_position_count,
            retained_position_count=retained_position_count,
        ),
        select_seeded_random_positions(
            input_position_count=input_position_count,
            retained_position_count=retained_position_count,
            seed=seed,
        ),
    )
