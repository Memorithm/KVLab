"""Deterministic temporal C1 calibration for future KV utility.

This module extends the synthetic additive world with an explicit time axis so
immediate sensitivity can differ from future utility. It remains a non-model
calibration substrate: no attention semantics, CUDA placement, or scientific
claim about real KV caches is encoded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class TemporalKvRegion:
    """One synthetic region with fixed storage cost and per-step contributions."""

    region_id: str
    storage_bytes: int
    contributions: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.region_id:
            raise ValueError("region_id must be non-empty")
        if self.storage_bytes <= 0:
            raise ValueError("storage_bytes must be positive")
        if not self.contributions:
            raise ValueError("contributions must be non-empty")
        width = len(self.contributions[0])
        if width == 0:
            raise ValueError("contribution width must be positive")
        if any(len(step) != width for step in self.contributions):
            raise ValueError("all temporal contributions must have equal width")


@dataclass(frozen=True)
class TemporalKvTrace:
    """Synthetic time series with aligned region lifetimes and output width."""

    trace_id: str
    regions: tuple[TemporalKvRegion, ...]

    def __post_init__(self) -> None:
        if not self.trace_id:
            raise ValueError("trace_id must be non-empty")
        if not self.regions:
            raise ValueError("regions must be non-empty")
        ids = [region.region_id for region in self.regions]
        if len(ids) != len(set(ids)):
            raise ValueError("region_id values must be unique")
        steps = len(self.regions[0].contributions)
        width = len(self.regions[0].contributions[0])
        if any(len(region.contributions) != steps for region in self.regions):
            raise ValueError("all regions must have equal temporal length")
        if any(len(step) != width for region in self.regions for step in region.contributions):
            raise ValueError("all contributions must have equal width")

    @property
    def steps(self) -> int:
        return len(self.regions[0].contributions)

    def full_cache_output(self, step: int) -> tuple[float, ...]:
        if step < 0 or step >= self.steps:
            raise IndexError("step outside trace")
        width = len(self.regions[0].contributions[0])
        return tuple(
            sum(region.contributions[step][index] for region in self.regions)
            for index in range(width)
        )


@dataclass(frozen=True)
class TemporalUtility:
    """Exact synthetic immediate and future removal sensitivity for one region."""

    region_id: str
    decision_step: int
    immediate_l2_delta: float
    future_l2_delta_sum: float
    storage_bytes: int

    @property
    def future_utility_per_byte(self) -> float:
        return self.future_l2_delta_sum / self.storage_bytes


def _l2(values: tuple[float, ...]) -> float:
    return sqrt(sum(value * value for value in values))


def evaluate_temporal_utility(
    trace: TemporalKvTrace,
    region_id: str,
    decision_step: int,
) -> TemporalUtility:
    """Measure exact additive removal sensitivity now and after the decision step.

    The future score is a calibration oracle because it uses post-decision
    synthetic contributions. It must not be exposed to a deployable policy as
    an online feature.
    """

    if decision_step < 0 or decision_step >= trace.steps:
        raise IndexError("decision_step outside trace")
    matches = [region for region in trace.regions if region.region_id == region_id]
    if len(matches) != 1:
        raise KeyError(f"unknown region_id: {region_id}")
    region = matches[0]
    immediate = _l2(region.contributions[decision_step])
    future = sum(_l2(region.contributions[step]) for step in range(decision_step + 1, trace.steps))
    return TemporalUtility(
        region_id=region.region_id,
        decision_step=decision_step,
        immediate_l2_delta=immediate,
        future_l2_delta_sum=future,
        storage_bytes=region.storage_bytes,
    )
