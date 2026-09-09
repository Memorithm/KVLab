"""Deterministic non-model C1 calibration substrate.

This module deliberately models only a synthetic additive reference world. It
provides exact provenance-friendly byte accounting and single-region removal
interventions before any real-model or confirmatory KV experiment is allowed.
It does not implement FLAT-ATTENTION semantics or NVIDIA placement.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class KvRegion:
    """One synthetic KV region with an explicit storage cost and contribution."""

    region_id: str
    storage_bytes: int
    contribution: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.region_id:
            raise ValueError("region_id must be non-empty")
        if self.storage_bytes <= 0:
            raise ValueError("storage_bytes must be positive")
        if not self.contribution:
            raise ValueError("contribution must be non-empty")


@dataclass(frozen=True)
class SyntheticKvTrace:
    """A deterministic collection of independent synthetic KV contributions."""

    trace_id: str
    regions: tuple[KvRegion, ...]

    def __post_init__(self) -> None:
        if not self.trace_id:
            raise ValueError("trace_id must be non-empty")
        if not self.regions:
            raise ValueError("regions must be non-empty")
        ids = [region.region_id for region in self.regions]
        if len(ids) != len(set(ids)):
            raise ValueError("region_id values must be unique")
        width = len(self.regions[0].contribution)
        if any(len(region.contribution) != width for region in self.regions):
            raise ValueError("all contributions must have equal width")

    @property
    def total_storage_bytes(self) -> int:
        return sum(region.storage_bytes for region in self.regions)

    def full_cache_output(self) -> tuple[float, ...]:
        width = len(self.regions[0].contribution)
        return tuple(
            sum(region.contribution[index] for region in self.regions)
            for index in range(width)
        )


@dataclass(frozen=True)
class RemovalEvaluation:
    """Exact result of removing one synthetic region from the full-cache oracle."""

    trace_id: str
    removed_region_id: str
    full_cache_output: tuple[float, ...]
    intervened_output: tuple[float, ...]
    output_l2_delta: float
    bytes_saved: int
    remaining_bytes: int


def evaluate_removal(trace: SyntheticKvTrace, region_id: str) -> RemovalEvaluation:
    """Remove exactly one region and compare with the synthetic full-cache oracle."""

    matches = [region for region in trace.regions if region.region_id == region_id]
    if len(matches) != 1:
        raise KeyError(f"unknown region_id: {region_id}")
    removed = matches[0]
    full = trace.full_cache_output()
    intervened = tuple(
        value - contribution
        for value, contribution in zip(full, removed.contribution, strict=True)
    )
    delta = sqrt(sum((left - right) ** 2 for left, right in zip(full, intervened, strict=True)))
    return RemovalEvaluation(
        trace_id=trace.trace_id,
        removed_region_id=region_id,
        full_cache_output=full,
        intervened_output=intervened,
        output_l2_delta=delta,
        bytes_saved=removed.storage_bytes,
        remaining_bytes=trace.total_storage_bytes - removed.storage_bytes,
    )
