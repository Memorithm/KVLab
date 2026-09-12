"""Deterministic prefix-cache accounting for K2 experiments.

This module reports cache reuse separately from logical KV size. A prefix hit can
reduce duplicated allocation or prefill work in a concrete backend, but it does
not by itself shrink the logical KV state required by the requests.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefixLookup:
    """One prefix-cache lookup observation supplied by an instrumented backend."""

    request_id: str
    prefix_tokens: int
    shared_full_pages: int
    hit: bool

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must be non-empty")
        if self.prefix_tokens < 0:
            raise ValueError("prefix_tokens must be non-negative")
        if self.shared_full_pages < 0:
            raise ValueError("shared_full_pages must be non-negative")
        if not self.hit and self.shared_full_pages != 0:
            raise ValueError("a miss cannot report shared pages")


@dataclass(frozen=True)
class PrefixCacheSummary:
    requests: int
    hits: int
    misses: int
    hit_rate: float
    prefix_tokens_requested: int
    prefix_tokens_hit: int
    shared_full_pages: int

    @property
    def logical_kv_reduction_tokens(self) -> int:
        """Prefix reuse does not reduce the requests' logical KV token count."""

        return 0


def summarize_prefix_cache(lookups: list[PrefixLookup]) -> PrefixCacheSummary:
    """Aggregate explicit prefix-cache lookup observations.

    Empty samples are rejected rather than manufacturing a zero-rate experiment.
    This function does not estimate byte savings or prefill latency savings; those
    require backend-specific measurements recorded by KVLab instrumentation.
    """

    if not lookups:
        raise ValueError("at least one prefix lookup is required")

    hits = sum(1 for lookup in lookups if lookup.hit)
    requests = len(lookups)
    prefix_tokens_requested = sum(lookup.prefix_tokens for lookup in lookups)
    prefix_tokens_hit = sum(lookup.prefix_tokens for lookup in lookups if lookup.hit)
    shared_full_pages = sum(lookup.shared_full_pages for lookup in lookups)

    return PrefixCacheSummary(
        requests=requests,
        hits=hits,
        misses=requests - hits,
        hit_rate=hits / requests,
        prefix_tokens_requested=prefix_tokens_requested,
        prefix_tokens_hit=prefix_tokens_hit,
        shared_full_pages=shared_full_pages,
    )
