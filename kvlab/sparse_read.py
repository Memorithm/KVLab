"""Deterministic query-aware sparse-read accounting for KV experiments.

Sparse reads model which already-stored KV pages are consulted for a query.
They do not, by themselves, reduce the logical size of the cache. This module
therefore reports selected read volume separately from unchanged cache size.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable


@dataclass(frozen=True)
class PageScore:
    """One query-specific relevance score for an existing KV page."""

    page_id: int
    score: float

    def __post_init__(self) -> None:
        if self.page_id < 0:
            raise ValueError("page_id must be non-negative")
        if not isfinite(self.score):
            raise ValueError("score must be finite")


@dataclass(frozen=True)
class SparseReadAccounting:
    """Exact structural accounting for one sparse page selection.

    ``logical_cache_bytes`` remains the full cache size. ``selected_read_bytes``
    is the maximum page payload volume selected by this policy; it is not a
    hardware bandwidth measurement and must not be reported as such.
    """

    total_pages: int
    selected_pages: tuple[int, ...]
    skipped_pages: tuple[int, ...]
    page_bytes: int
    logical_cache_bytes: int
    selected_read_bytes: int

    @property
    def selected_fraction(self) -> float:
        if self.total_pages == 0:
            return 0.0
        return len(self.selected_pages) / self.total_pages


@dataclass(frozen=True)
class QueryAwarePagePolicy:
    """Select the highest-scoring pages with deterministic tie-breaking."""

    max_pages: int

    def __post_init__(self) -> None:
        if self.max_pages < 0:
            raise ValueError("max_pages must be non-negative")


def select_query_aware_pages(
    *,
    scores: Iterable[PageScore],
    page_bytes: int,
    policy: QueryAwarePagePolicy,
) -> SparseReadAccounting:
    """Select pages by descending score, then ascending page id.

    Page identifiers must be unique. Logical cache size is preserved exactly;
    only the candidate read set is reduced. Residency, transfer volume, latency,
    throughput, and downstream quality require separate measurements.
    """

    if page_bytes <= 0:
        raise ValueError("page_bytes must be positive")

    materialized = tuple(scores)
    page_ids = [item.page_id for item in materialized]
    if len(page_ids) != len(set(page_ids)):
        raise ValueError("page_id values must be unique")

    ranked = sorted(materialized, key=lambda item: (-item.score, item.page_id))
    selected_set = {
        item.page_id for item in ranked[: min(policy.max_pages, len(ranked))]
    }
    selected_pages = tuple(sorted(selected_set))
    skipped_pages = tuple(sorted(set(page_ids) - selected_set))
    logical_cache_bytes = len(materialized) * page_bytes

    return SparseReadAccounting(
        total_pages=len(materialized),
        selected_pages=selected_pages,
        skipped_pages=skipped_pages,
        page_bytes=page_bytes,
        logical_cache_bytes=logical_cache_bytes,
        selected_read_bytes=len(selected_pages) * page_bytes,
    )
