"""Deterministic KV cache tiering accounting.

Tiering models where already-materialized KV pages reside. Moving pages between
GPU/HBM, host RAM, and secondary storage does not by itself reduce the logical
size of the cache. This module therefore keeps logical cache size invariant and
reports tier residency separately. Transfer volume is only recorded when it is
explicitly supplied by an instrumented caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class Tier(str, Enum):
    """Supported physical-residency tiers."""

    GPU = "gpu"
    HOST = "host"
    SECONDARY = "secondary"


@dataclass(frozen=True)
class TieredPage:
    """One logical KV page and its current residency tier."""

    page_id: int
    tier: Tier

    def __post_init__(self) -> None:
        if self.page_id < 0:
            raise ValueError("page_id must be non-negative")


@dataclass(frozen=True)
class TieringAccounting:
    """Exact structural accounting for one KV tier placement."""

    total_pages: int
    page_bytes: int
    logical_cache_bytes: int
    gpu_resident_bytes: int
    host_resident_bytes: int
    secondary_resident_bytes: int
    measured_transfer_bytes: int | None

    @property
    def accounted_resident_bytes(self) -> int:
        return (
            self.gpu_resident_bytes
            + self.host_resident_bytes
            + self.secondary_resident_bytes
        )


def account_tiering(
    *,
    pages: Iterable[TieredPage],
    page_bytes: int,
    measured_transfer_bytes: int | None = None,
) -> TieringAccounting:
    """Account logical KV size and residency without inventing transfer data.

    Page identifiers must be unique. ``logical_cache_bytes`` is invariant under
    tier placement. ``measured_transfer_bytes`` is optional and must only be
    supplied from actual instrumentation; ``None`` means the transfer volume was
    not exposed or measured.
    """

    if page_bytes <= 0:
        raise ValueError("page_bytes must be positive")
    if measured_transfer_bytes is not None and measured_transfer_bytes < 0:
        raise ValueError("measured_transfer_bytes must be non-negative")

    materialized = tuple(pages)
    page_ids = [page.page_id for page in materialized]
    if len(page_ids) != len(set(page_ids)):
        raise ValueError("page_id values must be unique")

    counts = {tier: 0 for tier in Tier}
    for page in materialized:
        counts[page.tier] += 1

    logical_cache_bytes = len(materialized) * page_bytes
    accounting = TieringAccounting(
        total_pages=len(materialized),
        page_bytes=page_bytes,
        logical_cache_bytes=logical_cache_bytes,
        gpu_resident_bytes=counts[Tier.GPU] * page_bytes,
        host_resident_bytes=counts[Tier.HOST] * page_bytes,
        secondary_resident_bytes=counts[Tier.SECONDARY] * page_bytes,
        measured_transfer_bytes=measured_transfer_bytes,
    )
    if accounting.accounted_resident_bytes != logical_cache_bytes:
        raise AssertionError("tier residency must partition the logical cache")
    return accounting
