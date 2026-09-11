"""Structured KV-cache event traces with explicit resource semantics.

The trace records observable cache operations without implying that paging,
prefix reuse, sparse reads, or tiering reduce the logical KV-cache size.
Those mechanisms may instead change residency, traffic, duplication, or work.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class KvEventKind(str, Enum):
    ALLOCATE = "allocate"
    READ = "read"
    WRITE = "write"
    EVICT = "evict"
    MOVE = "move"
    REUSE = "reuse"


class KvResourceEffect(str, Enum):
    LOGICAL_SIZE = "logical_size"
    RESIDENCY = "residency"
    TRANSFER = "transfer"
    FRAGMENTATION = "fragmentation"
    DUPLICATION = "duplication"
    COMPUTE = "compute"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class KvTraceEvent:
    sequence: int
    token_start: int
    token_count: int
    page_id: int | None
    block_id: int | None
    kind: KvEventKind
    resource_effect: KvResourceEffect
    bytes_observed: int | None = None
    source_tier: str | None = None
    destination_tier: str | None = None

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.token_start < 0:
            raise ValueError("token_start must be non-negative")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if self.page_id is not None and self.page_id < 0:
            raise ValueError("page_id must be non-negative when present")
        if self.block_id is not None and self.block_id < 0:
            raise ValueError("block_id must be non-negative when present")
        if self.bytes_observed is not None and self.bytes_observed < 0:
            raise ValueError("bytes_observed must be non-negative when present")
        if self.kind is KvEventKind.MOVE:
            if not self.source_tier or not self.destination_tier:
                raise ValueError("move events require source_tier and destination_tier")
            if self.source_tier == self.destination_tier:
                raise ValueError("move events require distinct source and destination tiers")
        elif self.source_tier is not None or self.destination_tier is not None:
            raise ValueError("tier endpoints are only valid for move events")


@dataclass(frozen=True, slots=True)
class KvTrace:
    events: tuple[KvTraceEvent, ...]

    def __post_init__(self) -> None:
        previous = -1
        for event in self.events:
            if event.sequence <= previous:
                raise ValueError("event sequence must be strictly increasing")
            previous = event.sequence

    def events_for_token(self, token_index: int) -> tuple[KvTraceEvent, ...]:
        if token_index < 0:
            raise ValueError("token_index must be non-negative")
        return tuple(
            event
            for event in self.events
            if event.token_start <= token_index < event.token_start + event.token_count
        )

    def bytes_by_effect(self, effect: KvResourceEffect) -> int | None:
        """Return a total only when every matching event exposes bytes.

        ``None`` means the quantity is not fully exposed and must not be
        reconstructed or silently treated as zero.
        """

        matching = [event for event in self.events if event.resource_effect is effect]
        if not matching:
            return 0
        if any(event.bytes_observed is None for event in matching):
            return None
        return sum(event.bytes_observed or 0 for event in matching)
