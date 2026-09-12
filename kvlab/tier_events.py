"""Measured KV tier movement events.

This module records movement observations emitted by an instrumented runtime.
It does not infer transfers from residency changes: every byte counted here
must arrive as an explicit measured event from the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from kvlab.tiering import Tier


class TierMovementKind(str, Enum):
    """Direction of one observed physical KV movement."""

    PROMOTE = "promote"
    DEMOTE = "demote"


_TIER_RANK = {
    Tier.SECONDARY: 0,
    Tier.HOST: 1,
    Tier.GPU: 2,
}


@dataclass(frozen=True)
class TierMovementEvent:
    """One runtime-reported movement between two distinct residency tiers."""

    page_id: int
    source: Tier
    destination: Tier
    bytes_moved: int
    kind: TierMovementKind

    def __post_init__(self) -> None:
        if self.page_id < 0:
            raise ValueError("page_id must be non-negative")
        if self.source == self.destination:
            raise ValueError("source and destination tiers must differ")
        if self.bytes_moved <= 0:
            raise ValueError("bytes_moved must be positive")

        expected_kind = (
            TierMovementKind.PROMOTE
            if _TIER_RANK[self.destination] > _TIER_RANK[self.source]
            else TierMovementKind.DEMOTE
        )
        if self.kind is not expected_kind:
            raise ValueError(
                f"movement kind {self.kind.value} contradicts "
                f"{self.source.value}->{self.destination.value}"
            )


@dataclass(frozen=True)
class TierMovementSummary:
    """Exact aggregation of explicitly measured movement events."""

    event_count: int
    promoted_bytes: int
    demoted_bytes: int
    gpu_in_bytes: int
    gpu_out_bytes: int
    host_in_bytes: int
    host_out_bytes: int
    secondary_in_bytes: int
    secondary_out_bytes: int

    @property
    def total_measured_transfer_bytes(self) -> int:
        return self.promoted_bytes + self.demoted_bytes


def summarize_tier_movements(events: Iterable[TierMovementEvent]) -> TierMovementSummary:
    """Aggregate runtime-reported events without synthesizing missing traffic."""

    materialized = tuple(events)
    promoted = 0
    demoted = 0
    inbound = {tier: 0 for tier in Tier}
    outbound = {tier: 0 for tier in Tier}

    for event in materialized:
        outbound[event.source] += event.bytes_moved
        inbound[event.destination] += event.bytes_moved
        if event.kind is TierMovementKind.PROMOTE:
            promoted += event.bytes_moved
        else:
            demoted += event.bytes_moved

    return TierMovementSummary(
        event_count=len(materialized),
        promoted_bytes=promoted,
        demoted_bytes=demoted,
        gpu_in_bytes=inbound[Tier.GPU],
        gpu_out_bytes=outbound[Tier.GPU],
        host_in_bytes=inbound[Tier.HOST],
        host_out_bytes=outbound[Tier.HOST],
        secondary_in_bytes=inbound[Tier.SECONDARY],
        secondary_out_bytes=outbound[Tier.SECONDARY],
    )
