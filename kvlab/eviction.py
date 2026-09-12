"""Deterministic token-eviction accounting for KV-cache experiments.

This module models logical eviction only. It does not infer allocator release,
GPU/HBM residency, transfer volume, latency, or downstream quality from the
number of evicted tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class EvictionOrder(str, Enum):
    """Supported deterministic eviction orders."""

    OLDEST_FIRST = "oldest_first"


@dataclass(frozen=True)
class EvictionPolicy:
    """Keep at most ``max_tokens`` logical KV entries."""

    max_tokens: int
    order: EvictionOrder = EvictionOrder.OLDEST_FIRST

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


@dataclass(frozen=True)
class EvictionResult:
    """Exact logical outcome of applying one eviction policy."""

    input_token_ids: tuple[int, ...]
    retained_token_ids: tuple[int, ...]
    evicted_token_ids: tuple[int, ...]
    bytes_per_token: int
    logical_input_bytes: int
    logical_retained_bytes: int
    logical_evicted_bytes: int

    @property
    def retained_tokens(self) -> int:
        return len(self.retained_token_ids)

    @property
    def evicted_tokens(self) -> int:
        return len(self.evicted_token_ids)


def apply_eviction(
    *,
    token_ids: Sequence[int],
    bytes_per_token: int,
    policy: EvictionPolicy,
) -> EvictionResult:
    """Apply a deterministic logical eviction policy.

    ``token_ids`` are ordered from oldest to newest. Duplicate identifiers are
    rejected because they make audit traces ambiguous. Physical residency and
    performance effects must be recorded independently by instrumentation.
    """

    if bytes_per_token <= 0:
        raise ValueError("bytes_per_token must be positive")

    ordered = tuple(token_ids)
    if any(token_id < 0 for token_id in ordered):
        raise ValueError("token ids must be non-negative")
    if len(set(ordered)) != len(ordered):
        raise ValueError("token ids must be unique")
    if policy.order is not EvictionOrder.OLDEST_FIRST:
        raise ValueError(f"unsupported eviction order: {policy.order}")

    evict_count = max(0, len(ordered) - policy.max_tokens)
    evicted = ordered[:evict_count]
    retained = ordered[evict_count:]
    logical_input_bytes = len(ordered) * bytes_per_token
    logical_retained_bytes = len(retained) * bytes_per_token

    return EvictionResult(
        input_token_ids=ordered,
        retained_token_ids=retained,
        evicted_token_ids=evicted,
        bytes_per_token=bytes_per_token,
        logical_input_bytes=logical_input_bytes,
        logical_retained_bytes=logical_retained_bytes,
        logical_evicted_bytes=logical_input_bytes - logical_retained_bytes,
    )
