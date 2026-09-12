"""Deterministic retention policies for KV-cache experiments.

This module models *which logical tokens remain addressable*. It deliberately
separates logical retention from allocator residency: keeping fewer logical KV
entries does not, by itself, prove that GPU/HBM allocation shrank.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlidingWindowPolicy:
    """Keep at most the most recent ``window_tokens`` logical KV entries."""

    window_tokens: int

    def __post_init__(self) -> None:
        if self.window_tokens <= 0:
            raise ValueError("window_tokens must be positive")


@dataclass(frozen=True)
class RetentionAccounting:
    """Logical retention accounting for one sequence.

    ``resident_bytes`` is intentionally absent. Physical allocation belongs to
    backend instrumentation and must not be inferred from logical retention.
    """

    input_tokens: int
    retained_tokens: int
    evicted_tokens: int
    bytes_per_token: int
    logical_input_bytes: int
    logical_retained_bytes: int
    logical_evicted_bytes: int

    @property
    def logical_retention_ratio(self) -> float:
        if self.input_tokens == 0:
            return 1.0
        return self.retained_tokens / self.input_tokens


def apply_sliding_window(
    *,
    input_tokens: int,
    bytes_per_token: int,
    policy: SlidingWindowPolicy,
) -> RetentionAccounting:
    """Return exact logical accounting for a deterministic sliding window.

    No latency, bandwidth, GPU-residency, or downstream-quality claim is made.
    Those quantities require separate measured or explicitly estimated records.
    """

    if input_tokens < 0:
        raise ValueError("input_tokens must be non-negative")
    if bytes_per_token <= 0:
        raise ValueError("bytes_per_token must be positive")

    retained_tokens = min(input_tokens, policy.window_tokens)
    evicted_tokens = input_tokens - retained_tokens
    logical_input_bytes = input_tokens * bytes_per_token
    logical_retained_bytes = retained_tokens * bytes_per_token

    return RetentionAccounting(
        input_tokens=input_tokens,
        retained_tokens=retained_tokens,
        evicted_tokens=evicted_tokens,
        bytes_per_token=bytes_per_token,
        logical_input_bytes=logical_input_bytes,
        logical_retained_bytes=logical_retained_bytes,
        logical_evicted_bytes=logical_input_bytes - logical_retained_bytes,
    )
