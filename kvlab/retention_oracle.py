"""Bind K4 logical retention accounting to an unchanged full-cache oracle.

The comparison in this module is deliberately limited to logical KV state. It
verifies that the candidate accounting starts from exactly the oracle sequence
length and logical byte count before reporting retained/evicted logical state.
It does not infer allocator release, GPU/HBM residency, transfer volume,
bandwidth, latency, throughput, or downstream quality.
"""

from __future__ import annotations

from dataclasses import dataclass

from .oracle import FullCacheSnapshot, OracleError


@dataclass(frozen=True)
class RetentionOracleComparison:
    """Auditable logical comparison against one full-cache oracle snapshot."""

    oracle_sequence_length: int
    oracle_logical_bytes: int
    retained_tokens: int
    retained_logical_bytes: int
    logical_reduction_tokens: int
    logical_reduction_bytes: int

    @property
    def logical_retention_ratio(self) -> float:
        return self.retained_tokens / self.oracle_sequence_length


def compare_retention_to_oracle(
    *,
    oracle: FullCacheSnapshot,
    candidate_input_tokens: int,
    candidate_logical_input_bytes: int,
    retained_tokens: int,
    retained_logical_bytes: int,
) -> RetentionOracleComparison:
    """Compare candidate logical retention to an unchanged full-cache oracle.

    The oracle digest is replay-verified first. Candidate input accounting must
    exactly match the oracle baseline; otherwise comparison fails closed rather
    than silently comparing different contexts or representations.
    """

    try:
        tuple(oracle.replay())
    except OracleError as exc:
        raise ValueError("oracle snapshot failed replay verification") from exc

    if candidate_input_tokens != oracle.sequence_length:
        raise ValueError("candidate input token count must match oracle sequence length")
    if candidate_logical_input_bytes != oracle.logical_bytes:
        raise ValueError("candidate logical input bytes must match oracle logical bytes")
    if retained_tokens < 0 or retained_tokens > candidate_input_tokens:
        raise ValueError("retained_tokens must be within the candidate input range")
    if retained_logical_bytes < 0 or retained_logical_bytes > candidate_logical_input_bytes:
        raise ValueError("retained_logical_bytes must be within the candidate input range")
    if retained_tokens == 0 and retained_logical_bytes != 0:
        raise ValueError("zero retained tokens require zero retained logical bytes")
    if retained_tokens > 0 and retained_logical_bytes == 0:
        raise ValueError("retained tokens require positive retained logical bytes")

    return RetentionOracleComparison(
        oracle_sequence_length=oracle.sequence_length,
        oracle_logical_bytes=oracle.logical_bytes,
        retained_tokens=retained_tokens,
        retained_logical_bytes=retained_logical_bytes,
        logical_reduction_tokens=oracle.sequence_length - retained_tokens,
        logical_reduction_bytes=oracle.logical_bytes - retained_logical_bytes,
    )
