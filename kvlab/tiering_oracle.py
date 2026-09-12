"""Bind K6 tiering accounting to an unchanged full-cache oracle.

Tier placement changes physical residency, not logical KV size. This module
replay-verifies the full-cache oracle and requires tier accounting to cover the
same logical byte volume before residency fractions can be compared.
"""

from __future__ import annotations

from dataclasses import dataclass

from .oracle import FullCacheSnapshot, OracleError
from .tiering import TieringAccounting


@dataclass(frozen=True)
class TieringOracleComparison:
    """Auditable structural comparison against one full-cache snapshot."""

    oracle_sequence_length: int
    oracle_logical_bytes: int
    logical_cache_bytes: int
    gpu_resident_bytes: int
    host_resident_bytes: int
    secondary_resident_bytes: int
    measured_transfer_bytes: int | None

    @property
    def gpu_resident_fraction(self) -> float:
        if self.logical_cache_bytes == 0:
            return 0.0
        return self.gpu_resident_bytes / self.logical_cache_bytes


def compare_tiering_to_oracle(
    *,
    oracle: FullCacheSnapshot,
    accounting: TieringAccounting,
) -> TieringOracleComparison:
    """Validate tier placement against the unchanged full-cache baseline.

    This comparison makes no latency, bandwidth, throughput, quality, or energy
    claim. Transfer bytes remain absent unless the caller supplied an actual
    measurement to the tiering accounting record.
    """

    try:
        tuple(oracle.replay())
    except OracleError as exc:
        raise ValueError("oracle snapshot failed replay verification") from exc

    if accounting.logical_cache_bytes != oracle.logical_bytes:
        raise ValueError("tiered logical cache bytes must match oracle logical bytes")
    if accounting.accounted_resident_bytes != accounting.logical_cache_bytes:
        raise ValueError("tier residency must partition the logical cache")

    return TieringOracleComparison(
        oracle_sequence_length=oracle.sequence_length,
        oracle_logical_bytes=oracle.logical_bytes,
        logical_cache_bytes=accounting.logical_cache_bytes,
        gpu_resident_bytes=accounting.gpu_resident_bytes,
        host_resident_bytes=accounting.host_resident_bytes,
        secondary_resident_bytes=accounting.secondary_resident_bytes,
        measured_transfer_bytes=accounting.measured_transfer_bytes,
    )
