"""Bind K5 sparse-read accounting to an unchanged full-cache oracle.

Sparse page selection changes which already-stored pages are candidates for a
query read. It does not, by itself, reduce the logical KV cache size. This
module therefore verifies the full-cache baseline and reports candidate-read
coverage separately from unchanged logical cache bytes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .oracle import FullCacheSnapshot, OracleError
from .sparse_read import SparseReadAccounting


@dataclass(frozen=True)
class SparseReadOracleComparison:
    """Auditable structural comparison against one full-cache oracle snapshot."""

    oracle_sequence_length: int
    oracle_logical_bytes: int
    total_pages: int
    selected_pages: int
    skipped_pages: int
    logical_cache_bytes: int
    selected_read_bytes: int

    @property
    def selected_page_fraction(self) -> float:
        if self.total_pages == 0:
            return 0.0
        return self.selected_pages / self.total_pages

    @property
    def candidate_read_fraction(self) -> float:
        if self.logical_cache_bytes == 0:
            return 0.0
        return self.selected_read_bytes / self.logical_cache_bytes


def compare_sparse_read_to_oracle(
    *,
    oracle: FullCacheSnapshot,
    accounting: SparseReadAccounting,
) -> SparseReadOracleComparison:
    """Validate sparse-read accounting against an unchanged full-cache oracle.

    The oracle digest is replay-verified first. The sparse accounting must cover
    exactly the oracle logical byte volume; otherwise comparison fails closed.
    The result deliberately contains no claim about HBM residency, physical
    bandwidth, transfers, latency, throughput, or downstream quality.
    """

    try:
        tuple(oracle.replay())
    except OracleError as exc:
        raise ValueError("oracle snapshot failed replay verification") from exc

    if accounting.logical_cache_bytes != oracle.logical_bytes:
        raise ValueError("sparse logical cache bytes must match oracle logical bytes")
    if len(accounting.selected_pages) + len(accounting.skipped_pages) != accounting.total_pages:
        raise ValueError("selected and skipped pages must partition the page inventory")
    if set(accounting.selected_pages) & set(accounting.skipped_pages):
        raise ValueError("selected and skipped page sets must be disjoint")
    if accounting.selected_read_bytes > accounting.logical_cache_bytes:
        raise ValueError("selected read bytes cannot exceed logical cache bytes")

    return SparseReadOracleComparison(
        oracle_sequence_length=oracle.sequence_length,
        oracle_logical_bytes=oracle.logical_bytes,
        total_pages=accounting.total_pages,
        selected_pages=len(accounting.selected_pages),
        skipped_pages=len(accounting.skipped_pages),
        logical_cache_bytes=accounting.logical_cache_bytes,
        selected_read_bytes=accounting.selected_read_bytes,
    )
