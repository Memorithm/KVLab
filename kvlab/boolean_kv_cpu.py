"""BKV-K2 packed CPU scan baseline and measurement protocol.

This module is intentionally simple and deterministic.  It establishes a
single-process packed-bit reference that later Rust/SIMD/multicore/NUMA paths
must match exactly before performance comparisons are accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
import statistics
import time
from typing import Sequence

from .boolean_kv import BooleanKvError, PackedBits


class CpuPackedScanError(BooleanKvError):
    """Raised when a packed CPU scan request is malformed."""


@dataclass(frozen=True)
class PackedScanResult:
    selected_pages: tuple[int, ...]
    pages_scanned: int
    signature_bits: int
    bits_compared: int


@dataclass(frozen=True)
class PackedScanBenchmark:
    pages: int
    signature_bits: int
    warmup_queries: int
    measured_queries: int
    median_query_ns: int
    min_query_ns: int
    max_query_ns: int
    pages_per_second: float
    bits_per_second: float
    queries_per_second: float
    selected_pages: tuple[int, ...]


def packed_hamming_distance(left: PackedBits, right: PackedBits) -> int:
    """Exact packed Hamming distance using native integer ``bit_count``."""

    if left.bit_length != right.bit_length:
        raise CpuPackedScanError(
            f"signature width mismatch: {left.bit_length} != {right.bit_length}"
        )
    return sum((a ^ b).bit_count() for a, b in zip(left.words, right.words))


def scan_packed_pages(
    *,
    query: PackedBits,
    page_signatures: Sequence[PackedBits],
    max_distance: int,
) -> PackedScanResult:
    """Scan every page in deterministic page order and return accepted ids."""

    if not page_signatures:
        raise CpuPackedScanError("page_signatures must be non-empty")
    if (
        not isinstance(max_distance, int)
        or isinstance(max_distance, bool)
        or max_distance < 0
        or max_distance > query.bit_length
    ):
        raise CpuPackedScanError("max_distance must be in [0, signature_bits]")
    if any(signature.bit_length != query.bit_length for signature in page_signatures):
        raise CpuPackedScanError("all page signatures must match query width")

    selected = tuple(
        page_id
        for page_id, signature in enumerate(page_signatures)
        if packed_hamming_distance(query, signature) <= max_distance
    )
    pages = len(page_signatures)
    return PackedScanResult(
        selected_pages=selected,
        pages_scanned=pages,
        signature_bits=query.bit_length,
        bits_compared=pages * query.bit_length,
    )


def benchmark_packed_scan(
    *,
    query: PackedBits,
    page_signatures: Sequence[PackedBits],
    max_distance: int,
    warmup_queries: int = 5,
    measured_queries: int = 25,
) -> PackedScanBenchmark:
    """Measure the exact BKV-K2 scalar/packed baseline.

    Timings include the full page scan and candidate tuple construction.  They
    do not include signature generation, file I/O or numerical KV access.
    Results are device-local evidence only and are not universal performance
    claims.
    """

    for name, value in (
        ("warmup_queries", warmup_queries),
        ("measured_queries", measured_queries),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise CpuPackedScanError(f"{name} must be a positive integer")

    reference = scan_packed_pages(
        query=query,
        page_signatures=page_signatures,
        max_distance=max_distance,
    )

    for _ in range(warmup_queries):
        observed = scan_packed_pages(
            query=query,
            page_signatures=page_signatures,
            max_distance=max_distance,
        )
        if observed.selected_pages != reference.selected_pages:
            raise RuntimeError("packed scan became non-deterministic during warmup")

    durations_ns: list[int] = []
    for _ in range(measured_queries):
        started = time.perf_counter_ns()
        observed = scan_packed_pages(
            query=query,
            page_signatures=page_signatures,
            max_distance=max_distance,
        )
        elapsed = time.perf_counter_ns() - started
        if observed.selected_pages != reference.selected_pages:
            raise RuntimeError("packed scan became non-deterministic during measurement")
        durations_ns.append(elapsed)

    median_ns = int(statistics.median(durations_ns))
    if median_ns <= 0:
        raise RuntimeError("timer returned a non-positive median duration")
    seconds = median_ns / 1_000_000_000
    pages_per_second = reference.pages_scanned / seconds
    bits_per_second = reference.bits_compared / seconds
    queries_per_second = 1.0 / seconds
    return PackedScanBenchmark(
        pages=reference.pages_scanned,
        signature_bits=reference.signature_bits,
        warmup_queries=warmup_queries,
        measured_queries=measured_queries,
        median_query_ns=median_ns,
        min_query_ns=min(durations_ns),
        max_query_ns=max(durations_ns),
        pages_per_second=pages_per_second,
        bits_per_second=bits_per_second,
        queries_per_second=queries_per_second,
        selected_pages=reference.selected_pages,
    )
