"""BKV-K3 deterministic sharding contract for multicore qualification.

This module defines how Boolean-KV pages are partitioned and how independently
scanned shard results are merged. It is a correctness/orchestration layer only:
it makes no multicore speedup claim and does not infer CPU topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .boolean_kv import BooleanKvError, PackedBits
from .boolean_kv_cpu import PackedScanResult, scan_packed_pages


class MulticoreScanError(BooleanKvError):
    """Raised when a BKV-K3 sharded scan request is malformed."""


@dataclass(frozen=True)
class PageShard:
    shard_id: int
    start_page: int
    end_page: int

    @property
    def page_count(self) -> int:
        return self.end_page - self.start_page


@dataclass(frozen=True)
class ShardedScanResult:
    selected_pages: tuple[int, ...]
    pages_scanned: int
    signature_bits: int
    bits_compared: int
    shards: tuple[PageShard, ...]


def contiguous_page_shards(page_count: int, workers: int) -> tuple[PageShard, ...]:
    """Partition pages into deterministic, non-overlapping contiguous shards.

    The number of returned shards is ``min(page_count, workers)``. Page counts
    differ by at most one, and earlier shards receive the remainder.
    """

    if not isinstance(page_count, int) or isinstance(page_count, bool) or page_count <= 0:
        raise MulticoreScanError("page_count must be a positive integer")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise MulticoreScanError("workers must be a positive integer")

    shard_count = min(page_count, workers)
    base, remainder = divmod(page_count, shard_count)
    shards: list[PageShard] = []
    start = 0
    for shard_id in range(shard_count):
        size = base + (1 if shard_id < remainder else 0)
        end = start + size
        shards.append(PageShard(shard_id=shard_id, start_page=start, end_page=end))
        start = end
    return tuple(shards)


def scan_sharded_pages(
    *,
    query: PackedBits,
    page_signatures: Sequence[PackedBits],
    max_distance: int,
    workers: int,
) -> ShardedScanResult:
    """Execute the exact packed oracle independently per deterministic shard.

    This function intentionally executes shards serially. It freezes the shard
    semantics and deterministic merge oracle that later thread/process/Rust
    multicore backends must match before their timings are comparable.
    """

    if not page_signatures:
        raise MulticoreScanError("page_signatures must be non-empty")
    shards = contiguous_page_shards(len(page_signatures), workers)

    selected: list[int] = []
    pages_scanned = 0
    bits_compared = 0
    for shard in shards:
        local: PackedScanResult = scan_packed_pages(
            query=query,
            page_signatures=page_signatures[shard.start_page : shard.end_page],
            max_distance=max_distance,
        )
        selected.extend(shard.start_page + page_id for page_id in local.selected_pages)
        pages_scanned += local.pages_scanned
        bits_compared += local.bits_compared

    return ShardedScanResult(
        selected_pages=tuple(selected),
        pages_scanned=pages_scanned,
        signature_bits=query.bit_length,
        bits_compared=bits_compared,
        shards=shards,
    )
