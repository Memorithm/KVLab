"""Validate and summarize paired T430 NUMA wall-clock samples."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


REQUIRED_FIELDS = (
    "repetition",
    "execution_order",
    "shard_launch_order",
    "monolithic_e2e_ns",
    "shard0_scan_ns",
    "shard1_scan_ns",
    "pair_scan_wall_ns",
    "merge_ns",
    "pair_e2e_ns",
    "selected_pages",
    "candidate_equality",
)


def _positive_int(row: dict[str, str], field: str) -> int:
    try:
        value = int(row[field])
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid {field}") from error
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def integer_median(values: Iterable[int]) -> int:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("at least one sample is required")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2


def load_samples(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REQUIRED_FIELDS:
            raise ValueError("unexpected paired-sample CSV schema")
        rows = list(reader)
    if not rows:
        raise ValueError("paired-sample CSV is empty")

    expected_repetitions = list(range(1, len(rows) + 1))
    repetitions = [_positive_int(row, "repetition") for row in rows]
    if repetitions != expected_repetitions:
        raise ValueError("repetitions must be contiguous and ordered")

    selected_counts = set()
    execution_orders = set()
    shard_orders = set()
    for index, row in enumerate(rows, start=1):
        execution_order = row["execution_order"]
        shard_order = row["shard_launch_order"]
        if execution_order not in {"monolithic-first", "dual-first"}:
            raise ValueError("invalid execution_order")
        if shard_order not in {"shard0-first", "shard1-first"}:
            raise ValueError("invalid shard_launch_order")
        execution_orders.add(execution_order)
        shard_orders.add(shard_order)
        expected_execution_order = (
            "monolithic-first" if index % 2 else "dual-first"
        )
        expected_shard_order = "shard0-first" if index % 2 else "shard1-first"
        if execution_order != expected_execution_order:
            raise ValueError("execution order must alternate every repetition")
        if shard_order != expected_shard_order:
            raise ValueError("shard launch order must alternate every repetition")
        if row["candidate_equality"] != "exact":
            raise ValueError("candidate equality must be exact in every repetition")

        pair_wall = _positive_int(row, "pair_scan_wall_ns")
        merge = _positive_int(row, "merge_ns")
        pair_e2e = _positive_int(row, "pair_e2e_ns")
        if pair_e2e != pair_wall + merge:
            raise ValueError("pair_e2e_ns must include scan wall time and merge")
        for field in (
            "monolithic_e2e_ns",
            "shard0_scan_ns",
            "shard1_scan_ns",
        ):
            _positive_int(row, field)
        selected_counts.add(_positive_int(row, "selected_pages"))

    if len(rows) < 3:
        raise ValueError("at least three paired samples are required")
    if execution_orders != {"monolithic-first", "dual-first"}:
        raise ValueError("execution order was not alternated")
    if shard_orders != {"shard0-first", "shard1-first"}:
        raise ValueError("shard launch order was not alternated")
    if len(selected_counts) != 1:
        raise ValueError("selected-page count changed between repetitions")
    return rows


def summarize(
    rows: list[dict[str, str]], total_pages: int, signature_bits: int
) -> dict[str, str]:
    if total_pages <= 0 or signature_bits <= 0:
        raise ValueError("total_pages and signature_bits must be positive")

    monolithic = integer_median(int(row["monolithic_e2e_ns"]) for row in rows)
    pair_e2e = integer_median(int(row["pair_e2e_ns"]) for row in rows)
    pair_scan = integer_median(int(row["pair_scan_wall_ns"]) for row in rows)
    merge = integer_median(int(row["merge_ns"]) for row in rows)
    shard0 = integer_median(int(row["shard0_scan_ns"]) for row in rows)
    shard1 = integer_median(int(row["shard1_scan_ns"]) for row in rows)
    logical_bytes = total_pages * signature_bits / 8

    return {
        "schema": "bkv-k4-distinct-dual-local-v2",
        "sample_count": str(len(rows)),
        "latency_method": "paired-process-wall-clock-including-merge",
        "execution_order": "alternated",
        "shard_launch_order": "alternated",
        "monolithic_e2e_median_ns": str(monolithic),
        "shard0_reported_scan_median_ns": str(shard0),
        "shard1_reported_scan_median_ns": str(shard1),
        "pair_scan_wall_median_ns": str(pair_scan),
        "merge_median_ns": str(merge),
        "dual_pair_e2e_median_ns": str(pair_e2e),
        "paired_speedup_vs_monolithic": f"{monolithic / pair_e2e:.6f}",
        "dual_pages_per_second": f"{total_pages / (pair_e2e / 1e9):.6f}",
        "dual_logical_gb_per_second": f"{logical_bytes / (pair_e2e / 1e9) / 1e9:.6f}",
        "selected_pages": rows[0]["selected_pages"],
        "candidate_equality": "exact-every-repetition",
        "performance_claim": "host-and-workload-only",
    }


def write_summary(path: Path, summary: dict[str, str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("total_pages", type=int)
    parser.add_argument("signature_bits", type=int)
    args = parser.parse_args()
    rows = load_samples(args.samples)
    write_summary(
        args.summary,
        summarize(rows, args.total_pages, args.signature_bits),
    )


if __name__ == "__main__":
    main()
