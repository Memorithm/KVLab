#!/usr/bin/env python3
"""Run the BKV-K2 deterministic packed CPU scan benchmark.

Example:
    python tools/benchmark_bkv_cpu.py --pages 100000 --signature-bits 256 \
        --max-distance 96 --cpu 0 > bkv-k2.json

Capture the full host/NUMA evidence separately with ``tools/capture_bkv_host.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kvlab.boolean_kv_cpu import benchmark_packed_scan
from kvlab.boolean_kv_signatures import random_control_signature


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _pin_cpu(cpu: int | None) -> dict[str, object]:
    if cpu is None:
        return {"requested": None, "applied": False, "reason": "not requested"}
    if not hasattr(os, "sched_setaffinity"):
        return {
            "requested": cpu,
            "applied": False,
            "reason": "os.sched_setaffinity unavailable",
        }
    try:
        os.sched_setaffinity(0, {cpu})
    except (OSError, ValueError) as exc:
        return {
            "requested": cpu,
            "applied": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {"requested": cpu, "applied": True, "reason": None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=_positive_int, default=10_000)
    parser.add_argument("--signature-bits", type=_positive_int, default=256)
    parser.add_argument("--max-distance", type=_non_negative_int, default=96)
    parser.add_argument("--seed", type=_non_negative_int, default=1)
    parser.add_argument("--warmup", type=_positive_int, default=5)
    parser.add_argument("--queries", type=_positive_int, default=25)
    parser.add_argument("--cpu", type=_non_negative_int)
    args = parser.parse_args()

    if args.max_distance > args.signature_bits:
        parser.error("--max-distance cannot exceed --signature-bits")

    affinity = _pin_cpu(args.cpu)
    query = random_control_signature(
        bit_length=args.signature_bits,
        seed=args.seed,
        identity=0,
    )
    pages = [
        random_control_signature(
            bit_length=args.signature_bits,
            seed=args.seed,
            identity=page_id + 1,
        )
        for page_id in range(args.pages)
    ]

    report = benchmark_packed_scan(
        query=query,
        page_signatures=pages,
        max_distance=args.max_distance,
        warmup_queries=args.warmup,
        measured_queries=args.queries,
    )
    payload = {
        "schema_version": 1,
        "experiment": "BKV-K2-packed-cpu-baseline",
        "semantic_note": (
            "synthetic random-control signatures; timing is a local systems baseline, "
            "not evidence of attention quality"
        ),
        "parameters": {
            "pages": args.pages,
            "signature_bits": args.signature_bits,
            "max_distance": args.max_distance,
            "seed": args.seed,
        },
        "affinity": affinity,
        "runtime": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "system": platform.system(),
            "logical_cpu_count": os.cpu_count(),
        },
        "benchmark": asdict(report),
        "host_evidence_command": "python tools/capture_bkv_host.py",
    }
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
