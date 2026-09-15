#!/usr/bin/env python3
"""Verify one retained BIKV evidence artifact against its canonical receipt.

This command is provenance-only. It validates receipt encoding plus exact source
byte identity; it does not interpret or promote scientific/performance fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kvlab.bikv_evidence_receipt import BikvEvidenceReceiptV1  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify exact BIKV evidence bytes against a canonical KVLab receipt."
    )
    parser.add_argument("--receipt", required=True, type=Path, help="Canonical receipt JSON path")
    parser.add_argument("--source", required=True, type=Path, help="Retained upstream evidence path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        receipt_bytes = args.receipt.read_bytes()
        source_bytes = args.source.read_bytes()
        receipt = BikvEvidenceReceiptV1.from_canonical_json_bytes(receipt_bytes)
        receipt.verify_source_bytes(source_bytes)
    except (OSError, ValueError) as exc:
        print(f"BIKV receipt verification failed: {exc}", file=sys.stderr)
        return 2

    summary = {
        "schema": receipt.schema,
        "producer_repo": receipt.producer_repo,
        "producer_commit": receipt.producer_commit,
        "source_sha256": receipt.source_sha256,
        "source_bytes": receipt.source_bytes,
        "receipt_sha256": receipt.receipt_sha256(),
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
