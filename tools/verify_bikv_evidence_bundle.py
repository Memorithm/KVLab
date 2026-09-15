#!/usr/bin/env python3
"""Verify one canonical BIKV evidence bundle and its bound receipt files.

This command is provenance-only. It validates the canonical bundle encoding,
requires exactly one canonical receipt file for every declared role, and checks
that each supplied receipt matches the identity frozen in the bundle. It does
not open or interpret the upstream scientific evidence payloads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kvlab.bikv_evidence_bundle import BikvEvidenceBundleV1  # noqa: E402
from kvlab.bikv_evidence_receipt import BikvEvidenceReceiptV1  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a canonical BIKV evidence bundle against canonical receipt files."
    )
    parser.add_argument("--bundle", required=True, type=Path, help="Canonical bundle JSON path")
    parser.add_argument(
        "--receipt",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="Canonical receipt JSON bound to one bundle role; repeat for every role",
    )
    return parser.parse_args()


def parse_receipt_paths(values: list[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        role, separator, raw_path = value.partition("=")
        if not separator or not role or not raw_path:
            raise ValueError("--receipt must use ROLE=PATH")
        if role in paths:
            raise ValueError(f"duplicate receipt role: {role}")
        paths[role] = Path(raw_path)
    return paths


def main() -> int:
    args = parse_args()
    try:
        bundle = BikvEvidenceBundleV1.from_canonical_json_bytes(args.bundle.read_bytes())
        receipt_paths = parse_receipt_paths(args.receipt)
        expected_roles = {entry.role for entry in bundle.entries}
        supplied_roles = set(receipt_paths)
        if supplied_roles != expected_roles:
            missing = sorted(expected_roles - supplied_roles)
            extra = sorted(supplied_roles - expected_roles)
            raise ValueError(f"receipt roles do not match bundle: missing={missing}, extra={extra}")

        verified_entries: list[dict[str, object]] = []
        for entry in bundle.entries:
            receipt = BikvEvidenceReceiptV1.from_canonical_json_bytes(
                receipt_paths[entry.role].read_bytes()
            )
            entry.verify_receipt(receipt)
            verified_entries.append(
                {
                    "role": entry.role,
                    "receipt_sha256": entry.receipt_sha256,
                    "producer_repo": entry.producer_repo,
                    "producer_commit": entry.producer_commit,
                    "source_sha256": entry.source_sha256,
                    "source_bytes": entry.source_bytes,
                }
            )
    except (OSError, ValueError) as exc:
        print(f"BIKV bundle verification failed: {exc}", file=sys.stderr)
        return 2

    summary = {
        "schema": bundle.schema,
        "bundle_sha256": bundle.bundle_sha256(),
        "entry_count": len(bundle.entries),
        "entries": verified_entries,
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
