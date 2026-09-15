#!/usr/bin/env python3
"""Verify one canonical BIKV evidence bundle and its bound provenance files.

The command always validates the canonical bundle encoding and requires exactly
one canonical receipt file for every declared role. When ``--source`` arguments
are supplied, they must cover every role and the exact source bytes are also
verified against each receipt. Source payloads are never parsed or interpreted.

This remains provenance-only: successful verification does not validate any
scientific, quality, latency, traffic, energy, or performance claim contained in
an upstream artifact.
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
        description=(
            "Verify a canonical BIKV evidence bundle against canonical receipts "
            "and, optionally, exact upstream source bytes."
        )
    )
    parser.add_argument("--bundle", required=True, type=Path, help="Canonical bundle JSON path")
    parser.add_argument(
        "--receipt",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="Canonical receipt JSON bound to one bundle role; repeat for every role",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help=(
            "Exact retained upstream artifact bound to one role; when supplied, "
            "repeat for every bundle role"
        ),
    )
    return parser.parse_args()


def parse_role_paths(values: list[str], *, option: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        role, separator, raw_path = value.partition("=")
        if not separator or not role or not raw_path:
            raise ValueError(f"{option} must use ROLE=PATH")
        if role in paths:
            raise ValueError(f"duplicate {option.removeprefix('--')} role: {role}")
        paths[role] = Path(raw_path)
    return paths


def require_exact_roles(
    *,
    kind: str,
    expected_roles: set[str],
    supplied_roles: set[str],
) -> None:
    if supplied_roles != expected_roles:
        missing = sorted(expected_roles - supplied_roles)
        extra = sorted(supplied_roles - expected_roles)
        raise ValueError(f"{kind} roles do not match bundle: missing={missing}, extra={extra}")


def main() -> int:
    args = parse_args()
    try:
        bundle = BikvEvidenceBundleV1.from_canonical_json_bytes(args.bundle.read_bytes())
        receipt_paths = parse_role_paths(args.receipt, option="--receipt")
        source_paths = parse_role_paths(args.source, option="--source")
        expected_roles = {entry.role for entry in bundle.entries}
        require_exact_roles(
            kind="receipt",
            expected_roles=expected_roles,
            supplied_roles=set(receipt_paths),
        )
        if source_paths:
            require_exact_roles(
                kind="source",
                expected_roles=expected_roles,
                supplied_roles=set(source_paths),
            )

        verified_entries: list[dict[str, object]] = []
        for entry in bundle.entries:
            receipt = BikvEvidenceReceiptV1.from_canonical_json_bytes(
                receipt_paths[entry.role].read_bytes()
            )
            entry.verify_receipt(receipt)
            source_verified = False
            if source_paths:
                receipt.verify_source_bytes(source_paths[entry.role].read_bytes())
                source_verified = True
            verified_entries.append(
                {
                    "role": entry.role,
                    "receipt_sha256": entry.receipt_sha256,
                    "producer_repo": entry.producer_repo,
                    "producer_commit": entry.producer_commit,
                    "source_sha256": entry.source_sha256,
                    "source_bytes": entry.source_bytes,
                    "source_verified": source_verified,
                }
            )
    except (OSError, ValueError) as exc:
        print(f"BIKV bundle verification failed: {exc}", file=sys.stderr)
        return 2

    summary = {
        "schema": bundle.schema,
        "bundle_sha256": bundle.bundle_sha256(),
        "entry_count": len(bundle.entries),
        "source_payloads_verified": bool(source_paths),
        "entries": verified_entries,
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
