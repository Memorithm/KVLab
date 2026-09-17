#!/usr/bin/env python3
"""Verify one canonical FLAT M13B.5 Boolean-KV selection envelope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kvlab.flat_boolean_kv_selection import (  # noqa: E402
    FLAT_BOOLEAN_KV_SELECTION_REFERENCE_REVISION,
    FlatBooleanKvSelectionError,
    FlatBooleanKvSelectionV1,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection", type=Path)
    args = parser.parse_args()
    try:
        payload = args.selection.read_bytes()
        selection = FlatBooleanKvSelectionV1.from_canonical_json_bytes(payload)
    except (OSError, FlatBooleanKvSelectionError) as exc:
        print(f"invalid FLAT Boolean-KV selection evidence: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "boolean_key_bytes_read": selection.boolean_key_bytes_read,
                "flat_reference_revision": FLAT_BOOLEAN_KV_SELECTION_REFERENCE_REVISION,
                "mapped_pages": selection.mapped_pages,
                "schema": selection.schema,
                "selected_page_count": len(selection.selected_pages),
                "selection_sha256": selection.selection_sha256(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
