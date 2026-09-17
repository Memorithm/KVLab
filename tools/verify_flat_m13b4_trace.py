#!/usr/bin/env python3
"""Verify one canonical FLAT-ATTENTION M13B.4 trace evidence envelope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kvlab.flat_m13b4_trace import (  # noqa: E402
    FLAT_M13B4_REFERENCE_REVISION,
    FlatM13B4TraceError,
    FlatM13B4TraceV1,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    try:
        payload = args.trace.read_bytes()
        trace = FlatM13B4TraceV1.from_canonical_json_bytes(payload)
    except (OSError, FlatM13B4TraceError) as exc:
        print(f"invalid FLAT M13B.4 trace evidence: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "event_count": len(trace.events),
                "flat_reference_revision": FLAT_M13B4_REFERENCE_REVISION,
                "scheduling_variant": trace.scheduling_variant,
                "schema": trace.schema,
                "scope": trace.scope,
                "timing_source": trace.timing_source,
                "trace_sha256": trace.trace_sha256(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
