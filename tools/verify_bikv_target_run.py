#!/usr/bin/env python3
"""Verify one canonical BIKV target-run record against its frozen protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kvlab.bikv_target_protocol import (  # noqa: E402
    BikvTargetProtocolError,
    BikvTargetProtocolV1,
)
from kvlab.bikv_target_run import BikvTargetRunError, BikvTargetRunV1  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    try:
        protocol = BikvTargetProtocolV1.from_canonical_json(
            args.protocol.read_text(encoding="utf-8")
        )
        run = BikvTargetRunV1.from_canonical_json(args.run.read_text(encoding="utf-8"))
        run.validate_against(protocol)
    except (OSError, UnicodeError, BikvTargetProtocolError, BikvTargetRunError) as exc:
        print(f"invalid BIKV target run: {exc}", file=sys.stderr)
        return 2

    status_counts = {"measured": 0, "not_exposed": 0, "failed": 0}
    for metric in run.metrics:
        status_counts[metric.status] += 1
    print(
        json.dumps(
            {
                "attempt_id": run.attempt_id,
                "campaign_id": run.campaign_id,
                "metric_status_counts": status_counts,
                "protocol_sha256": run.protocol_sha256,
                "run_sha256": run.run_sha256(),
                "schema": run.schema,
                "status": run.status,
                "variant": run.variant,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
