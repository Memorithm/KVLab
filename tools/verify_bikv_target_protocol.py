#!/usr/bin/env python3
"""Verify one canonical frozen BIKV target-host/model protocol."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    args = parser.parse_args()
    try:
        payload = args.protocol.read_text(encoding="utf-8")
        protocol = BikvTargetProtocolV1.from_canonical_json(payload)
    except (OSError, UnicodeError, BikvTargetProtocolError) as exc:
        print(f"invalid BIKV target protocol: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "campaign_id": protocol.campaign_id,
                "phase": protocol.phase,
                "partition_role": protocol.partition_role,
                "protocol_sha256": protocol.protocol_sha256(),
                "schema": protocol.schema,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
