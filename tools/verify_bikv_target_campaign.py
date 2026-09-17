#!/usr/bin/env python3
"""Verify one canonical BIKV campaign manifest against retained run records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kvlab.bikv_target_campaign import (  # noqa: E402
    BikvTargetCampaignError,
    BikvTargetCampaignV1,
)
from kvlab.bikv_target_protocol import (  # noqa: E402
    BikvTargetProtocolError,
    BikvTargetProtocolV1,
)
from kvlab.bikv_target_run import BikvTargetRunError, BikvTargetRunV1  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        protocol = BikvTargetProtocolV1.from_canonical_json(
            args.protocol.read_text(encoding="utf-8")
        )
        campaign = BikvTargetCampaignV1.from_canonical_json_bytes(
            args.campaign.read_bytes()
        )
        campaign.validate_against(protocol)
        runs = tuple(
            BikvTargetRunV1.from_canonical_json(path.read_text(encoding="utf-8"))
            for path in args.runs
        )
        campaign.verify_runs(protocol=protocol, runs=runs)
    except (
        OSError,
        UnicodeError,
        BikvTargetProtocolError,
        BikvTargetRunError,
        BikvTargetCampaignError,
    ) as exc:
        print(f"invalid BIKV target campaign: {exc}", file=sys.stderr)
        return 2

    completed = sum(run.status == "completed" for run in campaign.runs)
    failed = sum(run.status == "failed" for run in campaign.runs)
    print(
        json.dumps(
            {
                "campaign_id": campaign.campaign_id,
                "campaign_sha256": campaign.campaign_sha256(),
                "completed_attempts": completed,
                "failed_attempts": failed,
                "protocol_sha256": campaign.protocol_sha256,
                "retained_attempts": len(campaign.runs),
                "schema": campaign.schema,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
