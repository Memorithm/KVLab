#!/usr/bin/env python3
"""Verify retained BIKV target evidence and emit a paired descriptive summary."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kvlab.bikv_target_campaign import (  # noqa: E402
    BikvTargetCampaignError,
    BikvTargetCampaignV1,
)
from kvlab.bikv_target_paired_summary import (  # noqa: E402
    BikvTargetPairedSummaryError,
    build_paired_summary,
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
    parser.add_argument(
        "--output",
        type=Path,
        help="write canonical summary bytes to this path instead of stdout",
    )
    args = parser.parse_args()

    try:
        protocol = BikvTargetProtocolV1.from_canonical_json(
            args.protocol.read_text(encoding="utf-8")
        )
        campaign = BikvTargetCampaignV1.from_canonical_json_bytes(
            args.campaign.read_bytes()
        )
        runs = tuple(
            BikvTargetRunV1.from_canonical_json(path.read_text(encoding="utf-8"))
            for path in args.runs
        )
        summary = build_paired_summary(protocol=protocol, campaign=campaign, runs=runs)
        payload = summary.canonical_json_bytes()
        if args.output is None:
            sys.stdout.buffer.write(payload + b"\n")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_name(args.output.name + ".tmp")
            temporary.write_bytes(payload)
            temporary.replace(args.output)
            print(
                f"summary_sha256={summary.summary_sha256()} "
                f"campaign_sha256={summary.campaign_sha256} "
                f"output={args.output}"
            )
    except (
        OSError,
        UnicodeError,
        BikvTargetProtocolError,
        BikvTargetRunError,
        BikvTargetCampaignError,
        BikvTargetPairedSummaryError,
    ) as exc:
        print(f"invalid BIKV paired summary input: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
