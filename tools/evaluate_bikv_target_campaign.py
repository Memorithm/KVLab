#!/usr/bin/env python3
"""Apply a frozen v2 BIKV decision plan to retained target evidence."""

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
from kvlab.bikv_target_decision import (  # noqa: E402
    BikvTargetDecisionError,
    BikvTargetDecisionPlanV2,
    evaluate_target_campaign,
)
from kvlab.bikv_target_paired_summary import BikvTargetPairedSummaryError  # noqa: E402
from kvlab.bikv_target_protocol import (  # noqa: E402
    BikvTargetProtocolError,
    BikvTargetProtocolV1,
)
from kvlab.bikv_target_run import BikvTargetRunError, BikvTargetRunV1  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("decision_plan", type=Path)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="write canonical decision evidence to this path instead of stdout",
    )
    args = parser.parse_args()

    try:
        protocol = BikvTargetProtocolV1.from_canonical_json(
            args.protocol.read_text(encoding="utf-8")
        )
        campaign = BikvTargetCampaignV1.from_canonical_json_bytes(
            args.campaign.read_bytes()
        )
        decision_plan = BikvTargetDecisionPlanV2.from_canonical_json(
            args.decision_plan.read_text(encoding="utf-8")
        )
        runs = tuple(
            BikvTargetRunV1.from_canonical_json(path.read_text(encoding="utf-8"))
            for path in args.runs
        )
        decision = evaluate_target_campaign(
            protocol=protocol,
            campaign=campaign,
            runs=runs,
            decision_plan=decision_plan,
        )
        payload = decision.canonical_json_bytes()
        if args.output is None:
            sys.stdout.buffer.write(payload + b"\n")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_name(args.output.name + ".tmp")
            temporary.write_bytes(payload)
            temporary.replace(args.output)
            print(
                f"decision_sha256={decision.decision_sha256()} "
                f"disposition={decision.disposition} output={args.output}"
            )
    except (
        OSError,
        UnicodeError,
        BikvTargetProtocolError,
        BikvTargetRunError,
        BikvTargetCampaignError,
        BikvTargetPairedSummaryError,
        BikvTargetDecisionError,
    ) as exc:
        print(f"invalid BIKV target decision input: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
