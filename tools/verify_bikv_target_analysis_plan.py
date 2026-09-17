#!/usr/bin/env python3
"""Verify a preregistered BIKV target analysis plan against its protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kvlab.bikv_target_analysis_plan import (  # noqa: E402
    BikvTargetAnalysisPlanError,
    BikvTargetAnalysisPlanV1,
)
from kvlab.bikv_target_protocol import (  # noqa: E402
    BikvTargetProtocolError,
    BikvTargetProtocolV1,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("analysis_plan", type=Path)
    args = parser.parse_args()

    try:
        protocol = BikvTargetProtocolV1.from_canonical_json(
            args.protocol.read_text(encoding="utf-8")
        )
        plan = BikvTargetAnalysisPlanV1.from_canonical_json(
            args.analysis_plan.read_text(encoding="utf-8")
        )
        plan.validate_against(protocol)
    except (
        OSError,
        UnicodeError,
        BikvTargetProtocolError,
        BikvTargetAnalysisPlanError,
    ) as exc:
        print(f"invalid BIKV target analysis plan: {exc}", file=sys.stderr)
        return 2

    print(
        f"plan_sha256={plan.plan_sha256()} "
        f"protocol_sha256={plan.protocol_sha256} "
        f"campaign_id={plan.campaign_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
