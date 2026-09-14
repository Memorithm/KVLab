"""Fail-closed preflight for comparable position-native KV campaigns.

The generic v4 campaign runner deliberately permits selections with different
retained budgets. Comparative claims across policies require a narrower
condition: every candidate must retain the same logical KV budget. This module
adds that condition without changing the generic campaign schema or executor.

A successful preflight is not execution evidence and does not establish model
quality, latency, throughput, HBM release, or avoided physical traffic.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

from .prospect_real_model_campaign_v4 import PositionCampaignSpecV1


class ProspectKvPositionComparisonError(ValueError):
    """Raised when a campaign is not suitable for budget-matched comparison."""


@dataclass(frozen=True, slots=True)
class PositionComparisonPreflight:
    campaign_spec_sha256: str
    policies: tuple[str, ...]
    logical_input_bytes: int
    logical_retained_bytes: int
    logical_evicted_bytes: int
    retained_position_count: int

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def preflight_budget_matched_campaign(
    spec: PositionCampaignSpecV1,
) -> PositionComparisonPreflight:
    """Require one common logical retained budget across all campaign policies."""

    spec.validate()
    _context, _trace, selections = spec.build_execution_inputs()
    if not selections:
        raise ProspectKvPositionComparisonError("campaign has no selections")

    first = selections[0]
    expected_retained_bytes = first.logical_retained_bytes
    expected_retained_count = len(first.retained_positions)
    for selection in selections[1:]:
        if selection.logical_retained_bytes != expected_retained_bytes:
            raise ProspectKvPositionComparisonError(
                "campaign policies do not share one logical retained-byte budget"
            )
        if len(selection.retained_positions) != expected_retained_count:
            raise ProspectKvPositionComparisonError(
                "campaign policies do not share one retained-position count"
            )

    return PositionComparisonPreflight(
        campaign_spec_sha256=spec.sha256,
        policies=tuple(selection.policy for selection in selections),
        logical_input_bytes=first.logical_input_bytes,
        logical_retained_bytes=expected_retained_bytes,
        logical_evicted_bytes=first.logical_evicted_bytes,
        retained_position_count=expected_retained_count,
    )


def _parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that a canonical KVLab position campaign is suitable for "
            "budget-matched policy comparison."
        )
    )
    parser.add_argument("campaign", help="canonical position campaign JSON file")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_cli(argv)
    payload = Path(args.campaign).read_text(encoding="utf-8")
    spec = PositionCampaignSpecV1.from_canonical_json(payload)
    summary = preflight_budget_matched_campaign(spec)
    print(summary.canonical_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
