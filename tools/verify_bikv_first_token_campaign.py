#!/usr/bin/env python3
"""Create a verified BKV-K8 first-token campaign binding receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kvlab.bikv_evidence_bundle import BikvEvidenceBundleV1  # noqa: E402
from kvlab.bikv_first_token_campaign import BikvK8FirstTokenCampaignBindingV1  # noqa: E402
from kvlab.bikv_first_token_v2 import BikvK8FirstTokenObservationV2  # noqa: E402
from kvlab.bikv_target_campaign import BikvTargetCampaignV1  # noqa: E402
from kvlab.bikv_target_protocol import BikvTargetProtocolV1  # noqa: E402
from kvlab.bikv_target_run import BikvTargetRunV1  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("protocol")
    parser.add_argument("campaign")
    parser.add_argument("candidate_run")
    parser.add_argument("observation")
    parser.add_argument(
        "retained_runs",
        nargs="+",
        help="all retained target-run payloads required to reproduce the campaign manifest",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        bundle = BikvEvidenceBundleV1.from_canonical_json_bytes(Path(args.bundle).read_bytes())
        protocol = BikvTargetProtocolV1.from_canonical_json(
            Path(args.protocol).read_text(encoding="utf-8")
        )
        campaign = BikvTargetCampaignV1.from_canonical_json_bytes(
            Path(args.campaign).read_bytes()
        )
        candidate_run = BikvTargetRunV1.from_canonical_json(
            Path(args.candidate_run).read_text(encoding="utf-8")
        )
        observation = BikvK8FirstTokenObservationV2.from_canonical_json(
            Path(args.observation).read_text(encoding="utf-8")
        )
        runs = tuple(
            BikvTargetRunV1.from_canonical_json(Path(path).read_text(encoding="utf-8"))
            for path in args.retained_runs
        )
        binding = BikvK8FirstTokenCampaignBindingV1.from_evidence(
            evidence_bundle=bundle,
            protocol=protocol,
            campaign=campaign,
            retained_runs=runs,
            candidate_run=candidate_run,
            observation=observation,
        )
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    output = json.loads(binding.canonical_json())
    output["binding_sha256"] = binding.binding_sha256()
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
