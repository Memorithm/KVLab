#!/usr/bin/env python3
"""Verify one canonical BKV-K8 observation against frozen campaign provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kvlab.bikv_evidence_bundle import BikvEvidenceBundleV1  # noqa: E402
from kvlab.bikv_first_token import BikvFirstTokenObservationError  # noqa: E402
from kvlab.bikv_first_token_v2 import BikvK8FirstTokenObservationV2  # noqa: E402
from kvlab.bikv_target_protocol import (  # noqa: E402
    BikvTargetProtocolError,
    BikvTargetProtocolV1,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_bundle", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("observation", type=Path)
    args = parser.parse_args()

    try:
        bundle = BikvEvidenceBundleV1.from_canonical_json_bytes(
            args.evidence_bundle.read_bytes()
        )
        protocol = BikvTargetProtocolV1.from_canonical_json(
            args.protocol.read_text(encoding="utf-8")
        )
        observation = BikvK8FirstTokenObservationV2.from_canonical_json(
            args.observation.read_text(encoding="utf-8")
        )
        observation.validate_against(evidence_bundle=bundle, protocol=protocol)
    except (OSError, UnicodeError, ValueError, BikvTargetProtocolError, BikvFirstTokenObservationError) as exc:
        print(f"invalid BKV-K8 first-token observation: {exc}", file=sys.stderr)
        return 2

    ratio = observation.numerical_to_boolean_bytes_ratio()
    print(
        json.dumps(
            {
                "boolean_kv_bytes_read": observation.boolean_kv_bytes_read,
                "evidence_bundle_sha256": observation.evidence_bundle_sha256,
                "first_token_boolean_route_consumed": observation.first_token_boolean_route_consumed,
                "first_token_latency_ns": observation.first_token_latency_ns,
                "hardware_fingerprint_sha256": observation.hardware_fingerprint_sha256,
                "historical_signature_rebuilds": observation.historical_signature_rebuilds,
                "numerical_kv_bytes_avoided": observation.numerical_kv_bytes_avoided,
                "numerical_to_boolean_bytes_ratio": None if ratio is None else f"{ratio.numerator}/{ratio.denominator}",
                "observation_sha256": observation.observation_sha256(),
                "protocol_sha256": protocol.protocol_sha256(),
                "schema": observation.schema,
                "timing_source": observation.timing_source,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
