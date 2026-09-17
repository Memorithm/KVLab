#!/usr/bin/env python3
"""Verify an exact BKV-K8 observation against one retained candidate target run."""

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
from kvlab.bikv_target_protocol import BikvTargetProtocolV1  # noqa: E402
from kvlab.bikv_target_run import BikvTargetRunV1  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', type=Path)
    parser.add_argument('protocol', type=Path)
    parser.add_argument('run', type=Path)
    parser.add_argument('observation', type=Path)
    args = parser.parse_args()
    try:
        bundle = BikvEvidenceBundleV1.from_canonical_json_bytes(args.bundle.read_bytes())
        protocol = BikvTargetProtocolV1.from_canonical_json(args.protocol.read_text())
        run = BikvTargetRunV1.from_canonical_json(args.run.read_text())
        observation = BikvK8FirstTokenObservationV2.from_canonical_json(
            args.observation.read_text()
        )
        observation.validate_against_target_run(
            evidence_bundle=bundle,
            protocol=protocol,
            run=run,
        )
    except (OSError, ValueError, BikvFirstTokenObservationError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                'schema': 'kvlab.bkv-k8-target-run-binding.v1',
                'protocol_sha256': protocol.protocol_sha256(),
                'run_sha256': run.run_sha256(),
                'attempt_id': run.attempt_id,
                'observation_sha256': observation.observation_sha256(),
                'evidence_bundle_sha256': bundle.bundle_sha256(),
            },
            sort_keys=True,
            separators=(',', ':'),
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
