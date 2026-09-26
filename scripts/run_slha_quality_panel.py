#!/usr/bin/env python3
"""Evaluate one frozen SLHA/KVLab score-selection record from JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from kvlab.slha_evidence import SlhaQualityEvidenceV1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import json

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    evidence = SlhaQualityEvidenceV1.capture(
        experiment_id=str(payload["experiment_id"]),
        query_id=str(payload["query_id"]),
        candidate_id=str(payload["candidate_id"]),
        top_k=int(payload["top_k"]),
        reference_scores=tuple(float(value) for value in payload["reference_scores"]),
        selected_ids=tuple(int(value) for value in payload["selected_ids"]),
        candidate_scores=(
            None
            if "candidate_scores" not in payload
            else tuple(float(value) for value in payload["candidate_scores"])
        ),
    )

    encoded = evidence.canonical_json() + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
