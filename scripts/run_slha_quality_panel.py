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
        experiment_id=payload["experiment_id"],
        query_id=payload["query_id"],
        candidate_id=payload["candidate_id"],
        top_k=payload["top_k"],
        reference_scores=payload["reference_scores"],
        selected_ids=payload["selected_ids"],
        candidate_scores=(
            None
            if "candidate_scores" not in payload
            else payload["candidate_scores"]
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
