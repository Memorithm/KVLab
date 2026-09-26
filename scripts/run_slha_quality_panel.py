#!/usr/bin/env python3
"""Evaluate one frozen SLHA/KVLab score-selection record from JSON."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from kvlab.slha_quality import evaluate_ranking, evaluate_selection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    top_k = int(payload["top_k"])
    reference = tuple(float(value) for value in payload["reference_scores"])
    selected = tuple(int(value) for value in payload["selected_ids"])

    out = {
        "schema": "kvlab.slha-quality-panel/v1",
        "selection": dataclasses.asdict(
            evaluate_selection(reference, selected, top_k=top_k)
        ),
    }
    if "candidate_scores" in payload:
        candidate = tuple(float(value) for value in payload["candidate_scores"])
        out["ranking"] = dataclasses.asdict(
            evaluate_ranking(reference, candidate, top_k=top_k)
        )

    encoded = json.dumps(out, sort_keys=True, separators=(",", ":")) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
