#!/usr/bin/env python3
"""Emit the deterministic SKV-1 synthetic mass panel as canonical JSONL."""

from kvlab.slha_synthetic_panel import run_panel


def main() -> int:
    for record in run_panel():
        print(record.canonical_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
