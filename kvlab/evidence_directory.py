"""Fail-closed preparation of repository-local evidence directories.

Evidence campaign scripts must never recursively remove a path supplied by an
environment variable.  This module confines requested directories below the
repository's ``target`` directory, refuses pre-existing paths, and writes a
small ownership marker into each newly created directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


OWNERSHIP_MARKER = ".kvlab-eww-evidence-owner"
_MARKER_CONTENT = "schema=kvlab.eww-evidence-directory/v1\n"


def prepare_evidence_directory(*, repo_root: Path, requested: str) -> Path:
    """Create one new EWW evidence directory confined below ``target``.

    ``requested`` is deliberately required to be repository-relative.  Both
    traversal outside ``target`` and symlink escapes are rejected after
    canonicalization.  Existing paths are preserved rather than overwritten.
    """

    if not isinstance(requested, str) or not requested:
        raise ValueError("evidence directory must be a non-empty path")
    if any(ord(character) < 32 or ord(character) == 127 for character in requested):
        raise ValueError("evidence directory must not contain control characters")

    requested_path = Path(requested)
    if requested_path.is_absolute():
        raise ValueError("evidence directory must be repository-relative")

    root = Path(repo_root).resolve(strict=True)
    allowed_parent = root / "target"
    if allowed_parent.is_symlink():
        raise ValueError("repository target/ directory cannot be a symlink")
    allowed_parent.mkdir(parents=True, exist_ok=True)
    allowed_parent = allowed_parent.resolve(strict=True)

    raw_candidate = root / requested_path
    current = root
    for component in requested_path.parts:
        current /= component
        if current.is_symlink():
            raise ValueError("evidence directory path cannot contain symlinks")

    candidate = raw_candidate.resolve(strict=False)
    try:
        relative = candidate.relative_to(allowed_parent)
    except ValueError as exc:
        raise ValueError(
            "evidence directory must stay below repository target/"
        ) from exc
    if relative == Path("."):
        raise ValueError("evidence directory cannot be the target/ root")

    if os.path.lexists(raw_candidate):
        raise FileExistsError(
            f"refusing to overwrite existing evidence path: {raw_candidate}"
        )

    # lexists also detects a broken symlink, which Path.exists intentionally
    # does not.  No existing path is removed: prior evidence is preserved.
    if os.path.lexists(candidate):
        raise FileExistsError(
            f"refusing to overwrite existing evidence path: {candidate}"
        )

    candidate.mkdir(parents=True, exist_ok=False)
    marker = candidate / OWNERSHIP_MARKER
    with marker.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_MARKER_CONTENT)
    return candidate


def main(argv: list[str] | None = None) -> int:
    """Prepare a confined directory and print its canonical absolute path."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--requested", required=True)
    args = parser.parse_args(argv)

    try:
        evidence_dir = prepare_evidence_directory(
            repo_root=args.repo_root,
            requested=args.requested,
        )
    except (FileExistsError, OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(evidence_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
