from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from kvlab.evidence_directory import (
    OWNERSHIP_MARKER,
    prepare_evidence_directory,
)


class EvidenceDirectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repo_root = Path(self.temporary_directory.name) / "repo"
        self.repo_root.mkdir()

    def test_creates_default_directory_with_ownership_marker(self) -> None:
        evidence_dir = prepare_evidence_directory(
            repo_root=self.repo_root,
            requested="target/eww-evidence",
        )

        self.assertEqual(evidence_dir, self.repo_root / "target" / "eww-evidence")
        self.assertEqual(
            (evidence_dir / OWNERSHIP_MARKER).read_text(encoding="utf-8"),
            "schema=kvlab.eww-evidence-directory/v1\n",
        )

    def test_accepts_nested_custom_directory_below_target(self) -> None:
        evidence_dir = prepare_evidence_directory(
            repo_root=self.repo_root,
            requested="target/campaigns/run-001",
        )

        self.assertTrue(evidence_dir.is_dir())

    def test_rejects_absolute_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "repository-relative"):
            prepare_evidence_directory(
                repo_root=self.repo_root,
                requested=str(self.repo_root / "target" / "absolute"),
            )

    def test_rejects_parent_traversal(self) -> None:
        with self.assertRaisesRegex(ValueError, "stay below"):
            prepare_evidence_directory(
                repo_root=self.repo_root,
                requested="target/../../outside",
            )

    def test_rejects_target_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "target/ root"):
            prepare_evidence_directory(
                repo_root=self.repo_root,
                requested="target",
            )

    def test_preserves_existing_evidence(self) -> None:
        evidence_dir = self.repo_root / "target" / "eww-evidence"
        evidence_dir.mkdir(parents=True)
        sentinel = evidence_dir / "sentinel.txt"
        sentinel.write_text("preserve me\n", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            prepare_evidence_directory(
                repo_root=self.repo_root,
                requested="target/eww-evidence",
            )

        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve me\n")

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_rejects_symlink_escape(self) -> None:
        outside = Path(self.temporary_directory.name) / "outside"
        outside.mkdir()
        target = self.repo_root / "target"
        target.mkdir()
        (target / "escape").symlink_to(outside, target_is_directory=True)

        with self.assertRaisesRegex(ValueError, "cannot contain symlinks"):
            prepare_evidence_directory(
                repo_root=self.repo_root,
                requested="target/escape/evidence",
            )

        self.assertFalse((outside / "evidence").exists())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_rejects_broken_symlink_target(self) -> None:
        target = self.repo_root / "target"
        target.mkdir()
        requested = target / "eww-evidence"
        requested.symlink_to(target / "missing")

        with self.assertRaisesRegex(ValueError, "cannot contain symlinks"):
            prepare_evidence_directory(
                repo_root=self.repo_root,
                requested="target/eww-evidence",
            )


if __name__ == "__main__":
    unittest.main()
