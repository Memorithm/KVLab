import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from kvlab.bikv_evidence_bundle import BikvEvidenceBundleV1
from kvlab.bikv_evidence_receipt import BikvEvidenceReceiptV1


FLAT_REVISION = "0123456789abcdef0123456789abcdef01234567"
KVLAB_REVISION = "89abcdef0123456789abcdef0123456789abcdef"
CLI = Path(__file__).resolve().parents[1] / "tools" / "verify_bikv_evidence_bundle.py"


class BikvEvidenceBundleCliTests(unittest.TestCase):
    def write_case(
        self, directory: Path
    ) -> tuple[Path, dict[str, Path], BikvEvidenceBundleV1]:
        flat_receipt = BikvEvidenceReceiptV1.from_json_bytes(
            producer_repo="Memorithm/FLAT-ATTENTION",
            producer_commit=FLAT_REVISION,
            payload=b'{"candidate_density":0.25}',
        )
        kvlab_receipt = BikvEvidenceReceiptV1.from_json_bytes(
            producer_repo="Memorithm/KVLab",
            producer_commit=KVLAB_REVISION,
            payload=b'{"recall":1.0}',
        )
        bundle = BikvEvidenceBundleV1.from_receipts(
            [
                ("flat.m13b", flat_receipt),
                ("kvlab.k6", kvlab_receipt),
            ]
        )
        bundle_path = directory / "bundle.json"
        flat_path = directory / "flat-receipt.json"
        kvlab_path = directory / "kvlab-receipt.json"
        bundle_path.write_bytes(bundle.canonical_json_bytes())
        flat_path.write_bytes(flat_receipt.canonical_json_bytes())
        kvlab_path.write_bytes(kvlab_receipt.canonical_json_bytes())
        return (
            bundle_path,
            {"flat.m13b": flat_path, "kvlab.k6": kvlab_path},
            bundle,
        )

    def run_cli(
        self, bundle: Path, receipts: dict[str, Path]
    ) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(CLI), "--bundle", str(bundle)]
        for role, path in receipts.items():
            command.extend(["--receipt", f"{role}={path}"])
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_reports_only_verified_bundle_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path, receipt_paths, bundle = self.write_case(Path(tmp))
            completed = self.run_cli(bundle_path, receipt_paths)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        summary = json.loads(completed.stdout)
        self.assertEqual(summary["schema"], bundle.schema)
        self.assertEqual(summary["bundle_sha256"], bundle.bundle_sha256())
        self.assertEqual(summary["entry_count"], 2)
        self.assertEqual(
            [entry["role"] for entry in summary["entries"]],
            ["flat.m13b", "kvlab.k6"],
        )
        self.assertNotIn("candidate_density", completed.stdout)
        self.assertNotIn("recall", completed.stdout)

    def test_missing_or_extra_roles_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path, receipt_paths, _ = self.write_case(Path(tmp))
            missing = {"flat.m13b": receipt_paths["flat.m13b"]}
            missing_result = self.run_cli(bundle_path, missing)
            extra = dict(receipt_paths)
            extra["unexpected"] = receipt_paths["flat.m13b"]
            extra_result = self.run_cli(bundle_path, extra)

        self.assertEqual(missing_result.returncode, 2)
        self.assertIn("receipt roles do not match bundle", missing_result.stderr)
        self.assertEqual(extra_result.returncode, 2)
        self.assertIn("receipt roles do not match bundle", extra_result.stderr)

    def test_tampered_receipt_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path, receipt_paths, _ = self.write_case(Path(tmp))
            replacement = BikvEvidenceReceiptV1.from_json_bytes(
                producer_repo="Memorithm/FLAT-ATTENTION",
                producer_commit=FLAT_REVISION,
                payload=b'{"candidate_density":0.5}',
            )
            receipt_paths["flat.m13b"].write_bytes(replacement.canonical_json_bytes())
            completed = self.run_cli(bundle_path, receipt_paths)

        self.assertEqual(completed.returncode, 2)
        self.assertIn("receipt SHA-256", completed.stderr)
        self.assertEqual(completed.stdout, "")

    def test_noncanonical_bundle_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path, receipt_paths, bundle = self.write_case(Path(tmp))
            pretty = json.dumps(json.loads(bundle.canonical_json_bytes()), indent=2).encode()
            bundle_path.write_bytes(pretty)
            completed = self.run_cli(bundle_path, receipt_paths)

        self.assertEqual(completed.returncode, 2)
        self.assertIn("canonical JSON", completed.stderr)
        self.assertEqual(completed.stdout, "")

    def test_duplicate_receipt_argument_role_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path, receipt_paths, _ = self.write_case(Path(tmp))
            command = [
                sys.executable,
                str(CLI),
                "--bundle",
                str(bundle_path),
                "--receipt",
                f"flat.m13b={receipt_paths['flat.m13b']}",
                "--receipt",
                f"flat.m13b={receipt_paths['flat.m13b']}",
                "--receipt",
                f"kvlab.k6={receipt_paths['kvlab.k6']}",
            ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("duplicate receipt role", completed.stderr)
        self.assertEqual(completed.stdout, "")


if __name__ == "__main__":
    unittest.main()
