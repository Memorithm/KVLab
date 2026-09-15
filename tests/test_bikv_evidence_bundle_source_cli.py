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


class BikvEvidenceBundleSourceCliTests(unittest.TestCase):
    def write_case(
        self, directory: Path
    ) -> tuple[Path, dict[str, Path], dict[str, Path], BikvEvidenceBundleV1]:
        payloads = {
            "flat.m13b": b'{"candidate_density":0.25}',
            "kvlab.k6": b'{"recall":1.0}',
        }
        flat_receipt = BikvEvidenceReceiptV1.from_json_bytes(
            producer_repo="Memorithm/FLAT-ATTENTION",
            producer_commit=FLAT_REVISION,
            payload=payloads["flat.m13b"],
        )
        kvlab_receipt = BikvEvidenceReceiptV1.from_json_bytes(
            producer_repo="Memorithm/KVLab",
            producer_commit=KVLAB_REVISION,
            payload=payloads["kvlab.k6"],
        )
        bundle = BikvEvidenceBundleV1.from_receipts(
            [
                ("flat.m13b", flat_receipt),
                ("kvlab.k6", kvlab_receipt),
            ]
        )

        bundle_path = directory / "bundle.json"
        bundle_path.write_bytes(bundle.canonical_json_bytes())
        receipt_paths = {
            "flat.m13b": directory / "flat-receipt.json",
            "kvlab.k6": directory / "kvlab-receipt.json",
        }
        source_paths = {
            "flat.m13b": directory / "flat-source.json",
            "kvlab.k6": directory / "kvlab-source.json",
        }
        receipt_paths["flat.m13b"].write_bytes(flat_receipt.canonical_json_bytes())
        receipt_paths["kvlab.k6"].write_bytes(kvlab_receipt.canonical_json_bytes())
        for role, payload in payloads.items():
            source_paths[role].write_bytes(payload)
        return bundle_path, receipt_paths, source_paths, bundle

    def run_cli(
        self,
        bundle: Path,
        receipts: dict[str, Path],
        sources: dict[str, Path],
    ) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(CLI), "--bundle", str(bundle)]
        for role, path in receipts.items():
            command.extend(["--receipt", f"{role}={path}"])
        for role, path in sources.items():
            command.extend(["--source", f"{role}={path}"])
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_full_source_set_is_verified_without_exposing_payload_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path, receipts, sources, bundle = self.write_case(Path(tmp))
            completed = self.run_cli(bundle_path, receipts, sources)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        summary = json.loads(completed.stdout)
        self.assertEqual(summary["bundle_sha256"], bundle.bundle_sha256())
        self.assertTrue(summary["source_payloads_verified"])
        self.assertTrue(all(entry["source_verified"] for entry in summary["entries"]))
        self.assertNotIn("candidate_density", completed.stdout)
        self.assertNotIn("recall", completed.stdout)

    def test_partial_source_set_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path, receipts, sources, _ = self.write_case(Path(tmp))
            completed = self.run_cli(
                bundle_path,
                receipts,
                {"flat.m13b": sources["flat.m13b"]},
            )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("source roles do not match bundle", completed.stderr)
        self.assertEqual(completed.stdout, "")

    def test_tampered_source_bytes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path, receipts, sources, _ = self.write_case(Path(tmp))
            # Preserve the original byte length so this exercises the digest
            # mismatch path rather than the earlier length-mismatch guard.
            sources["flat.m13b"].write_bytes(b'{"candidate_density":0.50}')
            completed = self.run_cli(bundle_path, receipts, sources)

        self.assertEqual(completed.returncode, 2)
        self.assertIn("source SHA-256", completed.stderr)
        self.assertEqual(completed.stdout, "")

    def test_duplicate_source_role_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path, receipts, sources, _ = self.write_case(Path(tmp))
            command = [sys.executable, str(CLI), "--bundle", str(bundle_path)]
            for role, path in receipts.items():
                command.extend(["--receipt", f"{role}={path}"])
            command.extend(
                [
                    "--source",
                    f"flat.m13b={sources['flat.m13b']}",
                    "--source",
                    f"flat.m13b={sources['flat.m13b']}",
                    "--source",
                    f"kvlab.k6={sources['kvlab.k6']}",
                ]
            )
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("duplicate source role", completed.stderr)
        self.assertEqual(completed.stdout, "")


if __name__ == "__main__":
    unittest.main()
