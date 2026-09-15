import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from kvlab.bikv_evidence_receipt import BikvEvidenceReceiptV1


FLAT_REVISION = "0123456789abcdef0123456789abcdef01234567"
CLI = Path(__file__).resolve().parents[1] / "tools" / "verify_bikv_evidence_receipt.py"


class BikvEvidenceReceiptCliTests(unittest.TestCase):
    def write_case(self, directory: Path, source: bytes) -> tuple[Path, Path, BikvEvidenceReceiptV1]:
        receipt = BikvEvidenceReceiptV1.from_json_bytes(
            producer_repo="Memorithm/FLAT-ATTENTION",
            producer_commit=FLAT_REVISION,
            payload=source,
        )
        source_path = directory / "source.json"
        receipt_path = directory / "receipt.json"
        source_path.write_bytes(source)
        receipt_path.write_bytes(receipt.canonical_json_bytes())
        return receipt_path, source_path, receipt

    def run_cli(self, receipt: Path, source: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(CLI),
                "--receipt",
                str(receipt),
                "--source",
                str(source),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_reports_only_verified_provenance_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            receipt_path, source_path, receipt = self.write_case(
                Path(tmp), b'{"metrics":{"candidate_density":0.25}}'
            )
            completed = self.run_cli(receipt_path, source_path)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        summary = json.loads(completed.stdout)
        self.assertEqual(summary["schema"], receipt.schema)
        self.assertEqual(summary["producer_repo"], receipt.producer_repo)
        self.assertEqual(summary["producer_commit"], receipt.producer_commit)
        self.assertEqual(summary["source_sha256"], receipt.source_sha256)
        self.assertEqual(summary["source_bytes"], receipt.source_bytes)
        self.assertEqual(summary["receipt_sha256"], receipt.receipt_sha256())
        self.assertNotIn("metrics", summary)
        self.assertNotIn("candidate_density", summary)

    def test_tampered_source_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            receipt_path, source_path, _ = self.write_case(Path(tmp), b'{"a":1,"b":2}')
            source_path.write_bytes(b'{"b":2,"a":1}')
            completed = self.run_cli(receipt_path, source_path)

        self.assertEqual(completed.returncode, 2)
        self.assertIn("SHA-256", completed.stderr)
        self.assertEqual(completed.stdout, "")

    def test_noncanonical_receipt_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            receipt_path, source_path, receipt = self.write_case(Path(tmp), b'{"a":1}')
            pretty = json.dumps(json.loads(receipt.canonical_json_bytes()), indent=2).encode()
            receipt_path.write_bytes(pretty)
            completed = self.run_cli(receipt_path, source_path)

        self.assertEqual(completed.returncode, 2)
        self.assertIn("canonical JSON", completed.stderr)
        self.assertEqual(completed.stdout, "")


if __name__ == "__main__":
    unittest.main()
