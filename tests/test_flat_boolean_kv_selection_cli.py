import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "verify_flat_boolean_kv_selection.py"
VALID = (
    b'{"schema":"flat.boolean-kv-selection.v1","generation":0,"signature_bits":8,'
    b'"live_tokens":10,"mapped_pages":3,"boolean_pages_scanned":3,'
    b'"boolean_key_bytes_read":24,"full_numerical_kv_bytes":640,'
    b'"selected_numerical_kv_bytes":384,"avoided_numerical_kv_bytes":256,'
    b'"selected_pages":[{"logical_page":1,"physical_page":1,"live_tokens":4,'
    b'"hamming_distance":0,"xnor_matches":8},{"logical_page":2,"physical_page":2,'
    b'"live_tokens":2,"hamming_distance":1,"xnor_matches":7}],'
    b'"evidence_checksum":{"algorithm":"fnv1a64","value":"fd932815b27ff883"}}'
)


class FlatBooleanKvSelectionCliTests(unittest.TestCase):
    def run_tool(self, payload: bytes) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "selection.json"
            path.write_bytes(payload)
            return subprocess.run(
                [sys.executable, str(TOOL), str(path)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

    def test_valid_selection_reports_only_structural_provenance(self) -> None:
        result = self.run_tool(VALID)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output["schema"], "flat.boolean-kv-selection.v1")
        self.assertEqual(output["selected_page_count"], 2)
        self.assertEqual(output["mapped_pages"], 3)
        self.assertEqual(output["boolean_key_bytes_read"], 24)
        self.assertEqual(
            output["flat_reference_revision"],
            "c1249bad4fd3f41c93a35f8a6204c98de9a3b687",
        )
        self.assertEqual(len(output["selection_sha256"]), 64)
        self.assertNotIn("speedup", output)
        self.assertNotIn("dram", output)
        self.assertNotIn("latency", output)

    def test_noncanonical_selection_fails_closed(self) -> None:
        parsed = json.loads(VALID)
        payload = json.dumps(parsed, indent=2).encode("utf-8")
        result = self.run_tool(payload)
        self.assertEqual(result.returncode, 2)
        self.assertIn("not in canonical FLAT JSON form", result.stderr)


if __name__ == "__main__":
    unittest.main()
