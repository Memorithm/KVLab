import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "verify_flat_m13b4_trace.py"
VALID = (
    b'{"schema":"flat.m13b4-trace.v1","timing_source":"host_wall_clock",'
    b'"scheduling_variant":"serial_matched","scope":"prefill","events":['
    b'{"kind":"numerical_kv_commit","timestamp_ns":100},'
    b'{"kind":"boolean_signature_commit","timestamp_ns":110},'
    b'{"kind":"decode_visible","timestamp_ns":120}]}'
)


class FlatM13B4TraceCliTests(unittest.TestCase):
    def run_tool(self, payload: bytes) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.json"
            path.write_bytes(payload)
            return subprocess.run(
                [sys.executable, str(TOOL), str(path)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

    def test_valid_trace_reports_only_structural_provenance(self) -> None:
        result = self.run_tool(VALID)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output["schema"], "flat.m13b4-trace.v1")
        self.assertEqual(output["scope"], "prefill")
        self.assertEqual(output["event_count"], 3)
        self.assertEqual(
            output["flat_reference_revision"],
            "29c18275b5687b71599ef11f5badcc4571a52a41",
        )
        self.assertEqual(len(output["trace_sha256"]), 64)
        self.assertNotIn("speedup", output)
        self.assertNotIn("overlap", output)

    def test_noncanonical_trace_fails_closed(self) -> None:
        parsed = json.loads(VALID)
        payload = json.dumps(parsed, indent=2).encode("utf-8")
        result = self.run_tool(payload)
        self.assertEqual(result.returncode, 2)
        self.assertIn("not in canonical FLAT JSON form", result.stderr)


if __name__ == "__main__":
    unittest.main()
