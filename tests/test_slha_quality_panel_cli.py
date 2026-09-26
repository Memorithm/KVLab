import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_slha_quality_panel.py"


class SlhaQualityPanelCliTests(unittest.TestCase):
    def run_payload(self, payload: dict[str, object]) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(RUNNER), str(path)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                env={**os.environ, "PYTHONPATH": str(ROOT)},
                text=True,
            )

    def payload(self) -> dict[str, object]:
        return {
            "experiment_id": "skv0-cli-test",
            "query_id": "q0",
            "candidate_id": "candidate-a",
            "top_k": 1,
            "reference_scores": [2.0, 1.0],
            "selected_ids": [0],
            "candidate_scores": [2.0, 1.0],
        }

    def test_numeric_payload_emits_canonical_evidence(self) -> None:
        result = self.run_payload(self.payload())
        self.assertEqual(result.returncode, 0, result.stderr)
        decoded = json.loads(result.stdout)
        self.assertEqual(decoded["reference_scores"], [2.0, 1.0])
        self.assertEqual(decoded["candidate_scores"], [2.0, 1.0])

    def test_original_json_score_types_are_not_coerced(self) -> None:
        for field, invalid in (
            ("reference_scores", ["2.0", 1.0]),
            ("candidate_scores", [True, 1.0]),
        ):
            with self.subTest(field=field):
                payload = self.payload()
                payload[field] = invalid
                result = self.run_payload(payload)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("values must be numeric", result.stderr)


if __name__ == "__main__":
    unittest.main()
