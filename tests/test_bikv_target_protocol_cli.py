import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from kvlab.bikv_target_protocol import (
    BASELINE_FULL_CACHE_NATIVE_PREFILL,
    BKV_TARGET_PROTOCOL_SCHEMA_V1,
    REQUIRED_METRICS,
    BikvTargetProtocolV1,
)


class BikvTargetProtocolCliTests(unittest.TestCase):
    def protocol(self):
        return BikvTargetProtocolV1(
            schema=BKV_TARGET_PROTOCOL_SCHEMA_V1,
            campaign_id="bkv-k6-cli",
            phase="development",
            hypothesis_h0="H0",
            hypothesis_h1="H1",
            evidence_bundle_sha256="a" * 64,
            kvlab_commit="b" * 40,
            flat_commit="c" * 40,
            model_id="m",
            model_revision="d" * 40,
            tokenizer_id="t",
            tokenizer_revision="e" * 40,
            runtime_id="r",
            runtime_revision="f" * 40,
            hardware_fingerprint_sha256="1" * 64,
            precision="f32",
            context_tokens=128,
            batch_size=1,
            dataset_id="d",
            dataset_revision="2" * 40,
            partition_id="dev",
            partition_role="development",
            tuning_permitted=True,
            seeds=(1,),
            warmup_runs=0,
            repetitions=1,
            boolean_policy="policy",
            baseline_policy=BASELINE_FULL_CACHE_NATIVE_PREFILL,
            timing_source="host_wall_clock",
            byte_evidence_kind="logical_packed_payload",
            quality_metric="parity",
            quality_rule="exact",
            holdout_policy="holdout closed",
            required_metrics=REQUIRED_METRICS,
        )

    def test_cli_verifies_canonical_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.json"
            protocol = self.protocol()
            path.write_text(protocol.canonical_json(), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "tools/verify_bikv_target_protocol.py", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = json.loads(completed.stdout)
            self.assertEqual(output["protocol_sha256"], protocol.protocol_sha256())

    def test_cli_rejects_noncanonical_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.json"
            path.write_text(self.protocol().canonical_json() + "\n", encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "tools/verify_bikv_target_protocol.py", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("not canonical", completed.stderr)


if __name__ == "__main__":
    unittest.main()
