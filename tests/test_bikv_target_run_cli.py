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
from kvlab.bikv_target_run import (
    BKV_TARGET_RUN_SCHEMA_V1,
    BikvMetricObservationV1,
    BikvTargetRunV1,
)


class BikvTargetRunCliTests(unittest.TestCase):
    def protocol(self):
        return BikvTargetProtocolV1(
            schema=BKV_TARGET_PROTOCOL_SCHEMA_V1,
            campaign_id="bkv-k6-cli-run",
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

    def target_run(self, protocol):
        metrics = tuple(
            BikvMetricObservationV1(
                name=name,
                status="not_exposed",
                value=None,
                unit=None,
                reason="test backend does not expose metric",
            )
            for name in REQUIRED_METRICS
        )
        return BikvTargetRunV1(
            schema=BKV_TARGET_RUN_SCHEMA_V1,
            protocol_sha256=protocol.protocol_sha256(),
            campaign_id=protocol.campaign_id,
            attempt_id="baseline-1-0",
            variant="baseline",
            seed=1,
            repetition_index=0,
            status="completed",
            failure_reason=None,
            metrics=metrics,
        )

    def test_cli_binds_canonical_run_to_protocol(self):
        protocol = self.protocol()
        run = self.target_run(protocol)
        with tempfile.TemporaryDirectory() as tmp:
            protocol_path = Path(tmp) / "protocol.json"
            run_path = Path(tmp) / "run.json"
            protocol_path.write_text(protocol.canonical_json(), encoding="utf-8")
            run_path.write_text(run.canonical_json(), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "tools/verify_bikv_target_run.py",
                    str(protocol_path),
                    str(run_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = json.loads(completed.stdout)
            self.assertEqual(output["run_sha256"], run.run_sha256())
            self.assertEqual(output["metric_status_counts"]["not_exposed"], len(REQUIRED_METRICS))

    def test_cli_rejects_run_bound_to_other_protocol(self):
        protocol = self.protocol()
        run = self.target_run(protocol)
        with tempfile.TemporaryDirectory() as tmp:
            protocol_path = Path(tmp) / "protocol.json"
            run_path = Path(tmp) / "run.json"
            foreign = self.protocol()
            object.__setattr__(foreign, "campaign_id", "other-campaign")
            protocol_path.write_text(foreign.canonical_json(), encoding="utf-8")
            run_path.write_text(run.canonical_json(), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "tools/verify_bikv_target_run.py",
                    str(protocol_path),
                    str(run_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("protocol_sha256", completed.stderr)


if __name__ == "__main__":
    unittest.main()
