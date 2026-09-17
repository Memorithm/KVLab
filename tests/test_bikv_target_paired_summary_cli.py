import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from kvlab.bikv_target_campaign import BikvTargetCampaignV1
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


class BikvTargetPairedSummaryCliTests(unittest.TestCase):
    def protocol(self):
        return BikvTargetProtocolV1(
            schema=BKV_TARGET_PROTOCOL_SCHEMA_V1,
            campaign_id="paired-cli",
            phase="validation",
            hypothesis_h0="H0",
            hypothesis_h1="H1",
            evidence_bundle_sha256="a" * 64,
            kvlab_commit="b" * 40,
            flat_commit="c" * 40,
            model_id="model",
            model_revision="d" * 40,
            tokenizer_id="tokenizer",
            tokenizer_revision="e" * 40,
            runtime_id="runtime",
            runtime_revision="f" * 40,
            hardware_fingerprint_sha256="1" * 64,
            precision="bf16",
            context_tokens=1024,
            batch_size=1,
            dataset_id="dataset",
            dataset_revision="2" * 40,
            partition_id="validation",
            partition_role="validation",
            tuning_permitted=False,
            seeds=(5,),
            warmup_runs=1,
            repetitions=1,
            boolean_policy="policy-v1",
            baseline_policy=BASELINE_FULL_CACHE_NATIVE_PREFILL,
            timing_source="device_timestamp",
            byte_evidence_kind="logical_packed_payload",
            quality_metric="downstream_quality",
            quality_rule="frozen rule",
            holdout_policy="protected holdout closed",
            required_metrics=REQUIRED_METRICS,
        )

    def target_run(self, protocol, variant):
        metrics = tuple(
            BikvMetricObservationV1(
                name=name,
                status="measured",
                value=(True if name == "reset_reuse_correctness" else 1),
                unit=("boolean" if name == "reset_reuse_correctness" else "count"),
                reason=None,
            )
            for name in REQUIRED_METRICS
        )
        return BikvTargetRunV1(
            schema=BKV_TARGET_RUN_SCHEMA_V1,
            protocol_sha256=protocol.protocol_sha256(),
            campaign_id=protocol.campaign_id,
            attempt_id=f"{variant}-5-0",
            variant=variant,
            seed=5,
            repetition_index=0,
            status="completed",
            failure_reason=None,
            metrics=metrics,
        )

    def write_fixture(self, root):
        protocol = self.protocol()
        runs = tuple(self.target_run(protocol, variant) for variant in ("baseline", "candidate"))
        campaign = BikvTargetCampaignV1.from_runs(protocol=protocol, runs=runs)
        protocol_path = root / "protocol.json"
        campaign_path = root / "campaign.json"
        protocol_path.write_text(protocol.canonical_json(), encoding="utf-8")
        campaign_path.write_bytes(campaign.canonical_json_bytes())
        run_paths = []
        for run in runs:
            path = root / f"{run.variant}.json"
            path.write_text(run.canonical_json(), encoding="utf-8")
            run_paths.append(path)
        return protocol_path, campaign_path, run_paths

    def test_cli_emits_canonical_summary_to_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol, campaign, runs = self.write_fixture(Path(tmp))
            completed = subprocess.run(
                [sys.executable, "tools/summarize_bikv_target_campaign.py", str(protocol), str(campaign), *(str(path) for path in runs)],
                check=False,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            decoded = json.loads(completed.stdout)
            self.assertEqual(decoded["campaign_id"], "paired-cli")
            self.assertIn("no inferential interval", decoded["interpretation"])

    def test_cli_atomically_writes_output_and_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol, campaign, runs = self.write_fixture(root)
            output = root / "evidence" / "paired-summary.json"
            completed = subprocess.run(
                [sys.executable, "tools/summarize_bikv_target_campaign.py", str(protocol), str(campaign), *(str(path) for path in runs), "--output", str(output)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.is_file())
            self.assertFalse(output.with_name(output.name + ".tmp").exists())
            self.assertIn("summary_sha256=", completed.stdout)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["campaign_id"], "paired-cli")

    def test_cli_rejects_run_payload_substitution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol, campaign, runs = self.write_fixture(root)
            candidate = json.loads(runs[1].read_text(encoding="utf-8"))
            candidate["attempt_id"] = "substituted"
            runs[1].write_text(json.dumps(candidate, sort_keys=True, separators=(",", ":")), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "tools/summarize_bikv_target_campaign.py", str(protocol), str(campaign), *(str(path) for path in runs)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("do not match", completed.stderr)


if __name__ == "__main__":
    unittest.main()
