import dataclasses
import unittest

from kvlab.bikv_target_campaign import (
    BKV_TARGET_CAMPAIGN_SCHEMA_V1,
    BikvTargetCampaignError,
    BikvTargetCampaignV1,
)
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


class BikvTargetCampaignTests(unittest.TestCase):
    def protocol(self):
        return BikvTargetProtocolV1(
            schema=BKV_TARGET_PROTOCOL_SCHEMA_V1,
            campaign_id="bkv-k6-campaign-test",
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
            context_tokens=4096,
            batch_size=1,
            dataset_id="dataset",
            dataset_revision="2" * 40,
            partition_id="validation",
            partition_role="validation",
            tuning_permitted=False,
            seeds=(7, 11),
            warmup_runs=1,
            repetitions=2,
            boolean_policy="policy-v1",
            baseline_policy=BASELINE_FULL_CACHE_NATIVE_PREFILL,
            timing_source="device_timestamp",
            byte_evidence_kind="logical_packed_payload",
            quality_metric="exact-output-parity",
            quality_rule="must satisfy frozen tolerance",
            holdout_policy="protected holdout remains closed",
            required_metrics=REQUIRED_METRICS,
        )

    def metrics(self, *, failed=False):
        result = []
        for index, name in enumerate(REQUIRED_METRICS):
            if failed and index == 0:
                result.append(
                    BikvMetricObservationV1(
                        name=name,
                        status="failed",
                        value=None,
                        unit=None,
                        reason="fixture measurement failed",
                    )
                )
            else:
                result.append(
                    BikvMetricObservationV1(
                        name=name,
                        status="not_exposed",
                        value=None,
                        unit=None,
                        reason="fixture backend does not expose metric",
                    )
                )
        return tuple(result)

    def target_run(self, *, variant, seed, repetition, failed=False):
        protocol = self.protocol()
        return BikvTargetRunV1(
            schema=BKV_TARGET_RUN_SCHEMA_V1,
            protocol_sha256=protocol.protocol_sha256(),
            campaign_id=protocol.campaign_id,
            attempt_id=f"{variant}-seed{seed}-r{repetition}",
            variant=variant,
            seed=seed,
            repetition_index=repetition,
            status="failed" if failed else "completed",
            failure_reason="fixture runtime failed" if failed else None,
            metrics=self.metrics(failed=failed),
        )

    def complete_runs(self):
        return tuple(
            self.target_run(variant=variant, seed=seed, repetition=repetition)
            for seed in self.protocol().seeds
            for repetition in range(self.protocol().repetitions)
            for variant in ("candidate", "baseline")
        )

    def test_complete_campaign_is_canonical_and_replayable(self):
        protocol = self.protocol()
        campaign = BikvTargetCampaignV1.from_runs(
            protocol=protocol, runs=reversed(self.complete_runs())
        )
        self.assertEqual(campaign.schema, BKV_TARGET_CAMPAIGN_SCHEMA_V1)
        self.assertEqual(len(campaign.runs), 8)
        self.assertEqual(campaign.runs[0].variant, "baseline")
        self.assertEqual(campaign.runs[1].variant, "candidate")
        payload = campaign.canonical_json_bytes()
        decoded = BikvTargetCampaignV1.from_canonical_json_bytes(payload)
        self.assertEqual(decoded, campaign)
        decoded.validate_against(protocol)
        decoded.verify_runs(protocol=protocol, runs=self.complete_runs())
        self.assertEqual(len(decoded.campaign_sha256()), 64)

    def test_missing_candidate_attempt_fails_closed(self):
        runs = list(self.complete_runs())
        runs.pop()
        with self.assertRaisesRegex(BikvTargetCampaignError, "missing=1"):
            BikvTargetCampaignV1.from_runs(protocol=self.protocol(), runs=runs)

    def test_duplicate_logical_slot_fails_closed(self):
        runs = list(self.complete_runs())
        runs.append(dataclasses.replace(runs[0], attempt_id="duplicate-attempt"))
        with self.assertRaisesRegex(BikvTargetCampaignError, "canonical order|duplicate"):
            BikvTargetCampaignV1.from_runs(protocol=self.protocol(), runs=runs)

    def test_failed_attempt_is_retained_and_still_satisfies_slot_completeness(self):
        runs = list(self.complete_runs())
        runs[0] = self.target_run(variant="candidate", seed=7, repetition=0, failed=True)
        campaign = BikvTargetCampaignV1.from_runs(protocol=self.protocol(), runs=runs)
        failed = [run for run in campaign.runs if run.status == "failed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].attempt_id, "candidate-seed7-r0")

    def test_same_run_cannot_fill_multiple_slots(self):
        campaign = BikvTargetCampaignV1.from_runs(
            protocol=self.protocol(), runs=self.complete_runs()
        )
        changed = list(campaign.runs)
        changed[1] = dataclasses.replace(
            changed[1], run_sha256=changed[0].run_sha256
        )
        malformed = dataclasses.replace(campaign, runs=tuple(changed))
        with self.assertRaisesRegex(BikvTargetCampaignError, "one retained run"):
            malformed.validate()

    def test_verify_runs_detects_payload_substitution(self):
        protocol = self.protocol()
        runs = list(self.complete_runs())
        campaign = BikvTargetCampaignV1.from_runs(protocol=protocol, runs=runs)
        runs[0] = dataclasses.replace(
            runs[0], attempt_id="baseline-seed7-r0-replayed"
        )
        with self.assertRaisesRegex(BikvTargetCampaignError, "do not match"):
            campaign.verify_runs(protocol=protocol, runs=runs)

    def test_noncanonical_json_is_rejected(self):
        campaign = BikvTargetCampaignV1.from_runs(
            protocol=self.protocol(), runs=self.complete_runs()
        )
        with self.assertRaisesRegex(BikvTargetCampaignError, "not canonical"):
            BikvTargetCampaignV1.from_canonical_json_bytes(
                campaign.canonical_json_bytes() + b"\n"
            )

    def test_duplicate_json_key_is_rejected_before_canonicalization(self):
        campaign = BikvTargetCampaignV1.from_runs(
            protocol=self.protocol(), runs=self.complete_runs()
        )
        payload = campaign.canonical_json_bytes()
        duplicate = payload.replace(
            b'{"campaign_id":',
            b'{"campaign_id":"shadow","campaign_id":',
            1,
        )
        with self.assertRaisesRegex(BikvTargetCampaignError, "duplicate JSON key"):
            BikvTargetCampaignV1.from_canonical_json_bytes(duplicate)


if __name__ == "__main__":
    unittest.main()
