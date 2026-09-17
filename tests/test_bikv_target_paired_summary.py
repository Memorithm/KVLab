import dataclasses
import json
import unittest

from kvlab.bikv_target_campaign import BikvTargetCampaignV1
from kvlab.bikv_target_paired_summary import (
    BKV_TARGET_PAIRED_SUMMARY_SCHEMA_V1,
    BikvTargetPairedSummaryError,
    build_paired_summary,
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


class BikvTargetPairedSummaryTests(unittest.TestCase):
    def protocol(self):
        return BikvTargetProtocolV1(
            schema=BKV_TARGET_PROTOCOL_SCHEMA_V1,
            campaign_id="paired-summary-test",
            phase="validation",
            hypothesis_h0="candidate does not improve declared target metrics",
            hypothesis_h1="candidate improves declared target metrics",
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
            quality_metric="downstream_quality",
            quality_rule="frozen externally; this summary does not decide it",
            holdout_policy="protected holdout remains closed",
            required_metrics=REQUIRED_METRICS,
        )

    def metrics(self, *, variant, seed, repetition, incomplete=None, unit_override=None):
        result = []
        slot = seed + repetition
        for index, name in enumerate(REQUIRED_METRICS):
            if name == incomplete:
                result.append(
                    BikvMetricObservationV1(
                        name=name,
                        status="not_exposed",
                        value=None,
                        unit=None,
                        reason="fixture backend does not expose metric",
                    )
                )
                continue
            if name == "reset_reuse_correctness":
                value = variant == "baseline" or slot % 2 == 0
                unit = "boolean"
            elif name == "downstream_quality":
                value = 0.8 + slot / 100.0 + (0.01 if variant == "candidate" else 0.0)
                unit = "score"
            else:
                value = index * 100 + slot + (5 if variant == "candidate" else 0)
                unit = "ns" if name.endswith("_ns") else "count"
            if name == unit_override and variant == "candidate":
                unit = "different-unit"
            result.append(
                BikvMetricObservationV1(
                    name=name,
                    status="measured",
                    value=value,
                    unit=unit,
                    reason=None,
                )
            )
        return tuple(result)

    def target_run(self, *, variant, seed, repetition, incomplete=None, unit_override=None):
        protocol = self.protocol()
        return BikvTargetRunV1(
            schema=BKV_TARGET_RUN_SCHEMA_V1,
            protocol_sha256=protocol.protocol_sha256(),
            campaign_id=protocol.campaign_id,
            attempt_id=f"{variant}-{seed}-{repetition}",
            variant=variant,
            seed=seed,
            repetition_index=repetition,
            status="completed",
            failure_reason=None,
            metrics=self.metrics(
                variant=variant,
                seed=seed,
                repetition=repetition,
                incomplete=incomplete,
                unit_override=unit_override,
            ),
        )

    def runs(self, *, incomplete=None, unit_override=None):
        protocol = self.protocol()
        return tuple(
            self.target_run(
                variant=variant,
                seed=seed,
                repetition=repetition,
                incomplete=incomplete if variant == "candidate" and seed == 7 and repetition == 0 else None,
                unit_override=unit_override,
            )
            for seed in protocol.seeds
            for repetition in range(protocol.repetitions)
            for variant in ("baseline", "candidate")
        )

    def summarize(self, runs):
        protocol = self.protocol()
        campaign = BikvTargetCampaignV1.from_runs(protocol=protocol, runs=runs)
        return build_paired_summary(protocol=protocol, campaign=campaign, runs=runs)

    def test_complete_numeric_pairs_retain_exact_deltas(self):
        summary = self.summarize(self.runs())
        self.assertEqual(summary.schema, BKV_TARGET_PAIRED_SUMMARY_SCHEMA_V1)
        metric = next(item for item in summary.metrics if item.name == "first_token_latency_ns")
        self.assertEqual(metric.total_pairs, 4)
        self.assertEqual(metric.measured_pairs, 4)
        self.assertEqual(metric.incomplete_pairs, 0)
        self.assertEqual(metric.median_delta_candidate_minus_baseline, 5)
        self.assertEqual(metric.min_delta_candidate_minus_baseline, 5)
        self.assertEqual(metric.max_delta_candidate_minus_baseline, 5)
        self.assertEqual(len(summary.summary_sha256()), 64)

    def test_unavailable_candidate_metric_is_retained_not_dropped(self):
        summary = self.summarize(self.runs(incomplete="candidate_recall"))
        metric = next(item for item in summary.metrics if item.name == "candidate_recall")
        self.assertEqual(metric.measured_pairs, 3)
        self.assertEqual(metric.incomplete_pairs, 1)
        point = metric.points[0]
        self.assertEqual(point.candidate_metric_status, "not_exposed")
        self.assertIn("does not expose", point.candidate_reason)

    def test_boolean_metric_is_not_coerced_to_numeric_delta(self):
        summary = self.summarize(self.runs())
        metric = next(item for item in summary.metrics if item.name == "reset_reuse_correctness")
        self.assertIsNone(metric.median_delta_candidate_minus_baseline)
        self.assertEqual(metric.boolean_equal_pairs, 2)
        self.assertTrue(all(point.delta_candidate_minus_baseline is None for point in metric.points))

    def test_measured_unit_drift_fails_closed(self):
        with self.assertRaisesRegex(BikvTargetPairedSummaryError, "unit mismatch"):
            self.summarize(self.runs(unit_override="first_token_latency_ns"))

    def test_campaign_substitution_fails_before_summary(self):
        runs = list(self.runs())
        protocol = self.protocol()
        campaign = BikvTargetCampaignV1.from_runs(protocol=protocol, runs=runs)
        runs[0] = dataclasses.replace(runs[0], attempt_id="substituted")
        with self.assertRaisesRegex(Exception, "do not match"):
            build_paired_summary(protocol=protocol, campaign=campaign, runs=runs)

    def test_canonical_payload_keeps_non_inferential_boundary(self):
        summary = self.summarize(self.runs())
        decoded = json.loads(summary.canonical_json_bytes())
        self.assertEqual(decoded["campaign_sha256"], summary.campaign_sha256)
        self.assertIn("no inferential interval", decoded["interpretation"])
        self.assertNotIn("p_value", decoded)
        self.assertNotIn("promote", decoded)


if __name__ == "__main__":
    unittest.main()
