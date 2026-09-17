import dataclasses
import json
import unittest

from kvlab.bikv_target_analysis_plan import (
    BKV_TARGET_ANALYSIS_PLAN_SCHEMA_V1,
    BKV_TARGET_PAIRED_SUMMARY_SCHEMA_V1,
    INCOMPLETE_PAIR_POLICY,
    PROMOTION_RULE,
    BikvTargetAnalysisPlanV1,
)
from kvlab.bikv_target_campaign import BikvTargetCampaignV1
from kvlab.bikv_target_decision import (
    BKV_TARGET_DECISION_PLAN_SCHEMA_V2,
    BOOTSTRAP_QUANTILE_METHOD,
    DECISION_ORIENTATION,
    SCIRUST_STATS_OPERATION,
    SCIRUST_STATS_SCHEMA,
    SCIRUST_STATS_SOURCE_COMMIT,
    BikvCandidateMetricGuardV1,
    BikvTargetDecisionError,
    BikvTargetDecisionPlanV2,
    _bootstrap_interval,
    evaluate_target_campaign,
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


class BikvTargetDecisionTests(unittest.TestCase):
    def protocol(self, *, seeds=(7, 11), repetitions=2):
        return BikvTargetProtocolV1(
            schema=BKV_TARGET_PROTOCOL_SCHEMA_V1,
            campaign_id="decision-test",
            phase="validation",
            hypothesis_h0="candidate does not meet the frozen BIKV decision gate",
            hypothesis_h1="candidate meets the frozen BIKV decision gate",
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
            seeds=seeds,
            warmup_runs=1,
            repetitions=repetitions,
            boolean_policy="boolean-policy-v1",
            baseline_policy=BASELINE_FULL_CACHE_NATIVE_PREFILL,
            timing_source="device_timestamp",
            byte_evidence_kind="logical_packed_payload",
            quality_metric="downstream_quality",
            quality_rule="candidate quality uses the separately frozen noninferiority margin",
            holdout_policy="protected holdout remains closed",
            required_metrics=REQUIRED_METRICS,
        )

    def base_plan(self, protocol, *, uncertainty="paired_percentile_bootstrap"):
        bootstrap = uncertainty == "paired_percentile_bootstrap"
        return BikvTargetAnalysisPlanV1(
            schema=BKV_TARGET_ANALYSIS_PLAN_SCHEMA_V1,
            protocol_sha256=protocol.protocol_sha256(),
            campaign_id=protocol.campaign_id,
            paired_summary_schema=BKV_TARGET_PAIRED_SUMMARY_SCHEMA_V1,
            hypothesis_h0=protocol.hypothesis_h0,
            hypothesis_h1=protocol.hypothesis_h1,
            primary_performance_metric="first_token_latency_ns",
            primary_direction="lower_is_better",
            primary_minimum_effect=10,
            primary_unit="ns",
            quality_metric=protocol.quality_metric,
            quality_direction="higher_is_better",
            quality_noninferiority_margin=0.05,
            quality_unit="score",
            uncertainty_method=uncertainty,
            confidence_level_ppm=950_000,
            bootstrap_resamples=1000 if bootstrap else None,
            bootstrap_seed=20260917 if bootstrap else None,
            required_complete_pairs=len(protocol.seeds) * protocol.repetitions,
            incomplete_pair_policy=INCOMPLETE_PAIR_POLICY,
            multiplicity_policy="single primary metric; quality plus correctness guards are gates",
            holdout_policy=protocol.holdout_policy,
            created_before_outcome_inspection=True,
            tuning_permitted=False,
            promotion_rule=PROMOTION_RULE,
        )

    def guards(self):
        return (
            BikvCandidateMetricGuardV1("candidate_recall", "ge", 0.90, "ratio"),
            BikvCandidateMetricGuardV1("candidate_false_negative_rate", "le", 0.10, "ratio"),
            BikvCandidateMetricGuardV1("o_error", "le", 0.01, "abs_error"),
            BikvCandidateMetricGuardV1("lse_error", "le", 0.01, "abs_error"),
            BikvCandidateMetricGuardV1("reset_reuse_correctness", "eq_true", None, "boolean"),
        )

    def decision_plan(self, protocol, *, uncertainty="paired_percentile_bootstrap"):
        return BikvTargetDecisionPlanV2(
            schema=BKV_TARGET_DECISION_PLAN_SCHEMA_V2,
            base_plan=self.base_plan(protocol, uncertainty=uncertainty),
            primary_statistic="paired_mean",
            quality_statistic="paired_mean",
            bootstrap_quantile_method=BOOTSTRAP_QUANTILE_METHOD,
            statistics_schema=SCIRUST_STATS_SCHEMA,
            statistics_source_commit=SCIRUST_STATS_SOURCE_COMMIT,
            statistics_operation=SCIRUST_STATS_OPERATION,
            decision_orientation=DECISION_ORIENTATION,
            candidate_metric_guards=self.guards(),
        )

    def metrics(self, *, variant, seed, repetition, primary_shift=-20, quality_shift=-0.02,
                incomplete=None, guard_failure=None, unit_mismatch=None):
        values = []
        jitter = (seed % 3) + repetition
        for name in REQUIRED_METRICS:
            if name == incomplete:
                values.append(BikvMetricObservationV1(name, "not_exposed", None, None, "not exposed by fixture"))
                continue
            if name == "reset_reuse_correctness":
                value, unit = (guard_failure != name), "boolean"
            elif name == "candidate_recall":
                value, unit = (0.80 if guard_failure == name else 0.95), "ratio"
            elif name == "candidate_false_negative_rate":
                value, unit = (0.20 if guard_failure == name else 0.05), "ratio"
            elif name in {"o_error", "lse_error"}:
                value, unit = (0.02 if guard_failure == name else 0.001), "abs_error"
            elif name == "downstream_quality":
                value = 1.0 + (quality_shift if variant == "candidate" else 0.0) + jitter * 0.001
                unit = "score"
            elif name == "boolean_frontend_ns":
                value, unit = (20 + jitter), "ns"
            elif name == "first_token_latency_ns":
                value = 100 + jitter + (primary_shift if variant == "candidate" else 0)
                unit = "ns"
            elif name == "numerical_kv_bytes_avoided":
                value, unit = (4096 if variant == "candidate" else 0), "bytes"
            elif name == "boolean_kv_bytes_read":
                value, unit = 128, "bytes"
            elif name.endswith("_ns"):
                value, unit = 10 + jitter, "ns"
            elif name in {"tokens_per_second", "pages_per_second", "bits_compared_per_second",
                          "effective_bandwidth_bytes_per_second", "scaling_efficiency"}:
                value, unit = 1.0 + jitter * 0.01, "rate"
            else:
                value, unit = 1, "count"
            if unit_mismatch == name and variant == "candidate":
                unit = "wrong-unit"
            values.append(BikvMetricObservationV1(name, "measured", value, unit, None))
        return tuple(values)

    def runs(self, protocol, **metric_options):
        return tuple(
            BikvTargetRunV1(
                schema=BKV_TARGET_RUN_SCHEMA_V1,
                protocol_sha256=protocol.protocol_sha256(),
                campaign_id=protocol.campaign_id,
                attempt_id=f"{variant}-{seed}-{rep}",
                variant=variant,
                seed=seed,
                repetition_index=rep,
                status="completed",
                failure_reason=None,
                metrics=self.metrics(variant=variant, seed=seed, repetition=rep, **metric_options),
            )
            for seed in protocol.seeds
            for rep in range(protocol.repetitions)
            for variant in ("baseline", "candidate")
        )

    def evaluate(self, protocol, runs, plan=None):
        campaign = BikvTargetCampaignV1.from_runs(protocol=protocol, runs=runs)
        return evaluate_target_campaign(
            protocol=protocol,
            campaign=campaign,
            runs=runs,
            decision_plan=plan or self.decision_plan(protocol),
        )

    def test_bootstrap_gate_can_pass_without_authorizing_bkv_k9(self):
        protocol = self.protocol()
        result = self.evaluate(protocol, self.runs(protocol))
        self.assertEqual(result.disposition, "candidate_meets_preregistered_gate")
        self.assertTrue(result.primary.passed)
        self.assertTrue(result.quality.passed)
        self.assertTrue(all(item.passed for item in result.metric_guards))
        self.assertIn("does not authorize BKV-K9", result.interpretation)
        self.assertEqual(len(result.decision_sha256()), 64)

    def test_primary_negative_result_is_retained_as_gate_failure(self):
        protocol = self.protocol()
        result = self.evaluate(protocol, self.runs(protocol, primary_shift=-5))
        self.assertEqual(result.disposition, "candidate_does_not_meet_preregistered_gate")
        self.assertFalse(result.primary.passed)
        self.assertIsNone(result.primary.blocker)

    def test_guard_failure_is_negative_not_missing_evidence(self):
        protocol = self.protocol()
        result = self.evaluate(protocol, self.runs(protocol, guard_failure="candidate_recall"))
        recall = next(item for item in result.metric_guards if item.metric == "candidate_recall")
        self.assertFalse(recall.passed)
        self.assertIsNone(recall.blocker)
        self.assertEqual(result.disposition, "candidate_does_not_meet_preregistered_gate")

    def test_incomplete_primary_evidence_blocks_decision(self):
        protocol = self.protocol()
        result = self.evaluate(protocol, self.runs(protocol, incomplete="first_token_latency_ns"))
        self.assertEqual(result.disposition, "blocked_incomplete_evidence")
        self.assertEqual(result.primary.blocker, "incomplete paired evidence")

    def test_unit_substitution_fails_closed(self):
        protocol = self.protocol()
        with self.assertRaisesRegex(Exception, "unit mismatch"):
            self.evaluate(protocol, self.runs(protocol, unit_mismatch="first_token_latency_ns"))

    def test_bootstrap_seed_must_fit_declared_splitmix64_state(self):
        protocol = self.protocol()
        base = dataclasses.replace(
            self.base_plan(protocol), bootstrap_seed=1 << 64
        )
        plan = dataclasses.replace(self.decision_plan(protocol), base_plan=base)
        with self.assertRaisesRegex(BikvTargetDecisionError, "fit u64"):
            plan.validate_against(protocol)

    def test_decision_plan_requires_all_core_guards(self):
        protocol = self.protocol()
        plan = dataclasses.replace(
            self.decision_plan(protocol), candidate_metric_guards=self.guards()[:-1]
        )
        with self.assertRaisesRegex(BikvTargetDecisionError, "missing core gates"):
            plan.validate_against(protocol)

    def test_decision_plan_canonical_roundtrip(self):
        protocol = self.protocol()
        plan = self.decision_plan(protocol)
        encoded = plan.canonical_json()
        self.assertEqual(BikvTargetDecisionPlanV2.from_canonical_json(encoded), plan)
        self.assertEqual(len(plan.plan_sha256()), 64)
        with self.assertRaisesRegex(BikvTargetDecisionError, "canonical encoding"):
            BikvTargetDecisionPlanV2.from_canonical_json(json.dumps(plan.to_dict(), indent=2))

    def test_bootstrap_mirror_matches_pinned_scirust_worker_fixture(self):
        # SciRust be7fcca3..., scirust-research-stats-json/v1,
        # paired_mean_percentile([3,-1,5,2], 1000, 0.95, 20260917).
        self.assertEqual(
            _bootstrap_interval(
                [3.0, -1.0, 5.0, 2.0],
                statistic="paired_mean",
                confidence_ppm=950_000,
                resamples=1000,
                seed=20260917,
            ),
            (2.25, -0.25, 4.25),
        )

    def test_bootstrap_is_deterministic(self):
        protocol = self.protocol()
        runs = self.runs(protocol)
        left = self.evaluate(protocol, runs)
        right = self.evaluate(protocol, runs)
        self.assertEqual(left, right)
        self.assertEqual(left.decision_sha256(), right.decision_sha256())

    def test_paired_t_plan_is_rejected_until_scirust_owns_the_primitive(self):
        protocol = self.protocol()
        plan = self.decision_plan(protocol, uncertainty="paired_t_interval")
        with self.assertRaisesRegex(
            BikvTargetDecisionError, "canonical SciRust primitive"
        ):
            plan.validate_against(protocol)

    def test_statistics_source_binding_cannot_drift(self):
        protocol = self.protocol()
        plan = dataclasses.replace(
            self.decision_plan(protocol), statistics_source_commit="0" * 40
        )
        with self.assertRaisesRegex(BikvTargetDecisionError, "qualified SciRust"):
            plan.validate_against(protocol)


if __name__ == "__main__":
    unittest.main()
