import dataclasses
import json
import unittest

from kvlab.bikv_target_analysis_plan import (
    BKV_TARGET_ANALYSIS_PLAN_SCHEMA_V1,
    BKV_TARGET_PAIRED_SUMMARY_SCHEMA_V1,
    INCOMPLETE_PAIR_POLICY,
    PROMOTION_RULE,
    BikvTargetAnalysisPlanError,
    BikvTargetAnalysisPlanV1,
)
from kvlab.bikv_target_protocol import (
    BASELINE_FULL_CACHE_NATIVE_PREFILL,
    BKV_TARGET_PROTOCOL_SCHEMA_V1,
    REQUIRED_METRICS,
    BikvTargetProtocolV1,
)


class BikvTargetAnalysisPlanTests(unittest.TestCase):
    def protocol(self):
        return BikvTargetProtocolV1(
            schema=BKV_TARGET_PROTOCOL_SCHEMA_V1,
            campaign_id="analysis-plan-test",
            phase="validation",
            hypothesis_h0="candidate does not improve the declared primary metric",
            hypothesis_h1="candidate improves the declared primary metric without quality regression",
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
            quality_rule="candidate must remain inside the separately declared quality tolerance",
            holdout_policy="protected holdout remains closed",
            required_metrics=REQUIRED_METRICS,
        )

    def plan(self, **changes):
        protocol = self.protocol()
        plan = BikvTargetAnalysisPlanV1(
            schema=BKV_TARGET_ANALYSIS_PLAN_SCHEMA_V1,
            protocol_sha256=protocol.protocol_sha256(),
            campaign_id=protocol.campaign_id,
            paired_summary_schema=BKV_TARGET_PAIRED_SUMMARY_SCHEMA_V1,
            hypothesis_h0=protocol.hypothesis_h0,
            hypothesis_h1=protocol.hypothesis_h1,
            primary_performance_metric="first_token_latency_ns",
            primary_direction="lower_is_better",
            primary_minimum_effect=0,
            primary_unit="ns",
            quality_metric=protocol.quality_metric,
            quality_direction="higher_is_better",
            quality_noninferiority_margin=0,
            quality_unit="score",
            uncertainty_method="paired_percentile_bootstrap",
            confidence_level_ppm=950_000,
            bootstrap_resamples=10_000,
            bootstrap_seed=20260917,
            required_complete_pairs=len(protocol.seeds) * protocol.repetitions,
            incomplete_pair_policy=INCOMPLETE_PAIR_POLICY,
            multiplicity_policy="single frozen primary metric; quality is a guard, not a second winner test",
            holdout_policy=protocol.holdout_policy,
            created_before_outcome_inspection=True,
            tuning_permitted=False,
            promotion_rule=PROMOTION_RULE,
        )
        return dataclasses.replace(plan, **changes)

    def test_canonical_roundtrip_and_hash(self):
        protocol = self.protocol()
        plan = self.plan()
        plan.validate_against(protocol)
        encoded = plan.canonical_json()
        decoded = BikvTargetAnalysisPlanV1.from_canonical_json(encoded)
        self.assertEqual(decoded, plan)
        self.assertEqual(len(plan.plan_sha256()), 64)
        self.assertEqual(json.loads(encoded)["promotion_rule"], PROMOTION_RULE)

    def test_protocol_substitution_fails_closed(self):
        protocol = dataclasses.replace(self.protocol(), campaign_id="other")
        with self.assertRaisesRegex(BikvTargetAnalysisPlanError, "does not bind"):
            self.plan().validate_against(protocol)

    def test_quality_metric_must_match_protocol(self):
        with self.assertRaisesRegex(BikvTargetAnalysisPlanError, "quality_metric must match"):
            self.plan(quality_metric="candidate_recall").validate_against(self.protocol())

    def test_every_preregistered_pair_is_required(self):
        with self.assertRaisesRegex(BikvTargetAnalysisPlanError, "every preregistered"):
            self.plan(required_complete_pairs=3).validate_against(self.protocol())

    def test_outcome_inspection_declaration_is_mandatory(self):
        with self.assertRaisesRegex(BikvTargetAnalysisPlanError, "created_before_outcome_inspection"):
            self.plan(created_before_outcome_inspection=False).validate()

    def test_tuning_cannot_be_enabled(self):
        with self.assertRaisesRegex(BikvTargetAnalysisPlanError, "cannot permit tuning"):
            self.plan(tuning_permitted=True).validate()

    def test_boolean_metric_cannot_be_primary_numeric_metric(self):
        with self.assertRaisesRegex(BikvTargetAnalysisPlanError, "must be numeric"):
            self.plan(primary_performance_metric="reset_reuse_correctness").validate()

    def test_bootstrap_requires_frozen_parameters(self):
        with self.assertRaisesRegex(BikvTargetAnalysisPlanError, "requires bootstrap"):
            self.plan(bootstrap_seed=None).validate()

    def test_t_interval_rejects_bootstrap_parameters(self):
        with self.assertRaisesRegex(BikvTargetAnalysisPlanError, "must not carry bootstrap"):
            self.plan(uncertainty_method="paired_t_interval").validate()
        plan = self.plan(
            uncertainty_method="paired_t_interval",
            bootstrap_resamples=None,
            bootstrap_seed=None,
        )
        plan.validate_against(self.protocol())

    def test_noncanonical_json_is_rejected(self):
        encoded = json.dumps(self.plan().to_dict(), indent=2, sort_keys=True)
        with self.assertRaisesRegex(BikvTargetAnalysisPlanError, "canonical encoding"):
            BikvTargetAnalysisPlanV1.from_canonical_json(encoded)


if __name__ == "__main__":
    unittest.main()
