import json
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
    BikvTargetRunError,
    BikvTargetRunV1,
)


class BikvTargetRunTests(unittest.TestCase):
    def protocol(self):
        return BikvTargetProtocolV1(
            schema=BKV_TARGET_PROTOCOL_SCHEMA_V1,
            campaign_id="bkv-k6-run-test",
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
            repetitions=3,
            boolean_policy="policy-v1",
            baseline_policy=BASELINE_FULL_CACHE_NATIVE_PREFILL,
            timing_source="device_timestamp",
            byte_evidence_kind="logical_packed_payload",
            quality_metric="exact-output-parity",
            quality_rule="must satisfy frozen tolerance",
            holdout_policy="protected holdout remains closed",
            required_metrics=REQUIRED_METRICS,
        )

    def metrics(self, *, failed_name=None):
        metrics = []
        for name in REQUIRED_METRICS:
            if name == failed_name:
                metrics.append(
                    BikvMetricObservationV1(
                        name=name,
                        status="failed",
                        value=None,
                        unit=None,
                        reason="measurement backend failed",
                    )
                )
            elif name == "candidate_density":
                metrics.append(
                    BikvMetricObservationV1(
                        name=name,
                        status="measured",
                        value=0.5,
                        unit="ratio",
                        reason=None,
                    )
                )
            else:
                metrics.append(
                    BikvMetricObservationV1(
                        name=name,
                        status="not_exposed",
                        value=None,
                        unit=None,
                        reason="fixture backend does not expose this metric",
                    )
                )
        return tuple(metrics)

    def target_run(self, **overrides):
        protocol = self.protocol()
        values = dict(
            schema=BKV_TARGET_RUN_SCHEMA_V1,
            protocol_sha256=protocol.protocol_sha256(),
            campaign_id=protocol.campaign_id,
            attempt_id="candidate-seed7-r0",
            variant="candidate",
            seed=7,
            repetition_index=0,
            status="completed",
            failure_reason=None,
            metrics=self.metrics(),
        )
        values.update(overrides)
        return BikvTargetRunV1(**values)

    def test_canonical_round_trip_and_protocol_binding(self):
        protocol = self.protocol()
        run = self.target_run()
        run.validate_against(protocol)
        decoded = BikvTargetRunV1.from_canonical_json(run.canonical_json())
        self.assertEqual(decoded, run)
        self.assertEqual(len(run.run_sha256()), 64)

    def test_metric_obligations_cannot_be_removed_or_reordered(self):
        with self.assertRaisesRegex(BikvTargetRunError, "required-metric order"):
            self.target_run(metrics=self.metrics()[:-1]).validate()
        reordered = list(self.metrics())
        reordered[0], reordered[1] = reordered[1], reordered[0]
        with self.assertRaisesRegex(BikvTargetRunError, "required-metric order"):
            self.target_run(metrics=tuple(reordered)).validate()

    def test_boolean_and_signed_metric_types_remain_explicit(self):
        with self.assertRaisesRegex(BikvTargetRunError, "candidate_density.*numeric"):
            BikvMetricObservationV1(
                name="candidate_density",
                status="measured",
                value=True,
                unit="ratio",
                reason=None,
            ).validate()
        with self.assertRaisesRegex(BikvTargetRunError, "reset_reuse_correctness.*Boolean"):
            BikvMetricObservationV1(
                name="reset_reuse_correctness",
                status="measured",
                value=1,
                unit="boolean",
                reason=None,
            ).validate()
        BikvMetricObservationV1(
            name="downstream_quality",
            status="measured",
            value=-2.5,
            unit="log_likelihood",
            reason=None,
        ).validate()

    def test_discrete_metric_requires_integer_value(self):
        metrics = list(self.metrics())
        index = REQUIRED_METRICS.index("dispatch_count")
        metrics[index] = BikvMetricObservationV1(
            name="dispatch_count",
            status="measured",
            value=0.5,
            unit="count",
            reason=None,
        )
        with self.assertRaisesRegex(BikvTargetRunError, "dispatch_count.*integer"):
            self.target_run(metrics=tuple(metrics)).validate()

    def test_cross_metric_latency_decomposition_fails_closed(self):
        metrics = list(self.metrics())
        frontend = REQUIRED_METRICS.index("boolean_frontend_ns")
        first_token = REQUIRED_METRICS.index("first_token_latency_ns")
        metrics[frontend] = BikvMetricObservationV1(
            name="boolean_frontend_ns",
            status="measured",
            value=101,
            unit="ns",
            reason=None,
        )
        metrics[first_token] = BikvMetricObservationV1(
            name="first_token_latency_ns",
            status="measured",
            value=100,
            unit="ns",
            reason=None,
        )
        with self.assertRaisesRegex(BikvTargetRunError, "cannot exceed"):
            self.target_run(metrics=tuple(metrics)).validate()

    def test_cross_metric_avoided_read_ratio_fails_closed(self):
        metrics = list(self.metrics())
        avoided = REQUIRED_METRICS.index("numerical_kv_bytes_avoided")
        boolean_read = REQUIRED_METRICS.index("boolean_kv_bytes_read")
        metrics[avoided] = BikvMetricObservationV1(
            name="numerical_kv_bytes_avoided",
            status="measured",
            value=4096,
            unit="bytes",
            reason=None,
        )
        metrics[boolean_read] = BikvMetricObservationV1(
            name="boolean_kv_bytes_read",
            status="measured",
            value=0,
            unit="bytes",
            reason=None,
        )
        with self.assertRaisesRegex(BikvTargetRunError, "requires non-zero"):
            self.target_run(metrics=tuple(metrics)).validate()

    def test_cross_metric_checks_only_apply_when_both_operands_are_measured(self):
        metrics = list(self.metrics())
        avoided = REQUIRED_METRICS.index("numerical_kv_bytes_avoided")
        metrics[avoided] = BikvMetricObservationV1(
            name="numerical_kv_bytes_avoided",
            status="measured",
            value=4096,
            unit="bytes",
            reason=None,
        )
        self.target_run(metrics=tuple(metrics)).validate()

    def test_unavailable_and_failed_metrics_cannot_carry_fake_values(self):
        metric = BikvMetricObservationV1(
            name="tpot_ns_per_token",
            status="not_exposed",
            value=1,
            unit="ns/token",
            reason="not available",
        )
        with self.assertRaisesRegex(BikvTargetRunError, "must not carry"):
            metric.validate()

    def test_failed_runs_are_retained_explicitly(self):
        failed = self.target_run(
            status="failed",
            failure_reason="runtime aborted",
            metrics=self.metrics(failed_name="tpot_ns_per_token"),
        )
        failed.validate_against(self.protocol())
        self.assertIn('"status":"failed"', failed.canonical_json())
        self.assertIn("measurement backend failed", failed.canonical_json())

    def test_completed_run_cannot_hide_failed_metric(self):
        with self.assertRaisesRegex(BikvTargetRunError, "completed run"):
            self.target_run(metrics=self.metrics(failed_name="tpot_ns_per_token")).validate()

    def test_protocol_binding_rejects_foreign_seed_and_identity(self):
        with self.assertRaisesRegex(BikvTargetRunError, "seed"):
            self.target_run(seed=99).validate_against(self.protocol())
        with self.assertRaisesRegex(BikvTargetRunError, "protocol_sha256"):
            self.target_run(protocol_sha256="9" * 64).validate_against(self.protocol())

    def test_noncanonical_json_is_rejected(self):
        payload = self.target_run().canonical_json()
        with self.assertRaisesRegex(BikvTargetRunError, "not canonical"):
            BikvTargetRunV1.from_canonical_json(payload + "\n")
        decoded = json.loads(payload)
        decoded["metrics"] = decoded["metrics"][:-1]
        malformed = json.dumps(decoded, sort_keys=True, separators=(",", ":"))
        with self.assertRaisesRegex(BikvTargetRunError, "required-metric order"):
            BikvTargetRunV1.from_canonical_json(malformed)


if __name__ == "__main__":
    unittest.main()
