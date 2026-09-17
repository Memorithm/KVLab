import unittest

from kvlab.bikv_target_protocol import (
    BASELINE_FULL_CACHE_NATIVE_PREFILL,
    BKV_TARGET_PROTOCOL_SCHEMA_V1,
    REQUIRED_METRICS,
    BikvTargetProtocolError,
    BikvTargetProtocolV1,
)


class BikvTargetProtocolTests(unittest.TestCase):
    def protocol(self, **overrides):
        values = dict(
            schema=BKV_TARGET_PROTOCOL_SCHEMA_V1,
            campaign_id="bkv-k6-target-001",
            phase="validation",
            hypothesis_h0="matched BIKV does not improve the declared frontier",
            hypothesis_h1="matched BIKV improves the declared frontier",
            evidence_bundle_sha256="a" * 64,
            kvlab_commit="b" * 40,
            flat_commit="c" * 40,
            model_id="example/model",
            model_revision="d" * 40,
            tokenizer_id="example/tokenizer",
            tokenizer_revision="e" * 40,
            runtime_id="example/runtime",
            runtime_revision="f" * 40,
            hardware_fingerprint_sha256="1" * 64,
            precision="bf16",
            context_tokens=4096,
            batch_size=1,
            dataset_id="example/dataset",
            dataset_revision="2" * 40,
            partition_id="validation-v1",
            partition_role="validation",
            tuning_permitted=False,
            seeds=(11, 29, 47),
            warmup_runs=2,
            repetitions=5,
            boolean_policy="frozen-signature-policy-v1",
            baseline_policy=BASELINE_FULL_CACHE_NATIVE_PREFILL,
            timing_source="device_timestamp",
            byte_evidence_kind="logical_packed_payload",
            quality_metric="exact-output-parity",
            quality_rule="candidate must satisfy preregistered tolerance",
            holdout_policy="protected final holdout is never used for tuning",
            required_metrics=REQUIRED_METRICS,
        )
        values.update(overrides)
        return BikvTargetProtocolV1(**values)

    def test_canonical_round_trip_and_identity(self):
        protocol = self.protocol()
        payload = protocol.canonical_json()
        decoded = BikvTargetProtocolV1.from_canonical_json(payload)
        self.assertEqual(decoded, protocol)
        self.assertEqual(len(protocol.protocol_sha256()), 64)
        self.assertIn("\"required_metrics\"", payload)

    def test_required_metric_set_is_frozen(self):
        with self.assertRaisesRegex(BikvTargetProtocolError, "required_metrics"):
            self.protocol(required_metrics=REQUIRED_METRICS[:-1]).validate()

    def test_protected_holdout_cannot_tune(self):
        with self.assertRaisesRegex(BikvTargetProtocolError, "protected holdout"):
            self.protocol(partition_role="protected_holdout", tuning_permitted=True).validate()

    def test_confirmatory_phase_cannot_tune(self):
        with self.assertRaisesRegex(BikvTargetProtocolError, "confirmatory"):
            self.protocol(phase="confirmatory", tuning_permitted=True).validate()

    def test_requires_full_immutable_revisions(self):
        with self.assertRaisesRegex(BikvTargetProtocolError, "model_revision"):
            self.protocol(model_revision="main").validate()
        with self.assertRaisesRegex(BikvTargetProtocolError, "runtime_revision"):
            self.protocol(runtime_revision="v1.2.3").validate()

    def test_requires_full_cache_native_prefill_baseline(self):
        with self.assertRaisesRegex(BikvTargetProtocolError, "baseline_policy"):
            self.protocol(baseline_policy="paged-only").validate()

    def test_rejects_mixed_or_unknown_byte_evidence(self):
        with self.assertRaisesRegex(BikvTargetProtocolError, "byte_evidence_kind"):
            self.protocol(byte_evidence_kind="estimated-mixed").validate()

    def test_rejects_noncanonical_json(self):
        payload = self.protocol().canonical_json()
        with self.assertRaisesRegex(BikvTargetProtocolError, "not canonical"):
            BikvTargetProtocolV1.from_canonical_json(payload + "\n")


    def test_k6_transfer_and_full_roadmap_metrics_are_frozen(self):
        required = set(REQUIRED_METRICS)
        for metric in (
            "query_signature_transfer_bytes",
            "candidate_bitmap_transfer_bytes",
            "synchronization_wait_ns",
            "dispatch_count",
            "backpressure_wait_ns",
            "reset_reuse_correctness",
            "numerical_kv_bytes_touched",
            "pages_per_second",
            "scaling_efficiency",
        ):
            self.assertIn(metric, required)

    def test_rejects_noncanonical_revision_and_digest_spelling(self):
        with self.assertRaisesRegex(BikvTargetProtocolError, "model_revision"):
            self.protocol(model_revision="D" * 40).validate()
        with self.assertRaisesRegex(BikvTargetProtocolError, "kvlab_commit"):
            self.protocol(kvlab_commit=" " + "b" * 40).validate()
        with self.assertRaisesRegex(BikvTargetProtocolError, "evidence_bundle_sha256"):
            self.protocol(evidence_bundle_sha256="A" * 64).validate()

    def test_malformed_enum_types_fail_as_protocol_errors(self):
        for field in ("phase", "partition_role", "timing_source", "byte_evidence_kind"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(BikvTargetProtocolError, field):
                    self.protocol(**{field: []}).validate()

    def test_rejects_duplicate_seeds(self):
        with self.assertRaisesRegex(BikvTargetProtocolError, "duplicates"):
            self.protocol(seeds=(7, 7)).validate()


if __name__ == "__main__":
    unittest.main()
