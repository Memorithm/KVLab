import unittest

from kvlab.cross_model import (
    CrossModelTransferProtocol,
    KVGeometry,
    MapperBaseline,
    ModelRevision,
    TransferApplicability,
    assess_cross_model_transfer,
)


def model(model_id: str, revision: str, *, kv_heads: int = 8, head_dim: int = 128):
    return ModelRevision(
        model_id=model_id,
        revision=revision,
        tokenizer_revision=f"{revision}-tokenizer",
        kv=KVGeometry(kv_heads=kv_heads, head_dim=head_dim),
    )


def protocol(**overrides):
    values = dict(
        source=model("family/source", "src-sha"),
        target=model("family/target", "tgt-sha"),
        mapper=MapperBaseline.RIDGE,
        ridge_alpha=1.0,
        source_layers_per_target=4,
        remove_rope_from_keys=True,
        calibration_ids=("cal-001", "cal-002"),
        final_holdout_ids=("holdout-001",),
    )
    values.update(overrides)
    return CrossModelTransferProtocol(**values)


class CrossModelTransferProtocolTests(unittest.TestCase):
    def test_matched_geometry_with_rope_removal_is_applicable(self):
        decision = assess_cross_model_transfer(protocol())
        self.assertEqual(decision.applicability, TransferApplicability.APPLICABLE)
        self.assertTrue(decision.is_applicable)

    def test_mismatched_kv_geometry_is_not_applicable(self):
        decision = assess_cross_model_transfer(
            protocol(target=model("family/target", "tgt-sha", kv_heads=4))
        )
        self.assertEqual(decision.applicability, TransferApplicability.NOT_APPLICABLE)
        self.assertIn("matched KV-head count", decision.reasons[0])

    def test_ridge_without_rope_removal_is_rejected(self):
        decision = assess_cross_model_transfer(protocol(remove_rope_from_keys=False))
        self.assertEqual(decision.applicability, TransferApplicability.NOT_APPLICABLE)
        self.assertTrue(any("RoPE removal" in reason for reason in decision.reasons))

    def test_holdout_leakage_is_rejected_before_scheduling(self):
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            protocol(final_holdout_ids=("cal-002",))

    def test_duplicate_calibration_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            protocol(calibration_ids=("cal-001", "cal-001"))

    def test_identical_source_and_target_revision_is_not_cross_model(self):
        revision = model("family/model", "same-sha")
        decision = assess_cross_model_transfer(protocol(source=revision, target=revision))
        self.assertEqual(decision.applicability, TransferApplicability.NOT_APPLICABLE)
        self.assertTrue(any("identical" in reason for reason in decision.reasons))


class CrossModelProtocolIntegrityTests(unittest.TestCase):
    def test_caller_lists_are_snapshotted_before_later_mutation(self):
        calibration = ["cal-002", "cal-001"]
        holdout = ["holdout-001"]
        frozen = protocol(calibration_ids=calibration, final_holdout_ids=holdout)
        calibration.append("holdout-001")
        holdout[:] = ["cal-001"]
        self.assertEqual(frozen.calibration_ids, ("cal-002", "cal-001"))
        self.assertEqual(frozen.final_holdout_ids, ("holdout-001",))
        self.assertTrue(assess_cross_model_transfer(frozen).is_applicable)

    def test_identity_bytes_and_order_are_preserved(self):
        frozen = protocol(calibration_ids=(" cal-002 ", "cal-001"))
        self.assertEqual(frozen.calibration_ids, (" cal-002 ", "cal-001"))
        self.assertIsInstance(frozen.calibration_ids, tuple)

    def test_split_collections_require_explicit_ordered_sequences(self):
        for field in ("calibration_ids", "final_holdout_ids"):
            for value in ("abc", b"abc", {"a"}, {"a": 1}, None, iter(["a"])):
                with self.subTest(field=field, value=repr(value)):
                    with self.assertRaisesRegex(ValueError, "list or tuple"):
                        protocol(**{field: value})

    def test_split_entries_require_non_empty_strings(self):
        for field in ("calibration_ids", "final_holdout_ids"):
            for value in ((), (" ",), (42,), (None,), (["a"],)):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        protocol(**{field: value})

    def test_non_finite_or_negative_ridge_alpha_is_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf"), -1.0):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "ridge_alpha"):
                    protocol(ridge_alpha=value)

    def test_invalid_ridge_scalar_types_are_rejected(self):
        for value in (True, False, "1.0", None, [], 10**400):
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(ValueError, "ridge_alpha"):
                    protocol(ridge_alpha=value)

    def test_geometry_requires_positive_integer_counts_not_booleans(self):
        for field in ("kv_heads", "head_dim"):
            for value in (True, False, 1.5, "8", None, 0, -1, float("nan")):
                with self.subTest(field=field, value=value):
                    fields = dict(kv_heads=8, head_dim=128)
                    fields[field] = value
                    with self.assertRaisesRegex(ValueError, field):
                        KVGeometry(**fields)

    def test_source_layer_count_requires_a_positive_integer(self):
        for value in (True, 1.5, "4", None, 0, -1, float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "source_layers_per_target"):
                    protocol(source_layers_per_target=value)

    def test_mapper_and_rope_types_cannot_bypass_the_ridge_gate(self):
        for value in ("ridge", "linear", "unknown", None, True):
            with self.subTest(mapper=value):
                with self.assertRaisesRegex(ValueError, "mapper"):
                    protocol(mapper=value, remove_rope_from_keys=False)
        for value in ("False", "True", 0, 1, None):
            with self.subTest(rope=value):
                with self.assertRaisesRegex(ValueError, "remove_rope_from_keys"):
                    protocol(remove_rope_from_keys=value)

    def test_model_metadata_requires_strings_and_validated_geometry(self):
        fields = dict(model_id="family/source", revision="src-sha",
                      tokenizer_revision="tok-sha", kv=KVGeometry(8, 128))
        for field in ("model_id", "revision", "tokenizer_revision"):
            for value in (None, 1, "", " "):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field):
                        ModelRevision(**dict(fields, **{field: value}))
        with self.assertRaisesRegex(ValueError, "kv"):
            ModelRevision(**dict(fields, kv="8x128"))

    def test_protocol_requires_validated_model_revisions(self):
        for field in ("source", "target"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    protocol(**{field: "family/model"})

    def test_zero_ridge_and_existing_linear_ablation_are_preserved(self):
        for value in (0, 0.0, 1, 1.0):
            with self.subTest(alpha=value):
                frozen = protocol(ridge_alpha=value)
                self.assertEqual(frozen.ridge_alpha, value)
                self.assertIs(type(frozen.ridge_alpha), type(value))
                self.assertTrue(assess_cross_model_transfer(frozen).is_applicable)
        linear = protocol(mapper=MapperBaseline.LINEAR, remove_rope_from_keys=False)
        self.assertTrue(assess_cross_model_transfer(linear).is_applicable)


if __name__ == "__main__":
    unittest.main()
