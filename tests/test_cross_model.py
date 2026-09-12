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


if __name__ == "__main__":
    unittest.main()
