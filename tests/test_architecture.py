import unittest

from kvlab.architecture import (
    Applicability,
    ArchitectureMechanism,
    ModelArchitectureFeatures,
    assess_architecture_compatibility,
)


class ArchitectureCompatibilityTests(unittest.TestCase):
    def test_requires_pinned_model_identity(self):
        with self.assertRaises(ValueError):
            ModelArchitectureFeatures(model_id="", revision="rev")
        with self.assertRaises(ValueError):
            ModelArchitectureFeatures(model_id="model", revision="")

    def test_mla_is_not_applicable_without_native_latent_attention(self):
        features = ModelArchitectureFeatures(model_id="family/model", revision="abc123")
        result = assess_architecture_compatibility(features, ArchitectureMechanism.MLA)
        self.assertEqual(result.applicability, Applicability.NOT_APPLICABLE)
        self.assertFalse(result.is_applicable)

    def test_mla_is_applicable_when_explicitly_declared(self):
        features = ModelArchitectureFeatures(
            model_id="family/model",
            revision="abc123",
            latent_attention=True,
        )
        result = assess_architecture_compatibility(features, ArchitectureMechanism.MLA)
        self.assertTrue(result.is_applicable)

    def test_cross_layer_and_hybrid_are_gated_independently(self):
        features = ModelArchitectureFeatures(
            model_id="family/model",
            revision="abc123",
            cross_layer_kv_sharing=True,
            state_space_layers=False,
        )
        sharing = assess_architecture_compatibility(
            features, ArchitectureMechanism.CROSS_LAYER_KV_SHARING
        )
        hybrid = assess_architecture_compatibility(
            features, ArchitectureMechanism.HYBRID_SSM
        )
        self.assertTrue(sharing.is_applicable)
        self.assertFalse(hybrid.is_applicable)

    def test_reason_records_declared_fact_without_performance_claim(self):
        features = ModelArchitectureFeatures(
            model_id="family/model",
            revision="abc123",
            state_space_layers=True,
        )
        result = assess_architecture_compatibility(
            features, ArchitectureMechanism.HYBRID_SSM
        )
        self.assertEqual(result.reason, "declared native state-space layers")


if __name__ == "__main__":
    unittest.main()
