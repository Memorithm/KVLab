import unittest

from kvlab.quantization import (
    BackendQuantizationCapability,
    KvRepresentation,
    QuantizationConfig,
)


class QuantizationContractTests(unittest.TestCase):
    def test_theoretical_payload_is_separate_from_metadata(self) -> None:
        config = QuantizationConfig(
            source=KvRepresentation.FP16,
            target=KvRepresentation.INT4,
            values=1024,
            group_size=64,
            metadata_bits=512,
        )
        self.assertEqual(config.theoretical_source_payload_bits, 16384)
        self.assertEqual(config.theoretical_target_payload_bits, 4096)
        self.assertEqual(config.theoretical_target_total_bits, 4608)
        self.assertEqual(config.theoretical_payload_reduction_bits, 12288)

    def test_backend_support_is_explicit(self) -> None:
        capability = BackendQuantizationCapability(
            backend="cpu-reference",
            supported_targets=frozenset({KvRepresentation.INT8}),
        )
        capability.require_supported(KvRepresentation.INT8)
        with self.assertRaisesRegex(ValueError, "does not support int4"):
            capability.require_supported(KvRepresentation.INT4)

    def test_invalid_configs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            QuantizationConfig(KvRepresentation.FP16, KvRepresentation.INT8, 0)
        with self.assertRaises(ValueError):
            QuantizationConfig(KvRepresentation.FP16, KvRepresentation.FP16, 1)
        with self.assertRaises(ValueError):
            QuantizationConfig(
                KvRepresentation.FP16,
                KvRepresentation.INT8,
                1,
                group_size=0,
            )
        with self.assertRaises(ValueError):
            QuantizationConfig(
                KvRepresentation.FP16,
                KvRepresentation.INT8,
                1,
                metadata_bits=-1,
            )


if __name__ == "__main__":
    unittest.main()
