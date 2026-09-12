import unittest

from kvlab.instrumentation import (
    KVResourceAccounting,
    MeasurementKind,
    bytes_quantity,
    not_exposed,
    tokens_quantity,
)
from kvlab.quantization import (
    BackendQuantizationCapability,
    KvRepresentation,
    QuantizationAccountingComparison,
    QuantizationConfig,
    QuantizationQualityObservation,
    ReconstructionMetric,
)


def accounting(logical: int, gpu: int | None, kind: MeasurementKind) -> KVResourceAccounting:
    gpu_quantity = not_exposed("bytes") if gpu is None else bytes_quantity(gpu, kind)
    return KVResourceAccounting(
        logical_cache_bytes=bytes_quantity(logical, MeasurementKind.MEASURED),
        gpu_resident_bytes=gpu_quantity,
        host_resident_bytes=not_exposed("bytes"),
        secondary_storage_bytes=not_exposed("bytes"),
        fragmentation_bytes=bytes_quantity(0, MeasurementKind.ESTIMATED),
        host_to_gpu_bytes=not_exposed("bytes"),
        gpu_to_host_bytes=not_exposed("bytes"),
        bytes_read=not_exposed("bytes"),
        bytes_written=not_exposed("bytes"),
        recomputed_token_count=tokens_quantity(0, MeasurementKind.ESTIMATED),
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

    def test_comparison_keeps_logical_and_measured_gpu_savings_separate(self) -> None:
        comparison = QuantizationAccountingComparison(
            config=QuantizationConfig(
                KvRepresentation.FP16,
                KvRepresentation.INT4,
                values=1024,
                metadata_bits=64,
            ),
            source=accounting(2048, 4096, MeasurementKind.MEASURED),
            target=accounting(520, 1536, MeasurementKind.MEASURED),
        )
        self.assertEqual(comparison.theoretical_target_total_bytes_ceiling, 520)
        self.assertEqual(comparison.logical_cache_delta_bytes, 1528.0)
        self.assertEqual(comparison.measured_gpu_residency_delta_bytes, 2560.0)

    def test_estimated_or_unavailable_gpu_residency_never_becomes_measured_savings(self) -> None:
        config = QuantizationConfig(KvRepresentation.FP16, KvRepresentation.INT8, values=64)
        estimated = QuantizationAccountingComparison(
            config,
            accounting(128, 256, MeasurementKind.MEASURED),
            accounting(64, 128, MeasurementKind.ESTIMATED),
        )
        unavailable = QuantizationAccountingComparison(
            config,
            accounting(128, 256, MeasurementKind.MEASURED),
            accounting(64, None, MeasurementKind.NOT_EXPOSED),
        )
        self.assertIsNone(estimated.measured_gpu_residency_delta_bytes)
        self.assertIsNone(unavailable.measured_gpu_residency_delta_bytes)

    def test_reconstruction_quality_preserves_measurement_kind(self) -> None:
        measured = QuantizationQualityObservation(
            ReconstructionMetric.RMSE,
            0.0125,
            MeasurementKind.MEASURED,
        )
        estimated = QuantizationQualityObservation(
            ReconstructionMetric.MAX_ABS_ERROR,
            0.25,
            MeasurementKind.ESTIMATED,
        )
        unavailable = QuantizationQualityObservation(
            ReconstructionMetric.RELATIVE_L2_ERROR,
            None,
            MeasurementKind.NOT_EXPOSED,
        )
        self.assertEqual(measured.measured_value, 0.0125)
        self.assertIsNone(estimated.measured_value)
        self.assertIsNone(unavailable.measured_value)

    def test_invalid_quality_observations_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            QuantizationQualityObservation(
                ReconstructionMetric.RMSE,
                -1.0,
                MeasurementKind.MEASURED,
            )
        with self.assertRaises(ValueError):
            QuantizationQualityObservation(
                ReconstructionMetric.RMSE,
                0.0,
                MeasurementKind.NOT_EXPOSED,
            )


if __name__ == "__main__":
    unittest.main()
