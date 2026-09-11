import unittest

from kvlab.instrumentation import (
    KVResourceAccounting,
    MeasurementKind,
    Quantity,
    ResourceAccountingError,
    bytes_quantity,
    not_exposed,
    tokens_quantity,
)


class KVResourceAccountingTests(unittest.TestCase):
    def test_logical_size_and_gpu_residency_are_separate(self):
        accounting = KVResourceAccounting(
            logical_cache_bytes=bytes_quantity(4096, MeasurementKind.MEASURED),
            gpu_resident_bytes=bytes_quantity(3072, MeasurementKind.MEASURED),
            host_resident_bytes=bytes_quantity(1024, MeasurementKind.MEASURED),
            secondary_storage_bytes=not_exposed("bytes"),
            fragmentation_bytes=bytes_quantity(256, MeasurementKind.ESTIMATED),
            host_to_gpu_bytes=bytes_quantity(512, MeasurementKind.MEASURED),
            gpu_to_host_bytes=bytes_quantity(128, MeasurementKind.MEASURED),
            bytes_read=not_exposed("bytes"),
            bytes_written=not_exposed("bytes"),
            recomputed_token_count=tokens_quantity(0, MeasurementKind.MEASURED),
        )
        accounting.validate()
        self.assertEqual(accounting.logical_cache_bytes.value, 4096.0)
        self.assertEqual(accounting.measured_gpu_residency, 3072.0)
        self.assertIsNone(accounting.estimated_gpu_residency)

    def test_not_exposed_never_carries_a_value(self):
        with self.assertRaises(ResourceAccountingError):
            Quantity(1.0, "bytes", MeasurementKind.NOT_EXPOSED).validate()

    def test_estimate_is_not_reported_as_measurement(self):
        accounting = KVResourceAccounting(
            logical_cache_bytes=bytes_quantity(4096, MeasurementKind.MEASURED),
            gpu_resident_bytes=bytes_quantity(3000, MeasurementKind.ESTIMATED),
            host_resident_bytes=not_exposed("bytes"),
            secondary_storage_bytes=not_exposed("bytes"),
            fragmentation_bytes=not_exposed("bytes"),
            host_to_gpu_bytes=not_exposed("bytes"),
            gpu_to_host_bytes=not_exposed("bytes"),
            bytes_read=not_exposed("bytes"),
            bytes_written=not_exposed("bytes"),
            recomputed_token_count=not_exposed("tokens"),
        )
        accounting.validate()
        self.assertIsNone(accounting.measured_gpu_residency)
        self.assertEqual(accounting.estimated_gpu_residency, 3000.0)

    def test_wrong_units_fail_closed(self):
        accounting = KVResourceAccounting(
            logical_cache_bytes=Quantity(1.0, "bits", MeasurementKind.MEASURED),
            gpu_resident_bytes=not_exposed("bytes"),
            host_resident_bytes=not_exposed("bytes"),
            secondary_storage_bytes=not_exposed("bytes"),
            fragmentation_bytes=not_exposed("bytes"),
            host_to_gpu_bytes=not_exposed("bytes"),
            gpu_to_host_bytes=not_exposed("bytes"),
            bytes_read=not_exposed("bytes"),
            bytes_written=not_exposed("bytes"),
            recomputed_token_count=not_exposed("tokens"),
        )
        with self.assertRaises(ResourceAccountingError):
            accounting.validate()


if __name__ == "__main__":
    unittest.main()
