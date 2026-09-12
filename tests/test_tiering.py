import unittest

from kvlab.tiering import Tier, TieredPage, account_tiering


class TieringAccountingTests(unittest.TestCase):
    def test_residency_partitions_logical_cache(self) -> None:
        accounting = account_tiering(
            pages=(
                TieredPage(page_id=0, tier=Tier.GPU),
                TieredPage(page_id=1, tier=Tier.GPU),
                TieredPage(page_id=2, tier=Tier.HOST),
                TieredPage(page_id=3, tier=Tier.SECONDARY),
            ),
            page_bytes=4096,
        )

        self.assertEqual(accounting.logical_cache_bytes, 4 * 4096)
        self.assertEqual(accounting.gpu_resident_bytes, 2 * 4096)
        self.assertEqual(accounting.host_resident_bytes, 4096)
        self.assertEqual(accounting.secondary_resident_bytes, 4096)
        self.assertEqual(accounting.accounted_resident_bytes, 4 * 4096)
        self.assertIsNone(accounting.measured_transfer_bytes)

    def test_tiering_does_not_claim_logical_cache_reduction(self) -> None:
        gpu = account_tiering(
            pages=(TieredPage(page_id=0, tier=Tier.GPU),),
            page_bytes=1024,
        )
        host = account_tiering(
            pages=(TieredPage(page_id=0, tier=Tier.HOST),),
            page_bytes=1024,
        )

        self.assertEqual(gpu.logical_cache_bytes, host.logical_cache_bytes)
        self.assertEqual(gpu.logical_cache_bytes, 1024)
        self.assertEqual(gpu.gpu_resident_bytes, 1024)
        self.assertEqual(host.gpu_resident_bytes, 0)

    def test_transfer_volume_requires_explicit_measurement(self) -> None:
        unmeasured = account_tiering(
            pages=(TieredPage(page_id=0, tier=Tier.HOST),),
            page_bytes=512,
        )
        measured = account_tiering(
            pages=(TieredPage(page_id=0, tier=Tier.HOST),),
            page_bytes=512,
            measured_transfer_bytes=512,
        )

        self.assertIsNone(unmeasured.measured_transfer_bytes)
        self.assertEqual(measured.measured_transfer_bytes, 512)

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            TieredPage(page_id=-1, tier=Tier.GPU)
        with self.assertRaises(ValueError):
            account_tiering(pages=(), page_bytes=0)
        with self.assertRaises(ValueError):
            account_tiering(
                pages=(TieredPage(page_id=0, tier=Tier.GPU),),
                page_bytes=1024,
                measured_transfer_bytes=-1,
            )
        with self.assertRaises(ValueError):
            account_tiering(
                pages=(
                    TieredPage(page_id=0, tier=Tier.GPU),
                    TieredPage(page_id=0, tier=Tier.HOST),
                ),
                page_bytes=1024,
            )


if __name__ == "__main__":
    unittest.main()
