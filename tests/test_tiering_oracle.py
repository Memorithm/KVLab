import unittest

from kvlab.oracle import KVLayerRecord, capture_full_cache
from kvlab.tiering import Tier, TieredPage, account_tiering
from kvlab.tiering_oracle import compare_tiering_to_oracle


class TieringOracleTests(unittest.TestCase):
    def oracle(self):
        return capture_full_cache(
            model_revision="model@abc",
            tokenizer_revision="tokenizer@def",
            sequence_length=4,
            records=(
                KVLayerRecord(
                    layer=0,
                    dtype="fp16",
                    key_shape=(4,),
                    value_shape=(4,),
                    key_bytes=b"1234",
                    value_bytes=b"5678",
                ),
            ),
        )

    def test_tiering_is_bound_to_exact_oracle_logical_bytes(self) -> None:
        oracle = self.oracle()
        accounting = account_tiering(
            pages=(
                TieredPage(page_id=0, tier=Tier.GPU),
                TieredPage(page_id=1, tier=Tier.HOST),
            ),
            page_bytes=4,
        )

        comparison = compare_tiering_to_oracle(oracle=oracle, accounting=accounting)

        self.assertEqual(comparison.oracle_logical_bytes, 8)
        self.assertEqual(comparison.logical_cache_bytes, 8)
        self.assertEqual(comparison.gpu_resident_bytes, 4)
        self.assertEqual(comparison.host_resident_bytes, 4)
        self.assertEqual(comparison.secondary_resident_bytes, 0)
        self.assertAlmostEqual(comparison.gpu_resident_fraction, 0.5)
        self.assertIsNone(comparison.measured_transfer_bytes)

    def test_logical_byte_mismatch_fails_closed(self) -> None:
        accounting = account_tiering(
            pages=(TieredPage(page_id=0, tier=Tier.GPU),),
            page_bytes=4,
        )

        with self.assertRaises(ValueError):
            compare_tiering_to_oracle(oracle=self.oracle(), accounting=accounting)

    def test_tampered_oracle_fails_closed(self) -> None:
        oracle = self.oracle()
        tampered = type(oracle)(
            schema_version=oracle.schema_version,
            model_revision=oracle.model_revision,
            tokenizer_revision=oracle.tokenizer_revision,
            sequence_length=oracle.sequence_length,
            records=oracle.records,
            digest_sha256="0" * 64,
        )
        accounting = account_tiering(
            pages=(
                TieredPage(page_id=0, tier=Tier.GPU),
                TieredPage(page_id=1, tier=Tier.HOST),
            ),
            page_bytes=4,
        )

        with self.assertRaises(ValueError):
            compare_tiering_to_oracle(oracle=tampered, accounting=accounting)


if __name__ == "__main__":
    unittest.main()
