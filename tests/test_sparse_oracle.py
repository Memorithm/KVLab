import unittest

from kvlab.oracle import FullCacheSnapshot, KVLayerRecord, capture_full_cache
from kvlab.sparse_oracle import compare_sparse_read_to_oracle
from kvlab.sparse_read import PageScore, QueryAwarePagePolicy, select_query_aware_pages


class SparseReadOracleTests(unittest.TestCase):
    @staticmethod
    def _oracle() -> FullCacheSnapshot:
        return capture_full_cache(
            model_revision="model@abc123",
            tokenizer_revision="tokenizer@def456",
            sequence_length=4,
            records=(
                KVLayerRecord(
                    layer=0,
                    dtype="fp16",
                    key_shape=(4,),
                    value_shape=(4,),
                    key_bytes=b"kkkk",
                    value_bytes=b"vvvv",
                ),
            ),
        )

    def test_sparse_selection_is_bound_to_unchanged_logical_cache(self) -> None:
        oracle = self._oracle()
        accounting = select_query_aware_pages(
            scores=(
                PageScore(page_id=0, score=0.1),
                PageScore(page_id=1, score=0.9),
            ),
            page_bytes=4,
            policy=QueryAwarePagePolicy(max_pages=1),
        )

        comparison = compare_sparse_read_to_oracle(
            oracle=oracle,
            accounting=accounting,
        )

        self.assertEqual(comparison.oracle_logical_bytes, 8)
        self.assertEqual(comparison.logical_cache_bytes, 8)
        self.assertEqual(comparison.selected_read_bytes, 4)
        self.assertEqual(comparison.total_pages, 2)
        self.assertEqual(comparison.selected_pages, 1)
        self.assertEqual(comparison.skipped_pages, 1)
        self.assertAlmostEqual(comparison.selected_page_fraction, 0.5)
        self.assertAlmostEqual(comparison.candidate_read_fraction, 0.5)

    def test_mismatched_logical_cache_fails_closed(self) -> None:
        oracle = self._oracle()
        accounting = select_query_aware_pages(
            scores=(PageScore(page_id=0, score=1.0),),
            page_bytes=4,
            policy=QueryAwarePagePolicy(max_pages=1),
        )

        with self.assertRaises(ValueError):
            compare_sparse_read_to_oracle(oracle=oracle, accounting=accounting)

    def test_tampered_oracle_fails_closed(self) -> None:
        oracle = self._oracle()
        tampered = FullCacheSnapshot(
            schema_version=oracle.schema_version,
            model_revision=oracle.model_revision,
            tokenizer_revision=oracle.tokenizer_revision,
            sequence_length=oracle.sequence_length,
            records=oracle.records,
            digest_sha256="0" * 64,
        )
        accounting = select_query_aware_pages(
            scores=(
                PageScore(page_id=0, score=1.0),
                PageScore(page_id=1, score=0.5),
            ),
            page_bytes=4,
            policy=QueryAwarePagePolicy(max_pages=1),
        )

        with self.assertRaises(ValueError):
            compare_sparse_read_to_oracle(oracle=tampered, accounting=accounting)


if __name__ == "__main__":
    unittest.main()
