import unittest

from kvlab.boolean_kv import pack_bits
from kvlab.boolean_kv_cpu import scan_packed_pages
from kvlab.boolean_kv_multicore import (
    MulticoreScanError,
    contiguous_page_shards,
    scan_sharded_pages,
)


class MulticoreShardContractTests(unittest.TestCase):
    def test_partition_is_balanced_contiguous_and_complete(self) -> None:
        shards = contiguous_page_shards(page_count=10, workers=3)
        self.assertEqual(
            [(s.shard_id, s.start_page, s.end_page) for s in shards],
            [(0, 0, 4), (1, 4, 7), (2, 7, 10)],
        )
        self.assertLessEqual(max(s.page_count for s in shards) - min(s.page_count for s in shards), 1)

    def test_workers_above_page_count_do_not_create_empty_shards(self) -> None:
        shards = contiguous_page_shards(page_count=3, workers=8)
        self.assertEqual([(s.start_page, s.end_page) for s in shards], [(0, 1), (1, 2), (2, 3)])

    def test_sharded_oracle_matches_single_scan_exactly(self) -> None:
        query = pack_bits([False, False, False, False])
        pages = [
            pack_bits([False, False, False, False]),
            pack_bits([True, False, False, False]),
            pack_bits([True, True, False, False]),
            pack_bits([False, True, False, False]),
            pack_bits([True, True, True, True]),
        ]
        baseline = scan_packed_pages(query=query, page_signatures=pages, max_distance=1)
        sharded = scan_sharded_pages(
            query=query,
            page_signatures=pages,
            max_distance=1,
            workers=3,
        )
        self.assertEqual(sharded.selected_pages, baseline.selected_pages)
        self.assertEqual(sharded.pages_scanned, baseline.pages_scanned)
        self.assertEqual(sharded.bits_compared, baseline.bits_compared)
        self.assertEqual(sharded.signature_bits, baseline.signature_bits)

    def test_invalid_partition_requests_fail_closed(self) -> None:
        with self.assertRaises(MulticoreScanError):
            contiguous_page_shards(page_count=0, workers=1)
        with self.assertRaises(MulticoreScanError):
            contiguous_page_shards(page_count=1, workers=0)


if __name__ == "__main__":
    unittest.main()
