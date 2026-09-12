import unittest

from kvlab.boolean_kv import pack_bits
from kvlab.boolean_kv_cpu import (
    CpuPackedScanError,
    benchmark_packed_scan,
    packed_hamming_distance,
    scan_packed_pages,
)


class PackedCpuBaselineTests(unittest.TestCase):
    def test_packed_distance_matches_expected_bits(self) -> None:
        left = pack_bits([False, True, True, False, True])
        right = pack_bits([True, True, False, False, True])
        self.assertEqual(packed_hamming_distance(left, right), 2)

    def test_scan_returns_page_order_candidates_and_exact_accounting(self) -> None:
        query = pack_bits([False, False, False, False])
        pages = [
            pack_bits([False, False, False, False]),
            pack_bits([True, False, False, False]),
            pack_bits([True, True, False, False]),
            pack_bits([True, True, True, True]),
        ]
        result = scan_packed_pages(
            query=query,
            page_signatures=pages,
            max_distance=1,
        )
        self.assertEqual(result.selected_pages, (0, 1))
        self.assertEqual(result.pages_scanned, 4)
        self.assertEqual(result.signature_bits, 4)
        self.assertEqual(result.bits_compared, 16)

    def test_scan_validation_fails_closed(self) -> None:
        query = pack_bits([False, False])
        with self.assertRaises(CpuPackedScanError):
            scan_packed_pages(query=query, page_signatures=[], max_distance=1)
        with self.assertRaises(CpuPackedScanError):
            scan_packed_pages(
                query=query,
                page_signatures=[pack_bits([False, False])],
                max_distance=3,
            )
        with self.assertRaises(CpuPackedScanError):
            scan_packed_pages(
                query=query,
                page_signatures=[pack_bits([False])],
                max_distance=1,
            )

    def test_benchmark_preserves_candidate_set_and_reports_positive_rates(self) -> None:
        query = pack_bits([False] * 64)
        pages = [
            pack_bits([False] * 64),
            pack_bits([True] + [False] * 63),
            pack_bits([True, True] + [False] * 62),
        ]
        report = benchmark_packed_scan(
            query=query,
            page_signatures=pages,
            max_distance=1,
            warmup_queries=1,
            measured_queries=3,
        )
        self.assertEqual(report.selected_pages, (0, 1))
        self.assertEqual(report.pages, 3)
        self.assertEqual(report.signature_bits, 64)
        self.assertGreater(report.median_query_ns, 0)
        self.assertGreater(report.pages_per_second, 0.0)
        self.assertGreater(report.bits_per_second, 0.0)
        self.assertGreater(report.queries_per_second, 0.0)

    def test_benchmark_count_validation_fails_closed(self) -> None:
        query = pack_bits([False])
        pages = [pack_bits([False])]
        with self.assertRaises(CpuPackedScanError):
            benchmark_packed_scan(
                query=query,
                page_signatures=pages,
                max_distance=0,
                warmup_queries=0,
            )


if __name__ == "__main__":
    unittest.main()
