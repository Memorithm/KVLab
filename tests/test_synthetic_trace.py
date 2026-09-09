import math
import unittest

from kvlab import KvRegion, SyntheticKvTrace, evaluate_removal


class SyntheticTraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = SyntheticKvTrace(
            trace_id="trace-a",
            regions=(
                KvRegion("prompt", 16, (1.0, 2.0)),
                KvRegion("middle", 8, (3.0, -1.0)),
                KvRegion("recent", 12, (-2.0, 4.0)),
            ),
        )

    def test_full_cache_output_and_bytes_are_exact(self) -> None:
        self.assertEqual(self.trace.full_cache_output(), (2.0, 5.0))
        self.assertEqual(self.trace.total_storage_bytes, 36)

    def test_removal_records_delta_and_exact_byte_savings(self) -> None:
        result = evaluate_removal(self.trace, "middle")
        self.assertEqual(result.full_cache_output, (2.0, 5.0))
        self.assertEqual(result.intervened_output, (-1.0, 6.0))
        self.assertTrue(math.isclose(result.output_l2_delta, math.sqrt(10.0)))
        self.assertEqual(result.bytes_saved, 8)
        self.assertEqual(result.remaining_bytes, 28)

    def test_unknown_region_fails_closed(self) -> None:
        with self.assertRaises(KeyError):
            evaluate_removal(self.trace, "missing")

    def test_duplicate_ids_and_inconsistent_width_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SyntheticKvTrace(
                "bad-ids",
                (KvRegion("x", 4, (1.0,)), KvRegion("x", 4, (2.0,))),
            )
        with self.assertRaises(ValueError):
            SyntheticKvTrace(
                "bad-width",
                (KvRegion("x", 4, (1.0,)), KvRegion("y", 4, (1.0, 2.0))),
            )


if __name__ == "__main__":
    unittest.main()
