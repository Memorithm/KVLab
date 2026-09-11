import unittest

from kvlab.latency import (
    KVLatencyAccounting,
    LatencyError,
    LatencyKind,
    LatencyQuantity,
    estimated,
    measured,
    not_exposed,
)


class KVLatencyAccountingTests(unittest.TestCase):
    def test_prefill_decode_and_transform_costs_remain_separate(self):
        accounting = KVLatencyAccounting(
            prefill=measured(120.0),
            ttft=measured(128.0),
            decode_total=measured(80.0),
            tpot=measured(4.0),
            cache_transform=measured(9.0),
            cache_reconstruction=not_exposed(),
        )
        accounting.validate()
        self.assertEqual(
            accounting.measured_values(),
            {
                "prefill_ms": 120.0,
                "ttft_ms": 128.0,
                "decode_total_ms": 80.0,
                "tpot_ms": 4.0,
                "cache_transform_ms": 9.0,
            },
        )

    def test_estimates_are_not_reported_as_measurements(self):
        accounting = KVLatencyAccounting(
            prefill=estimated(100.0),
            ttft=not_exposed(),
            decode_total=not_exposed(),
            tpot=not_exposed(),
            cache_transform=not_exposed(),
            cache_reconstruction=not_exposed(),
        )
        accounting.validate()
        self.assertEqual(accounting.measured_values(), {})

    def test_not_exposed_cannot_carry_a_number(self):
        with self.assertRaises(LatencyError):
            LatencyQuantity(1.0, LatencyKind.NOT_EXPOSED).validate()

    def test_negative_or_nonfinite_latency_fails_closed(self):
        for value in (-1.0, float("inf"), float("nan")):
            with self.assertRaises(LatencyError):
                measured(value)


if __name__ == "__main__":
    unittest.main()
