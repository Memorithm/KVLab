"""Metric integrity regressions; no model execution or observed results."""

import math
import unittest

from kvlab.prospect_real_model_eviction import (
    ObservedMetric,
    ProspectKvRealModelEvidenceError,
    _nearly_equal,
)


class ObservedMetricDeltaFinitenessTests(unittest.TestCase):
    def test_rejects_finite_fields_with_overflowing_derived_delta(self):
        # A hand-built/decoded record can supply a finite delta even when the
        # difference of its finite endpoint values overflows.
        for baseline, candidate in ((-1e308, 1e308), (1e308, -1e308)):
            with self.subTest(baseline=baseline, candidate=candidate):
                self.assertTrue(math.isfinite(baseline))
                self.assertTrue(math.isfinite(candidate))
                self.assertFalse(math.isfinite(candidate - baseline))
                metric = ObservedMetric(
                    name="change", kind="numerical", unit="unit",
                    preference="none", baseline_value=baseline,
                    candidate_value=candidate, delta=0.0,
                )
                with self.assertRaises(ProspectKvRealModelEvidenceError):
                    metric.validate()

    def test_nonfinite_comparison_operands_never_match(self):
        for invalid in (math.inf, -math.inf, math.nan):
            for finite in (0.0, 1.0, -1.0, 1e308):
                with self.subTest(invalid=invalid, finite=finite):
                    self.assertFalse(_nearly_equal(finite, invalid))
                    self.assertFalse(_nearly_equal(invalid, finite))
            self.assertFalse(_nearly_equal(invalid, invalid))

    def test_large_but_finite_metric_delta_is_preserved(self):
        metric = ObservedMetric.capture(
            name="change", kind="numerical", unit="unit", preference="none",
            baseline_value=1e307, candidate_value=2e307,
        )
        metric.validate()
        self.assertTrue(math.isfinite(metric.delta))
        self.assertEqual(metric.delta, 1e307)

    def test_existing_finite_tolerances_are_unchanged(self):
        self.assertTrue(_nearly_equal(0.0, 5e-13))
        self.assertTrue(_nearly_equal(1e6, 1e6 + 5e-7))
        self.assertFalse(_nearly_equal(0.0, 1e-6))
        self.assertFalse(_nearly_equal(1e6, 1e6 + 1.0))


if __name__ == "__main__":
    unittest.main()
