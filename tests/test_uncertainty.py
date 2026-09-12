import math
import unittest

from kvlab.uncertainty import summarize_normal_interval


class UncertaintySummaryTests(unittest.TestCase):
    def test_constant_samples_have_zero_width_interval(self) -> None:
        result = summarize_normal_interval([4.0, 4.0, 4.0, 4.0])
        self.assertEqual(result.count, 4)
        self.assertEqual(result.mean, 4.0)
        self.assertEqual(result.sample_stddev, 0.0)
        self.assertEqual(result.standard_error, 0.0)
        self.assertEqual(result.interval_low, 4.0)
        self.assertEqual(result.interval_high, 4.0)

    def test_known_samples_use_sample_standard_deviation(self) -> None:
        result = summarize_normal_interval([1.0, 2.0, 3.0, 4.0], z_value=2.0)
        self.assertAlmostEqual(result.mean, 2.5)
        self.assertAlmostEqual(result.sample_stddev, math.sqrt(5.0 / 3.0))
        self.assertAlmostEqual(result.standard_error, math.sqrt(5.0 / 3.0) / 2.0)
        margin = 2.0 * result.standard_error
        self.assertAlmostEqual(result.interval_low, 2.5 - margin)
        self.assertAlmostEqual(result.interval_high, 2.5 + margin)

    def test_custom_confidence_metadata_is_preserved(self) -> None:
        result = summarize_normal_interval(
            (10.0, 12.0, 14.0), confidence_level=0.90, z_value=1.6448536269514722
        )
        self.assertEqual(result.confidence_level, 0.90)
        self.assertAlmostEqual(result.z_value, 1.6448536269514722)

    def test_single_sample_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            summarize_normal_interval([1.0])

    def test_non_finite_measurements_are_rejected(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                summarize_normal_interval([1.0, value])

    def test_invalid_interval_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            summarize_normal_interval([1.0, 2.0], confidence_level=1.0)
        with self.assertRaises(ValueError):
            summarize_normal_interval([1.0, 2.0], z_value=0.0)


if __name__ == "__main__":
    unittest.main()
