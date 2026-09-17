import math
import unittest

from kvlab.k9_rope import (
    RopeLayout,
    RopeNormalizationSpec,
    apply_rope_reference,
    normalization_record,
    remove_rope,
)


class K9RopeNormalizationTests(unittest.TestCase):
    def test_interleaved_known_angle_and_inverse(self):
        spec = RopeNormalizationSpec(theta_base=10_000.0, rotary_dim=2, layout=RopeLayout.INTERLEAVED_PAIRS)
        rotated = apply_rope_reference((1.0, 0.0), position=1, spec=spec)
        self.assertAlmostEqual(rotated[0], math.cos(1.0), places=14)
        self.assertAlmostEqual(rotated[1], math.sin(1.0), places=14)
        restored = remove_rope(rotated, position=1, spec=spec)
        self.assertAlmostEqual(restored[0], 1.0, places=14)
        self.assertAlmostEqual(restored[1], 0.0, places=14)

    def test_half_split_pairs_first_and_second_halves(self):
        spec = RopeNormalizationSpec(theta_base=100.0, rotary_dim=4, layout=RopeLayout.HALF_SPLIT)
        source = (1.0, 2.0, 3.0, 4.0, 99.0)
        rotated = apply_rope_reference(source, position=2, spec=spec)
        # Pair 0 is dimensions 0 and 2 with angle 2 radians.
        self.assertAlmostEqual(rotated[0], math.cos(2.0) - 3.0 * math.sin(2.0), places=14)
        self.assertAlmostEqual(rotated[2], math.sin(2.0) + 3.0 * math.cos(2.0), places=14)
        # Pair 1 is dimensions 1 and 3 with frequency 100^(-1/2)=0.1.
        self.assertAlmostEqual(rotated[1], 2.0 * math.cos(0.2) - 4.0 * math.sin(0.2), places=14)
        self.assertAlmostEqual(rotated[3], 2.0 * math.sin(0.2) + 4.0 * math.cos(0.2), places=14)
        self.assertEqual(rotated[4], 99.0)
        restored = remove_rope(rotated, position=2, spec=spec)
        for actual, expected in zip(restored, source, strict=True):
            self.assertAlmostEqual(actual, expected, places=13)

    def test_interleaved_round_trip_with_unrotated_tail(self):
        spec = RopeNormalizationSpec(theta_base=10_000.0, rotary_dim=4, layout=RopeLayout.INTERLEAVED_PAIRS)
        source = (0.25, -0.5, 1.5, 2.0, -7.0, 8.0)
        restored = remove_rope(apply_rope_reference(source, position=123, spec=spec), position=123, spec=spec)
        for actual, expected in zip(restored, source, strict=True):
            self.assertAlmostEqual(actual, expected, places=12)
        self.assertEqual(restored[4:], source[4:])

    def test_position_zero_is_identity(self):
        for layout in RopeLayout:
            spec = RopeNormalizationSpec(theta_base=10_000.0, rotary_dim=4, layout=layout)
            source = (1.0, 2.0, 3.0, 4.0)
            self.assertEqual(remove_rope(source, position=0, spec=spec), source)

    def test_record_is_deterministic_and_binds_position_and_spec(self):
        spec = RopeNormalizationSpec(theta_base=10_000.0, rotary_dim=4, layout=RopeLayout.HALF_SPLIT)
        kwargs = dict(sample_id="cal-0001", layer=7, head=2, position=17, spec=spec)
        normalized_a, record_a = normalization_record((1.0, 2.0, 3.0, 4.0), **kwargs)
        normalized_b, record_b = normalization_record((1.0, 2.0, 3.0, 4.0), **kwargs)
        self.assertEqual(normalized_a, normalized_b)
        self.assertEqual(record_a, record_b)
        self.assertEqual(record_a.spec_fingerprint, spec.fingerprint())
        _, later = normalization_record((1.0, 2.0, 3.0, 4.0), **{**kwargs, "position": 18})
        self.assertNotEqual(record_a.output_f64_sha256, later.output_f64_sha256)

    def test_validation_fails_closed(self):
        valid = RopeNormalizationSpec(theta_base=10_000.0, rotary_dim=4, layout=RopeLayout.INTERLEAVED_PAIRS)
        for theta in (1.0, 0.0, -1.0, float("nan"), float("inf"), True, "10000"):
            with self.subTest(theta=theta):
                with self.assertRaises(ValueError):
                    RopeNormalizationSpec(theta_base=theta, rotary_dim=4, layout=RopeLayout.INTERLEAVED_PAIRS)
        for dim in (0, -2, 3, True, 4.0):
            with self.subTest(dim=dim):
                with self.assertRaises(ValueError):
                    RopeNormalizationSpec(theta_base=10_000.0, rotary_dim=dim, layout=RopeLayout.INTERLEAVED_PAIRS)
        with self.assertRaises(ValueError):
            RopeNormalizationSpec(theta_base=10_000.0, rotary_dim=4, layout="half-split-v1")
        for position in (-1, True, 1.5):
            with self.subTest(position=position):
                with self.assertRaises(ValueError):
                    remove_rope((1.0, 2.0, 3.0, 4.0), position=position, spec=valid)
        for vector in ((1.0, 2.0), (1.0, 2.0, float("nan"), 4.0), (1.0, 2.0, True, 4.0)):
            with self.subTest(vector=vector):
                with self.assertRaises(ValueError):
                    remove_rope(vector, position=1, spec=valid)


if __name__ == "__main__":
    unittest.main()
