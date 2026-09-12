import unittest

from kvlab.k9_ridge import fit_ridge_calibration


class K9RidgeTests(unittest.TestCase):
    def test_exact_linear_map_with_zero_alpha(self) -> None:
        fit = fit_ridge_calibration(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[2.0, -1.0], [3.0, 4.0], [5.0, 3.0]],
            alpha=0.0,
            calibration_ids=("c1", "c2", "c3"),
            final_holdout_ids=("h1",),
        )
        self.assertEqual(fit.feature_dim, 2)
        self.assertEqual(fit.target_dim, 2)
        predicted = fit.predict([[2.0, 1.0]])[0]
        self.assertAlmostEqual(predicted[0], 7.0, places=12)
        self.assertAlmostEqual(predicted[1], 2.0, places=12)

    def test_positive_alpha_is_deterministic(self) -> None:
        kwargs = dict(
            features=[[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]],
            targets=[[1.0], [2.0], [3.0]],
            alpha=0.5,
            calibration_ids=("a", "b", "c"),
            final_holdout_ids=("holdout",),
        )
        self.assertEqual(fit_ridge_calibration(**kwargs), fit_ridge_calibration(**kwargs))

    def test_rejects_holdout_leakage(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            fit_ridge_calibration(
                [[1.0], [2.0]],
                [[3.0], [4.0]],
                alpha=1.0,
                calibration_ids=("shared", "c2"),
                final_holdout_ids=("shared",),
            )

    def test_rejects_nonfinite_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_ridge_calibration(
                [[1.0], [float("nan")]],
                [[3.0], [4.0]],
                alpha=1.0,
                calibration_ids=("c1", "c2"),
                final_holdout_ids=("h1",),
            )

    def test_zero_alpha_rejects_singular_normal_equations(self) -> None:
        with self.assertRaisesRegex(ValueError, "singular"):
            fit_ridge_calibration(
                [[1.0, 1.0], [2.0, 2.0]],
                [[1.0], [2.0]],
                alpha=0.0,
                calibration_ids=("c1", "c2"),
                final_holdout_ids=("h1",),
            )


if __name__ == "__main__":
    unittest.main()
