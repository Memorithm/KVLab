import math
import unittest

from kvlab.slha_quality import (
    SlhaQualityError,
    evaluate_ranking,
    evaluate_selection,
    softmax_probabilities,
    stable_rank,
)


class SlhaQualityTests(unittest.TestCase):
    def test_stable_rank_uses_item_id_for_ties(self) -> None:
        self.assertEqual(stable_rank((1.0, 2.0, 2.0, 0.0)), (1, 2, 0, 3))

    def test_softmax_is_stable_for_large_scores(self) -> None:
        probabilities = softmax_probabilities((1000.0, 999.0, 998.0))
        self.assertTrue(all(math.isfinite(value) for value in probabilities))
        self.assertAlmostEqual(sum(probabilities), 1.0, places=15)
        self.assertGreater(probabilities[0], probabilities[1])

    def test_selection_reports_mass_and_topk_separately(self) -> None:
        result = evaluate_selection((8.0, 7.0, 0.0, 0.0), (0, 2), top_k=2)
        self.assertEqual(result.top_k_hits, 1)
        self.assertEqual(result.top_k_recall, 0.5)
        self.assertEqual(result.candidate_density, 0.5)
        self.assertGreater(result.retained_softmax_mass, 0.5)
        self.assertGreater(result.top_k_missed_softmax_mass, 0.0)
        self.assertAlmostEqual(
            result.retained_softmax_mass + result.omitted_softmax_mass,
            1.0,
            places=15,
        )

    def test_topk_count_can_hide_large_mass_difference(self) -> None:
        scores = (10.0, 9.0, 1.0, 0.0)
        keep_heavy = evaluate_selection(scores, (0, 2), top_k=2)
        keep_light = evaluate_selection(scores, (1, 2), top_k=2)

        self.assertEqual(keep_heavy.top_k_recall, keep_light.top_k_recall)
        self.assertGreater(
            keep_heavy.retained_softmax_mass,
            keep_light.retained_softmax_mass,
        )

    def test_boundary_pair_accuracy_detects_wrong_order(self) -> None:
        reference = (4.0, 3.0, 2.0, 1.0)
        candidate = (4.0, 0.0, 3.0, 2.0)
        result = evaluate_ranking(reference, candidate, top_k=2)

        self.assertEqual(result.top_k_overlap_count, 1)
        self.assertLess(result.boundary_pair_accuracy, 1.0)

    def test_identical_ranking_is_perfect(self) -> None:
        scores = (0.1, 3.0, 2.0, -1.0)
        result = evaluate_ranking(scores, scores, top_k=2)

        self.assertEqual(result.top_k_recall, 1.0)
        self.assertEqual(result.boundary_pair_accuracy, 1.0)

    def test_rejects_malformed_inputs(self) -> None:
        with self.assertRaises(SlhaQualityError):
            evaluate_selection((1.0, float("nan")), (0,), top_k=1)
        with self.assertRaises(SlhaQualityError):
            evaluate_selection((1.0, 0.0), (1, 1), top_k=1)
        with self.assertRaises(SlhaQualityError):
            evaluate_ranking((1.0,), (1.0, 0.0), top_k=1)


if __name__ == "__main__":
    unittest.main()
