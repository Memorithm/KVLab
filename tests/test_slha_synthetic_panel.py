import unittest

from kvlab.slha_synthetic_panel import CASES, run_panel


class SlhaSyntheticPanelTests(unittest.TestCase):
    def records(self):
        return {(r.query_id, r.candidate_id): r for r in run_panel()}

    def test_panel_shape_is_frozen(self) -> None:
        records = run_panel()
        self.assertEqual(len(CASES), 6)
        self.assertEqual(len(records), 24)

    def test_exact_topk_is_perfect_for_every_case(self) -> None:
        for record in run_panel():
            if record.candidate_id != "exact-top-k":
                continue
            self.assertEqual(record.selection.top_k_recall, 1.0)
            self.assertEqual(record.selection.top_k_missed_softmax_mass, 0.0)

    def test_same_topk_recall_can_have_very_different_mass_loss(self) -> None:
        records = self.records()
        dominant = records[("dominant-head", "drop-top1-add-boundary")].selection
        near_tie = records[("near-tie-boundary", "drop-top1-add-boundary")].selection

        self.assertEqual(dominant.top_k_recall, 0.5)
        self.assertEqual(near_tie.top_k_recall, 0.5)
        self.assertGreater(
            dominant.top_k_missed_softmax_mass,
            near_tie.top_k_missed_softmax_mass * 20.0,
        )

    def test_missing_top1_is_worse_than_missing_boundary_on_dominant_case(self) -> None:
        records = self.records()
        top1 = records[("dominant-head", "drop-top1-add-boundary")].selection
        boundary = records[("dominant-head", "drop-boundary-add-next")].selection

        self.assertEqual(top1.top_k_recall, boundary.top_k_recall)
        self.assertGreater(
            top1.top_k_missed_softmax_mass,
            boundary.top_k_missed_softmax_mass,
        )
        self.assertLess(
            top1.retained_softmax_mass,
            boundary.retained_softmax_mass,
        )

    def test_tail_control_succeeds_only_when_relevance_is_late(self) -> None:
        records = self.records()
        early = records[("dominant-head", "tail-matched-density")].selection
        late = records[("late-relevance", "tail-matched-density")].selection

        self.assertEqual(early.candidate_density, late.candidate_density)
        self.assertEqual(early.top_k_recall, 0.0)
        self.assertEqual(late.top_k_recall, 1.0)
        self.assertGreater(late.retained_softmax_mass, early.retained_softmax_mass)


if __name__ == "__main__":
    unittest.main()
