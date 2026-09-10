import unittest

from kvlab.temporal import TemporalKvRegion, TemporalKvTrace, evaluate_temporal_utility


class TemporalUtilityTests(unittest.TestCase):
    def test_immediate_and_future_utility_can_disagree(self) -> None:
        trace = TemporalKvTrace(
            "divergence",
            (
                TemporalKvRegion("now", 4, ((5.0, 0.0), (0.0, 0.0), (0.0, 0.0))),
                TemporalKvRegion("later", 4, ((1.0, 0.0), (4.0, 0.0), (4.0, 0.0))),
            ),
        )

        now = evaluate_temporal_utility(trace, "now", 0)
        later = evaluate_temporal_utility(trace, "later", 0)

        self.assertGreater(now.immediate_l2_delta, later.immediate_l2_delta)
        self.assertLess(now.future_l2_delta_sum, later.future_l2_delta_sum)
        self.assertLess(now.future_utility_per_byte, later.future_utility_per_byte)

    def test_future_score_excludes_decision_step(self) -> None:
        trace = TemporalKvTrace(
            "boundary",
            (TemporalKvRegion("r", 2, ((3.0,), (4.0,), (5.0,))),),
        )

        utility = evaluate_temporal_utility(trace, "r", 1)
        self.assertEqual(utility.immediate_l2_delta, 4.0)
        self.assertEqual(utility.future_l2_delta_sum, 5.0)
        self.assertEqual(utility.future_utility_per_byte, 2.5)

    def test_temporal_lengths_must_match(self) -> None:
        with self.assertRaises(ValueError):
            TemporalKvTrace(
                "bad",
                (
                    TemporalKvRegion("a", 1, ((1.0,),)),
                    TemporalKvRegion("b", 1, ((1.0,), (2.0,))),
                ),
            )

    def test_invalid_decision_step_fails_closed(self) -> None:
        trace = TemporalKvTrace(
            "bounds",
            (TemporalKvRegion("r", 1, ((1.0,),)),),
        )
        with self.assertRaises(IndexError):
            evaluate_temporal_utility(trace, "r", 1)


if __name__ == "__main__":
    unittest.main()
