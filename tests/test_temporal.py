import unittest

from kvlab.temporal import (
    TemporalKvRegion,
    TemporalKvTrace,
    TemporalSelection,
    evaluate_temporal_regret,
    evaluate_temporal_utility,
    select_exact_future_oracle,
    select_observed_history_per_byte,
)


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

    def test_online_policy_can_incur_regret_without_future_leakage(self) -> None:
        trace = TemporalKvTrace(
            "online-vs-future",
            (
                TemporalKvRegion("visible-now", 4, ((8.0,), (0.0,), (0.0,))),
                TemporalKvRegion("useful-later", 4, ((1.0,), (6.0,), (6.0,))),
            ),
        )

        online = select_observed_history_per_byte(trace, decision_step=0, budget_bytes=4)
        oracle = select_exact_future_oracle(trace, decision_step=0, budget_bytes=4)
        regret = evaluate_temporal_regret(trace, online)

        self.assertEqual(online.retained_region_ids, ("visible-now",))
        self.assertEqual(oracle.retained_region_ids, ("useful-later",))
        self.assertEqual(regret.policy_future_utility, 0.0)
        self.assertEqual(regret.oracle_future_utility, 12.0)
        self.assertEqual(regret.regret, 12.0)

    def test_online_policy_ignores_changes_strictly_after_decision(self) -> None:
        before = TemporalKvTrace(
            "before",
            (
                TemporalKvRegion("a", 2, ((3.0,), (0.0,))),
                TemporalKvRegion("b", 2, ((1.0,), (1.0,))),
            ),
        )
        altered_future = TemporalKvTrace(
            "after",
            (
                TemporalKvRegion("a", 2, ((3.0,), (1000.0,))),
                TemporalKvRegion("b", 2, ((1.0,), (0.0,))),
            ),
        )

        first = select_observed_history_per_byte(before, decision_step=0, budget_bytes=2)
        second = select_observed_history_per_byte(altered_future, decision_step=0, budget_bytes=2)
        self.assertEqual(first.retained_region_ids, second.retained_region_ids)
        self.assertEqual(first.retained_region_ids, ("a",))

    def test_regret_rejects_inconsistent_byte_accounting(self) -> None:
        trace = TemporalKvTrace(
            "accounting",
            (TemporalKvRegion("r", 4, ((1.0,), (1.0,))),),
        )
        bad = TemporalSelection("bad", 0, 4, ("r",), 3)
        with self.assertRaises(ValueError):
            evaluate_temporal_regret(trace, bad)

    def test_exact_oracle_caps_exponential_calibration(self) -> None:
        trace = TemporalKvTrace(
            "too-large",
            tuple(TemporalKvRegion(f"r{i}", 1, ((1.0,), (1.0,))) for i in range(21)),
        )
        with self.assertRaises(ValueError):
            select_exact_future_oracle(trace, decision_step=0, budget_bytes=1)


if __name__ == "__main__":
    unittest.main()
