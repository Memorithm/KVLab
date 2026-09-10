import unittest

from kvlab.leakage import audit_future_suffix_invariance
from kvlab.temporal import TemporalKvRegion, TemporalKvTrace, TemporalSelection, select_observed_history_per_byte


def _trace(trace_id: str, future_a: float, future_b: float) -> TemporalKvTrace:
    return TemporalKvTrace(
        trace_id=trace_id,
        regions=(
            TemporalKvRegion("a", 4, ((3.0, 0.0), (future_a, 0.0))),
            TemporalKvRegion("b", 4, ((1.0, 0.0), (future_b, 0.0))),
        ),
    )


class FutureLeakageAuditTests(unittest.TestCase):
    def test_observed_history_policy_ignores_future_suffix(self) -> None:
        audit = audit_future_suffix_invariance(
            select_observed_history_per_byte,
            _trace("reference", 0.0, 100.0),
            _trace("counterfactual", 100.0, 0.0),
            decision_step=0,
            budget_bytes=4,
        )
        self.assertTrue(audit.passed)
        self.assertEqual(audit.reference_region_ids, ("a",))

    def test_prefix_mismatch_is_rejected_before_policy_execution(self) -> None:
        reference = _trace("reference", 0.0, 100.0)
        counterfactual = TemporalKvTrace(
            trace_id="counterfactual",
            regions=(
                TemporalKvRegion("a", 4, ((2.0, 0.0), (100.0, 0.0))),
                TemporalKvRegion("b", 4, ((1.0, 0.0), (0.0, 0.0))),
            ),
        )
        with self.assertRaisesRegex(ValueError, "identical through decision_step"):
            audit_future_suffix_invariance(
                select_observed_history_per_byte,
                reference,
                counterfactual,
                decision_step=0,
                budget_bytes=4,
            )

    def test_future_reading_policy_is_detected(self) -> None:
        def leaking_policy(trace: TemporalKvTrace, decision_step: int, budget_bytes: int) -> TemporalSelection:
            best = max(trace.regions, key=lambda region: region.contributions[-1][0])
            return TemporalSelection(
                policy_name="leaking_fixture",
                decision_step=decision_step,
                budget_bytes=budget_bytes,
                retained_region_ids=(best.region_id,),
                retained_bytes=best.storage_bytes,
            )

        audit = audit_future_suffix_invariance(
            leaking_policy,
            _trace("reference", 0.0, 100.0),
            _trace("counterfactual", 100.0, 0.0),
            decision_step=0,
            budget_bytes=4,
        )
        self.assertFalse(audit.passed)
        self.assertNotEqual(audit.reference_region_ids, audit.counterfactual_region_ids)

    def test_policy_metadata_drift_is_rejected(self) -> None:
        def drifting_policy(trace: TemporalKvTrace, decision_step: int, budget_bytes: int) -> TemporalSelection:
            return TemporalSelection(
                policy_name=trace.trace_id,
                decision_step=decision_step,
                budget_bytes=budget_bytes,
                retained_region_ids=(),
                retained_bytes=0,
            )

        with self.assertRaisesRegex(ValueError, "policy identity changed"):
            audit_future_suffix_invariance(
                drifting_policy,
                _trace("reference", 0.0, 100.0),
                _trace("counterfactual", 100.0, 0.0),
                decision_step=0,
                budget_bytes=4,
            )


if __name__ == "__main__":
    unittest.main()
