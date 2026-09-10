from kvlab.temporal import TemporalKvRegion, TemporalKvTrace
from kvlab.temporal_baselines import select_immediate_sensitivity_per_byte


def _trace(future_a: float, future_b: float) -> TemporalKvTrace:
    return TemporalKvTrace(
        trace_id="suffix-variant",
        regions=(
            TemporalKvRegion(
                region_id="a",
                storage_bytes=4,
                contributions=((4.0, 0.0), (future_a, 0.0)),
            ),
            TemporalKvRegion(
                region_id="b",
                storage_bytes=4,
                contributions=((2.0, 0.0), (future_b, 0.0)),
            ),
        ),
    )


def test_immediate_sensitivity_prefers_current_utility_per_byte() -> None:
    selection = select_immediate_sensitivity_per_byte(
        _trace(future_a=0.0, future_b=100.0),
        decision_step=0,
        budget_bytes=4,
    )

    assert selection.policy_name == "immediate_sensitivity_per_byte"
    assert selection.retained_region_ids == ("a",)
    assert selection.retained_bytes == 4


def test_immediate_sensitivity_is_invariant_to_future_suffix() -> None:
    first = select_immediate_sensitivity_per_byte(
        _trace(future_a=0.0, future_b=100.0),
        decision_step=0,
        budget_bytes=4,
    )
    second = select_immediate_sensitivity_per_byte(
        _trace(future_a=100.0, future_b=0.0),
        decision_step=0,
        budget_bytes=4,
    )

    assert first == second


def test_immediate_sensitivity_rejects_invalid_budget_and_step() -> None:
    trace = _trace(future_a=0.0, future_b=0.0)

    try:
        select_immediate_sensitivity_per_byte(trace, decision_step=2, budget_bytes=4)
    except IndexError:
        pass
    else:
        raise AssertionError("out-of-range decision step must fail")

    try:
        select_immediate_sensitivity_per_byte(trace, decision_step=0, budget_bytes=9)
    except ValueError:
        pass
    else:
        raise AssertionError("over-full byte budget must fail")
