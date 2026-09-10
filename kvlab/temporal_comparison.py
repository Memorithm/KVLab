"""Budget-matched comparison records for temporal C1 calibration.

This module keeps deployable selections separate from the offline future oracle.
It only evaluates selections that were already produced without granting any
policy access to post-decision information.
"""

from __future__ import annotations

from dataclasses import dataclass

from .temporal import TemporalKvTrace, TemporalSelection, evaluate_temporal_regret


@dataclass(frozen=True)
class TemporalPolicyResult:
    """One policy's offline future-utility score under a shared byte budget."""

    policy_name: str
    decision_step: int
    budget_bytes: int
    retained_region_ids: tuple[str, ...]
    retained_bytes: int
    policy_future_utility: float
    oracle_future_utility: float
    regret: float


def compare_temporal_selections(
    trace: TemporalKvTrace,
    selections: tuple[TemporalSelection, ...],
) -> tuple[TemporalPolicyResult, ...]:
    """Evaluate several already-made selections against one offline oracle.

    All selections must use the same decision step and byte budget. Policy names
    must be unique so downstream tables cannot silently overwrite evidence.
    The exact future oracle is used only after each selection has been made.
    """

    if not selections:
        raise ValueError("selections must be non-empty")

    decision_step = selections[0].decision_step
    budget_bytes = selections[0].budget_bytes
    names = [selection.policy_name for selection in selections]
    if len(names) != len(set(names)):
        raise ValueError("policy_name values must be unique")
    if any(selection.decision_step != decision_step for selection in selections):
        raise ValueError("all selections must use the same decision_step")
    if any(selection.budget_bytes != budget_bytes for selection in selections):
        raise ValueError("all selections must use the same budget_bytes")

    results = []
    for selection in selections:
        regret = evaluate_temporal_regret(trace, selection)
        results.append(
            TemporalPolicyResult(
                policy_name=selection.policy_name,
                decision_step=selection.decision_step,
                budget_bytes=selection.budget_bytes,
                retained_region_ids=selection.retained_region_ids,
                retained_bytes=selection.retained_bytes,
                policy_future_utility=regret.policy_future_utility,
                oracle_future_utility=regret.oracle_future_utility,
                regret=regret.regret,
            )
        )
    return tuple(results)
