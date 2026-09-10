"""Leak-safe online baselines for temporal C1 calibration.

These policies may inspect only information available at or before the decision
step. Future contributions remain reserved for offline evaluation/oracles.
"""

from __future__ import annotations

from math import sqrt

from .temporal import TemporalKvTrace, TemporalSelection


def _l2(values: tuple[float, ...]) -> float:
    return sqrt(sum(value * value for value in values))


def select_immediate_sensitivity_per_byte(
    trace: TemporalKvTrace,
    decision_step: int,
    budget_bytes: int,
) -> TemporalSelection:
    """Select regions by current-step removal sensitivity per stored byte.

    The policy is intentionally myopic: it reads only each region's contribution
    at ``decision_step``. This makes it a competent leak-safe baseline against
    which history-based or learned policies can be compared under an identical
    byte budget.
    """

    if decision_step < 0 or decision_step >= trace.steps:
        raise IndexError("decision_step outside trace")
    full_bytes = sum(region.storage_bytes for region in trace.regions)
    if budget_bytes < 0 or budget_bytes > full_bytes:
        raise ValueError("budget_bytes must be between zero and full-cache bytes")

    scored = [
        (
            -(_l2(region.contributions[decision_step]) / region.storage_bytes),
            region.region_id,
            region,
        )
        for region in trace.regions
    ]
    scored.sort(key=lambda item: (item[0], item[1]))

    retained: list[str] = []
    retained_bytes = 0
    for _, _, region in scored:
        if retained_bytes + region.storage_bytes <= budget_bytes:
            retained.append(region.region_id)
            retained_bytes += region.storage_bytes

    return TemporalSelection(
        policy_name="immediate_sensitivity_per_byte",
        decision_step=decision_step,
        budget_bytes=budget_bytes,
        retained_region_ids=tuple(retained),
        retained_bytes=retained_bytes,
    )
