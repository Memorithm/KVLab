"""Deterministic C1 evaluation of budget selections against the synthetic oracle.

This module scores metadata-only selections produced by calibration baselines. It
is intentionally limited to the additive synthetic world: no FLAT-ATTENTION
cache semantics, model attention, or NNIS placement are inferred here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from .baselines import BudgetSelection
from .synthetic_trace import SyntheticKvTrace


@dataclass(frozen=True)
class SelectionEvaluation:
    """Exact synthetic-oracle result for one byte-budget selection."""

    trace_id: str
    policy: str
    budget_bytes: int
    retained_region_ids: tuple[str, ...]
    retained_bytes: int
    unused_bytes: int
    full_cache_output: tuple[float, ...]
    selected_output: tuple[float, ...]
    output_l2_delta: float


def evaluate_selection(
    trace: SyntheticKvTrace,
    selection: BudgetSelection,
) -> SelectionEvaluation:
    """Compare a validated selection with the deterministic full-cache output."""

    if selection.budget_bytes < 0 or selection.budget_bytes > trace.total_storage_bytes:
        raise ValueError("selection budget is outside the trace storage range")
    if len(selection.retained_region_ids) != len(set(selection.retained_region_ids)):
        raise ValueError("retained_region_ids must be unique")

    by_id = {region.region_id: region for region in trace.regions}
    try:
        retained_regions = tuple(by_id[region_id] for region_id in selection.retained_region_ids)
    except KeyError as error:
        raise ValueError(f"selection references unknown region_id: {error.args[0]}") from error

    retained_bytes = sum(region.storage_bytes for region in retained_regions)
    if retained_bytes != selection.retained_bytes:
        raise ValueError("selection retained_bytes does not match retained regions")
    if retained_bytes > selection.budget_bytes:
        raise ValueError("selection exceeds its byte budget")

    full = trace.full_cache_output()
    width = len(full)
    selected = tuple(
        sum(region.contribution[index] for region in retained_regions)
        for index in range(width)
    )
    delta = sqrt(sum((left - right) ** 2 for left, right in zip(full, selected, strict=True)))

    return SelectionEvaluation(
        trace_id=trace.trace_id,
        policy=selection.policy,
        budget_bytes=selection.budget_bytes,
        retained_region_ids=selection.retained_region_ids,
        retained_bytes=retained_bytes,
        unused_bytes=selection.budget_bytes - retained_bytes,
        full_cache_output=full,
        selected_output=selected,
        output_l2_delta=delta,
    )
