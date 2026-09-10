"""Common deterministic comparison table for C1 calibration policies.

This module compares already-constructed :class:`BudgetSelection` values against
one :class:`SyntheticKvTrace` using the exact synthetic evaluator. It enforces a
single byte budget across the compared policies so an apparent advantage cannot
come from unequal storage allowance.

The result is calibration infrastructure only. It does not estimate future KV
recoverability, model quality, attention importance, or hardware cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .baselines import BudgetSelection
from .evaluation import SelectionEvaluation, evaluate_selection
from .synthetic_trace import SyntheticKvTrace


@dataclass(frozen=True)
class BudgetComparison:
    """One budget-matched deterministic comparison over synthetic policies."""

    trace_id: str
    budget_bytes: int
    evaluations: tuple[SelectionEvaluation, ...]

    def by_policy(self) -> dict[str, SelectionEvaluation]:
        """Return evaluations keyed by unique policy name."""

        return {evaluation.policy: evaluation for evaluation in self.evaluations}


def compare_budget_selections(
    trace: SyntheticKvTrace,
    selections: Iterable[BudgetSelection],
) -> BudgetComparison:
    """Evaluate distinct policies under one exact byte budget.

    Policy names must be unique because downstream tables use them as stable
    identifiers. At least one selection is required. Every selection must carry
    exactly the same byte budget; the underlying evaluator then independently
    validates retained region identities and byte accounting.
    """

    materialized = tuple(selections)
    if not materialized:
        raise ValueError("at least one selection is required")

    budget_bytes = materialized[0].budget_bytes
    policy_names = tuple(selection.policy for selection in materialized)
    if len(policy_names) != len(set(policy_names)):
        raise ValueError("policy names must be unique within one comparison")
    if any(selection.budget_bytes != budget_bytes for selection in materialized):
        raise ValueError("all compared selections must use the same byte budget")

    evaluations = tuple(evaluate_selection(trace, selection) for selection in materialized)
    return BudgetComparison(
        trace_id=trace.trace_id,
        budget_bytes=budget_bytes,
        evaluations=evaluations,
    )
