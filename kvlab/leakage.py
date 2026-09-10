"""Fail-closed online-policy leakage audits for temporal C1 calibration.

This module checks a necessary non-leakage property for policies evaluated in the
synthetic temporal substrate: changing only post-decision contributions must not
change an online decision. The audit is a deterministic calibration guard, not a
claim about real-model KV behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .temporal import TemporalKvTrace, TemporalSelection

TemporalPolicy = Callable[[TemporalKvTrace, int, int], TemporalSelection]


@dataclass(frozen=True)
class FutureLeakageAudit:
    """Result of comparing one online policy across two equal-prefix traces."""

    policy_name: str
    decision_step: int
    budget_bytes: int
    reference_region_ids: tuple[str, ...]
    counterfactual_region_ids: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.reference_region_ids == self.counterfactual_region_ids


def audit_future_suffix_invariance(
    policy: TemporalPolicy,
    reference: TemporalKvTrace,
    counterfactual: TemporalKvTrace,
    decision_step: int,
    budget_bytes: int,
) -> FutureLeakageAudit:
    """Check that a policy decision is invariant to an unobservable future suffix.

    The two traces must expose exactly the same region identities, storage costs,
    temporal length, and contributions through ``decision_step``. Only later
    contributions may differ. Any malformed comparison fails closed before the
    policy is executed.
    """

    if decision_step < 0 or decision_step >= reference.steps:
        raise IndexError("decision_step outside reference trace")
    if counterfactual.steps != reference.steps:
        raise ValueError("traces must have equal temporal length")

    reference_regions = {region.region_id: region for region in reference.regions}
    counterfactual_regions = {region.region_id: region for region in counterfactual.regions}
    if reference_regions.keys() != counterfactual_regions.keys():
        raise ValueError("traces must contain identical region identities")

    for region_id, reference_region in reference_regions.items():
        counterfactual_region = counterfactual_regions[region_id]
        if reference_region.storage_bytes != counterfactual_region.storage_bytes:
            raise ValueError("region storage costs must match")
        if reference_region.contributions[: decision_step + 1] != counterfactual_region.contributions[: decision_step + 1]:
            raise ValueError("traces must be identical through decision_step")

    reference_selection = policy(reference, decision_step, budget_bytes)
    counterfactual_selection = policy(counterfactual, decision_step, budget_bytes)

    if reference_selection.decision_step != decision_step or counterfactual_selection.decision_step != decision_step:
        raise ValueError("policy returned an inconsistent decision_step")
    if reference_selection.budget_bytes != budget_bytes or counterfactual_selection.budget_bytes != budget_bytes:
        raise ValueError("policy returned an inconsistent byte budget")
    if reference_selection.policy_name != counterfactual_selection.policy_name:
        raise ValueError("policy identity changed across traces")

    return FutureLeakageAudit(
        policy_name=reference_selection.policy_name,
        decision_step=decision_step,
        budget_bytes=budget_bytes,
        reference_region_ids=reference_selection.retained_region_ids,
        counterfactual_region_ids=counterfactual_selection.retained_region_ids,
    )
