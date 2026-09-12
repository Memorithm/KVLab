"""Machine-readable experiment outcome records.

The registry vocabulary preserves negative, inconclusive and non-applicable
outcomes as first-class data. It records whether a preregistered decision rule
was met without turning that classification into a broader scientific claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OutcomeClass(str, Enum):
    CRITERION_MET = "criterion_met"
    CRITERION_NOT_MET = "criterion_not_met"
    INCONCLUSIVE = "inconclusive"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    experiment_id: str
    commit_sha: str
    mechanism: str
    outcome: OutcomeClass
    decision_rule: str
    evidence_ref: str
    applicability_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.experiment_id or len(self.experiment_id) > 256:
            raise ValueError("experiment_id must be non-empty and bounded")
        if len(self.commit_sha) != 40 or any(
            char not in "0123456789abcdefABCDEF" for char in self.commit_sha
        ):
            raise ValueError("commit_sha must be a full 40-character Git SHA")
        if not self.mechanism.strip():
            raise ValueError("mechanism must be non-empty")
        if not self.decision_rule.strip():
            raise ValueError("decision_rule must be explicit")
        if not self.evidence_ref.strip():
            raise ValueError("evidence_ref must be explicit")
        if self.outcome is OutcomeClass.NOT_APPLICABLE:
            if not self.applicability_reason or not self.applicability_reason.strip():
                raise ValueError("not_applicable requires an applicability_reason")
        elif self.applicability_reason is not None:
            raise ValueError(
                "applicability_reason is reserved for not_applicable outcomes"
            )


class ResultRegistry:
    """In-memory append-only registry with duplicate experiment protection."""

    def __init__(self) -> None:
        self._records: list[ExperimentResult] = []
        self._ids: set[str] = set()

    def append(self, result: ExperimentResult) -> None:
        if result.experiment_id in self._ids:
            raise ValueError("experiment_id already recorded")
        self._ids.add(result.experiment_id)
        self._records.append(result)

    def records(self) -> tuple[ExperimentResult, ...]:
        return tuple(self._records)

    def by_outcome(self, outcome: OutcomeClass) -> tuple[ExperimentResult, ...]:
        return tuple(record for record in self._records if record.outcome is outcome)
