"""Mass-aware evidence binding for Boolean-indexed numerical KV (BIKV).

This module binds an already-canonical ProspectBkvHandoffV1 survivor set to one
explicit page-level reference-score oracle. The score semantics are caller
declared and evaluation-only; the Boolean router does not gain access to dense
scores by using this evidence type.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any

from .prospect_handoff import ProspectBkvHandoffV1
from .slha_quality import SelectionQuality, SlhaQualityError, evaluate_selection


BIKV_MASS_EVIDENCE_SCHEMA_V1 = "kvlab.bikv-mass-evidence/v1"


class BikvMassEvidenceError(ValueError):
    """Raised when BIKV mass evidence is malformed or inconsistent."""


def _nonempty_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise BikvMassEvidenceError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class BikvMassEvidenceV1:
    schema: str
    experiment_id: str
    query_id: str
    score_semantics: str
    evaluation_oracle_only: bool
    handoff_json: str
    reference_page_scores: tuple[float, ...]
    top_k: int
    selection: SelectionQuality

    @classmethod
    def capture(
        cls,
        *,
        experiment_id: str,
        query_id: str,
        score_semantics: str,
        handoff: ProspectBkvHandoffV1,
        reference_page_scores: tuple[float, ...] | list[float],
        top_k: int,
    ) -> "BikvMassEvidenceV1":
        _nonempty_text("experiment_id", experiment_id)
        _nonempty_text("query_id", query_id)
        _nonempty_text("score_semantics", score_semantics)
        if not isinstance(handoff, ProspectBkvHandoffV1):
            raise BikvMassEvidenceError("handoff must be ProspectBkvHandoffV1")

        if len(reference_page_scores) != len(handoff.page_words):
            raise BikvMassEvidenceError(
                "reference_page_scores length must match the handoff page universe"
            )
        try:
            selection = evaluate_selection(
                reference_page_scores,
                handoff.admitted_pages,
                top_k=top_k,
            )
        except SlhaQualityError as error:
            raise BikvMassEvidenceError(str(error)) from error
        scores = tuple(float(value) for value in reference_page_scores)

        return cls(
            schema=BIKV_MASS_EVIDENCE_SCHEMA_V1,
            experiment_id=experiment_id,
            query_id=query_id,
            score_semantics=score_semantics,
            evaluation_oracle_only=True,
            handoff_json=handoff.canonical_json(),
            reference_page_scores=scores,
            top_k=top_k,
            selection=selection,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "query_id": self.query_id,
            "score_semantics": self.score_semantics,
            "evaluation_oracle_only": self.evaluation_oracle_only,
            "handoff_json": self.handoff_json,
            "reference_page_scores": list(self.reference_page_scores),
            "top_k": self.top_k,
            "selection": dataclasses.asdict(self.selection),
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_canonical_json(cls, payload: str) -> "BikvMassEvidenceV1":
        if not isinstance(payload, str):
            raise BikvMassEvidenceError("payload must be text")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as error:
            raise BikvMassEvidenceError("payload is not valid JSON") from error
        if not isinstance(decoded, dict):
            raise BikvMassEvidenceError("payload must decode to an object")
        expected = {
            "schema",
            "experiment_id",
            "query_id",
            "score_semantics",
            "evaluation_oracle_only",
            "handoff_json",
            "reference_page_scores",
            "top_k",
            "selection",
        }
        if set(decoded) != expected:
            raise BikvMassEvidenceError("payload fields do not match schema v1")
        if decoded["schema"] != BIKV_MASS_EVIDENCE_SCHEMA_V1:
            raise BikvMassEvidenceError("unsupported BIKV mass evidence schema")
        if decoded["evaluation_oracle_only"] is not True:
            raise BikvMassEvidenceError("dense page scores must remain evaluation-oracle-only")
        if not isinstance(decoded["reference_page_scores"], list):
            raise BikvMassEvidenceError("reference_page_scores must be a JSON array")
        if isinstance(decoded["top_k"], bool) or not isinstance(decoded["top_k"], int):
            raise BikvMassEvidenceError("top_k must be an integer")

        try:
            handoff = ProspectBkvHandoffV1.from_canonical_json(decoded["handoff_json"])
        except Exception as error:
            raise BikvMassEvidenceError(f"invalid embedded BIKV handoff: {error}") from error

        rebuilt = cls.capture(
            experiment_id=_nonempty_text("experiment_id", decoded["experiment_id"]),
            query_id=_nonempty_text("query_id", decoded["query_id"]),
            score_semantics=_nonempty_text("score_semantics", decoded["score_semantics"]),
            handoff=handoff,
            reference_page_scores=tuple(decoded["reference_page_scores"]),
            top_k=decoded["top_k"],
        )
        if payload != rebuilt.canonical_json():
            raise BikvMassEvidenceError(
                "payload is not canonical or contains inconsistent derived metrics"
            )
        return rebuilt
