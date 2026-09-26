"""Canonical per-query SLHA quality evidence for KVLab.

Version 1 intentionally embeds one bounded score row so decoding can recompute
all mechanistic metrics independently. Large real-model campaigns may later
introduce a chunked/blob-bound schema; they must not weaken validation.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from typing import Any

from .slha_quality import (
    RankingQuality,
    SelectionQuality,
    SlhaQualityError,
    evaluate_ranking,
    evaluate_selection,
)


SLHA_QUALITY_EVIDENCE_SCHEMA_V1 = "kvlab.slha-quality-evidence/v1"


class SlhaEvidenceError(ValueError):
    """Raised when canonical SLHA quality evidence is malformed."""


def _plain_int(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SlhaEvidenceError(f"{name} must be an integer >= {minimum}")
    return value


def _plain_str(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise SlhaEvidenceError(f"{name} must be a non-empty string")
    return value


def _float_tuple(name: str, value: object) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise SlhaEvidenceError(f"{name} must be a non-empty JSON array")
    out: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise SlhaEvidenceError(f"{name} must contain numeric values")
        number = float(item)
        if not math.isfinite(number):
            raise SlhaEvidenceError(f"{name} must contain finite values")
        out.append(number)
    return tuple(out)


def _int_tuple(name: str, value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise SlhaEvidenceError(f"{name} must be a JSON array")
    out = tuple(value)
    for item in out:
        _plain_int(f"{name} item", item)
    return out


@dataclass(frozen=True, slots=True)
class SlhaQualityEvidenceV1:
    schema: str
    experiment_id: str
    query_id: str
    candidate_id: str
    top_k: int
    reference_scores: tuple[float, ...]
    selected_ids: tuple[int, ...]
    candidate_scores: tuple[float, ...] | None
    selection: SelectionQuality
    ranking: RankingQuality | None

    @classmethod
    def capture(
        cls,
        *,
        experiment_id: str,
        query_id: str,
        candidate_id: str,
        top_k: int,
        reference_scores: tuple[float, ...] | list[float],
        selected_ids: tuple[int, ...] | list[int],
        candidate_scores: tuple[float, ...] | list[float] | None = None,
    ) -> "SlhaQualityEvidenceV1":
        _plain_str("experiment_id", experiment_id)
        _plain_str("query_id", query_id)
        _plain_str("candidate_id", candidate_id)

        selected = tuple(selected_ids)
        try:
            selection = evaluate_selection(reference_scores, selected, top_k=top_k)
            ranking = (
                None
                if candidate_scores is None
                else evaluate_ranking(reference_scores, candidate_scores, top_k=top_k)
            )
        except SlhaQualityError as error:
            raise SlhaEvidenceError(str(error)) from error
        reference = tuple(float(value) for value in reference_scores)
        candidate = (
            None
            if candidate_scores is None
            else tuple(float(value) for value in candidate_scores)
        )

        return cls(
            schema=SLHA_QUALITY_EVIDENCE_SCHEMA_V1,
            experiment_id=experiment_id,
            query_id=query_id,
            candidate_id=candidate_id,
            top_k=top_k,
            reference_scores=reference,
            selected_ids=selected,
            candidate_scores=candidate,
            selection=selection,
            ranking=ranking,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "query_id": self.query_id,
            "candidate_id": self.candidate_id,
            "top_k": self.top_k,
            "reference_scores": list(self.reference_scores),
            "selected_ids": list(self.selected_ids),
            "candidate_scores": (
                None if self.candidate_scores is None else list(self.candidate_scores)
            ),
            "selection": dataclasses.asdict(self.selection),
            "ranking": None if self.ranking is None else dataclasses.asdict(self.ranking),
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_canonical_json(cls, payload: str) -> "SlhaQualityEvidenceV1":
        if not isinstance(payload, str):
            raise SlhaEvidenceError("payload must be text")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as error:
            raise SlhaEvidenceError("payload is not valid JSON") from error
        if not isinstance(decoded, dict):
            raise SlhaEvidenceError("payload must decode to an object")
        expected = {
            "schema",
            "experiment_id",
            "query_id",
            "candidate_id",
            "top_k",
            "reference_scores",
            "selected_ids",
            "candidate_scores",
            "selection",
            "ranking",
        }
        if set(decoded) != expected:
            raise SlhaEvidenceError("payload fields do not match schema v1")
        if decoded["schema"] != SLHA_QUALITY_EVIDENCE_SCHEMA_V1:
            raise SlhaEvidenceError("unsupported SLHA quality evidence schema")

        top_k = _plain_int("top_k", decoded["top_k"], minimum=1)
        reference = _float_tuple("reference_scores", decoded["reference_scores"])
        selected = _int_tuple("selected_ids", decoded["selected_ids"])
        raw_candidate = decoded["candidate_scores"]
        candidate = (
            None
            if raw_candidate is None
            else _float_tuple("candidate_scores", raw_candidate)
        )

        rebuilt = cls.capture(
            experiment_id=_plain_str("experiment_id", decoded["experiment_id"]),
            query_id=_plain_str("query_id", decoded["query_id"]),
            candidate_id=_plain_str("candidate_id", decoded["candidate_id"]),
            top_k=top_k,
            reference_scores=reference,
            selected_ids=selected,
            candidate_scores=candidate,
        )
        canonical = rebuilt.canonical_json()
        if payload != canonical:
            raise SlhaEvidenceError(
                "payload is not canonical or contains inconsistent derived metrics"
            )
        return rebuilt
