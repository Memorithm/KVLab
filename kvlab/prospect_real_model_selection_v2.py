"""Observed real-model evidence for position-identified KV selections.

This v2 evidence envelope embeds `kvlab.prospect-kv-selection/v2`, where
sequence positions are occurrence identities and vocabulary token values may
repeat. It binds exact model/runtime/trace provenance, paired baseline and
candidate artefacts, logical KV byte accounting, and finite observed metrics.

The policy label remains provenance only. Logical retained bytes are not a
claim about allocator release, HBM residency, memory traffic, latency,
throughput, or quality preservation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import re
from typing import Any, Iterable, Mapping

from .prospect_real_model_eviction import ObservedMetric
from .prospect_selection_handoff_v2 import (
    ProspectKvSelectionHandoffV2,
    ProspectKvSelectionHandoffV2Error,
)


PROSPECT_KV_REAL_MODEL_SELECTION_SCHEMA_V2 = (
    "kvlab.prospect-kv-real-model-selection/v2"
)

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ProspectKvRealModelSelectionV2Error(ValueError):
    """Raised when observed position-identified selection evidence is invalid."""


@dataclass(frozen=True, slots=True)
class ProspectKvRealModelSelectionEvidenceV2:
    schema: str
    experiment_id: str
    run_repository_revision: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    runtime_backend: str
    runtime_revision: str
    evaluation_id: str
    trace_sha256: str
    seed: int
    selection: ProspectKvSelectionHandoffV2
    baseline_output_sha256: str
    candidate_output_sha256: str
    baseline_logical_kv_bytes: int
    candidate_logical_kv_bytes: int
    metrics: tuple[ObservedMetric, ...]

    @classmethod
    def capture(
        cls,
        *,
        experiment_id: str,
        run_repository_revision: str,
        model_id: str,
        model_revision: str,
        tokenizer_revision: str,
        runtime_backend: str,
        runtime_revision: str,
        evaluation_id: str,
        trace_sha256: str,
        seed: int,
        selection: ProspectKvSelectionHandoffV2,
        baseline_output_sha256: str,
        candidate_output_sha256: str,
        metrics: Iterable[ObservedMetric],
    ) -> "ProspectKvRealModelSelectionEvidenceV2":
        evidence = cls(
            schema=PROSPECT_KV_REAL_MODEL_SELECTION_SCHEMA_V2,
            experiment_id=experiment_id,
            run_repository_revision=run_repository_revision,
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            runtime_backend=runtime_backend,
            runtime_revision=runtime_revision,
            evaluation_id=evaluation_id,
            trace_sha256=trace_sha256,
            seed=seed,
            selection=selection,
            baseline_output_sha256=baseline_output_sha256,
            candidate_output_sha256=candidate_output_sha256,
            baseline_logical_kv_bytes=selection.logical_input_bytes,
            candidate_logical_kv_bytes=selection.logical_retained_bytes,
            metrics=tuple(metrics),
        )
        evidence.validate()
        return evidence

    def validate(self) -> None:
        if self.schema != PROSPECT_KV_REAL_MODEL_SELECTION_SCHEMA_V2:
            raise ProspectKvRealModelSelectionV2Error(
                "unsupported real-model selection evidence schema"
            )
        for field_name, value in (
            ("experiment_id", self.experiment_id),
            ("model_id", self.model_id),
            ("model_revision", self.model_revision),
            ("tokenizer_revision", self.tokenizer_revision),
            ("runtime_backend", self.runtime_backend),
            ("runtime_revision", self.runtime_revision),
            ("evaluation_id", self.evaluation_id),
        ):
            _require_text(field_name, value)
        if not _GIT_SHA_RE.fullmatch(self.run_repository_revision):
            raise ProspectKvRealModelSelectionV2Error(
                "run_repository_revision must be a lowercase full Git SHA"
            )
        _require_sha256("trace_sha256", self.trace_sha256)
        _require_sha256("baseline_output_sha256", self.baseline_output_sha256)
        _require_sha256("candidate_output_sha256", self.candidate_output_sha256)
        if type(self.seed) is not int or not 0 <= self.seed <= (2**64 - 1):
            raise ProspectKvRealModelSelectionV2Error(
                "seed must be an unsigned 64-bit integer"
            )

        try:
            self.selection.validate_replay()
        except ProspectKvSelectionHandoffV2Error as error:
            raise ProspectKvRealModelSelectionV2Error(
                "invalid embedded selection"
            ) from error
        if self.baseline_logical_kv_bytes != self.selection.logical_input_bytes:
            raise ProspectKvRealModelSelectionV2Error(
                "baseline logical KV bytes do not match selection input"
            )
        if self.candidate_logical_kv_bytes != self.selection.logical_retained_bytes:
            raise ProspectKvRealModelSelectionV2Error(
                "candidate logical KV bytes do not match selection retention"
            )

        if not self.metrics:
            raise ProspectKvRealModelSelectionV2Error(
                "real-model selection evidence must contain at least one metric"
            )
        seen: set[str] = set()
        for metric in self.metrics:
            try:
                metric.validate()
            except ValueError as error:
                raise ProspectKvRealModelSelectionV2Error(
                    "invalid observed metric"
                ) from error
            if metric.name in seen:
                raise ProspectKvRealModelSelectionV2Error(
                    f"duplicate observed metric name: {metric.name}"
                )
            seen.add(metric.name)

    def canonical_json(self) -> str:
        self.validate()
        return json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_canonical_json(
        cls, payload: str
    ) -> "ProspectKvRealModelSelectionEvidenceV2":
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ProspectKvRealModelSelectionV2Error("invalid JSON") from error
        if not isinstance(raw, dict):
            raise ProspectKvRealModelSelectionV2Error("evidence must be a JSON object")
        if json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False) != payload:
            raise ProspectKvRealModelSelectionV2Error("evidence JSON is not canonical")

        expected = {
            "schema",
            "experiment_id",
            "run_repository_revision",
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "runtime_backend",
            "runtime_revision",
            "evaluation_id",
            "trace_sha256",
            "seed",
            "selection",
            "baseline_output_sha256",
            "candidate_output_sha256",
            "baseline_logical_kv_bytes",
            "candidate_logical_kv_bytes",
            "metrics",
        }
        if set(raw) != expected:
            raise ProspectKvRealModelSelectionV2Error(
                "evidence fields do not match schema v2"
            )

        selection_raw = raw["selection"]
        if not isinstance(selection_raw, dict):
            raise ProspectKvRealModelSelectionV2Error("selection must be an object")
        selection_payload = json.dumps(
            selection_raw, sort_keys=True, separators=(",", ":")
        )
        try:
            selection = ProspectKvSelectionHandoffV2.from_canonical_json(
                selection_payload
            )
        except ProspectKvSelectionHandoffV2Error as error:
            raise ProspectKvRealModelSelectionV2Error(
                "invalid embedded selection"
            ) from error

        metrics_raw = raw["metrics"]
        if not isinstance(metrics_raw, list):
            raise ProspectKvRealModelSelectionV2Error("metrics must be an array")
        metrics = tuple(_metric_from_mapping(item) for item in metrics_raw)

        evidence = cls(
            schema=_require_text("schema", raw["schema"]),
            experiment_id=_require_text("experiment_id", raw["experiment_id"]),
            run_repository_revision=_require_text(
                "run_repository_revision", raw["run_repository_revision"]
            ),
            model_id=_require_text("model_id", raw["model_id"]),
            model_revision=_require_text("model_revision", raw["model_revision"]),
            tokenizer_revision=_require_text(
                "tokenizer_revision", raw["tokenizer_revision"]
            ),
            runtime_backend=_require_text("runtime_backend", raw["runtime_backend"]),
            runtime_revision=_require_text(
                "runtime_revision", raw["runtime_revision"]
            ),
            evaluation_id=_require_text("evaluation_id", raw["evaluation_id"]),
            trace_sha256=_require_text("trace_sha256", raw["trace_sha256"]),
            seed=_require_u64("seed", raw["seed"]),
            selection=selection,
            baseline_output_sha256=_require_text(
                "baseline_output_sha256", raw["baseline_output_sha256"]
            ),
            candidate_output_sha256=_require_text(
                "candidate_output_sha256", raw["candidate_output_sha256"]
            ),
            baseline_logical_kv_bytes=_require_non_negative_int(
                "baseline_logical_kv_bytes", raw["baseline_logical_kv_bytes"]
            ),
            candidate_logical_kv_bytes=_require_non_negative_int(
                "candidate_logical_kv_bytes", raw["candidate_logical_kv_bytes"]
            ),
            metrics=metrics,
        )
        evidence.validate()
        return evidence


def _metric_from_mapping(raw: Any) -> ObservedMetric:
    if not isinstance(raw, Mapping):
        raise ProspectKvRealModelSelectionV2Error("metric must be an object")
    expected = {
        "name",
        "kind",
        "unit",
        "preference",
        "baseline_value",
        "candidate_value",
        "delta",
    }
    if set(raw) != expected:
        raise ProspectKvRealModelSelectionV2Error(
            "observed metric fields do not match schema v2"
        )
    metric = ObservedMetric(
        name=_require_text("metric.name", raw["name"]),
        kind=_require_text("metric.kind", raw["kind"]),
        unit=_require_text("metric.unit", raw["unit"]),
        preference=_require_text("metric.preference", raw["preference"]),
        baseline_value=_require_finite_number(
            "metric.baseline_value", raw["baseline_value"]
        ),
        candidate_value=_require_finite_number(
            "metric.candidate_value", raw["candidate_value"]
        ),
        delta=_require_finite_number("metric.delta", raw["delta"]),
    )
    try:
        metric.validate()
    except ValueError as error:
        raise ProspectKvRealModelSelectionV2Error(
            "invalid observed metric"
        ) from error
    return metric


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProspectKvRealModelSelectionV2Error(
            f"{name} must be a non-empty string"
        )
    return value


def _require_u64(name: str, value: Any) -> int:
    value = _require_non_negative_int(name, value)
    if value > (2**64 - 1):
        raise ProspectKvRealModelSelectionV2Error(f"{name} exceeds u64")
    return value


def _require_non_negative_int(name: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ProspectKvRealModelSelectionV2Error(
            f"{name} must be a non-negative integer"
        )
    return value


def _require_finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProspectKvRealModelSelectionV2Error(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ProspectKvRealModelSelectionV2Error(f"{name} must be finite")
    return converted


def _require_sha256(name: str, value: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise ProspectKvRealModelSelectionV2Error(
            f"{name} must be a lowercase 64-character SHA-256 digest"
        )
