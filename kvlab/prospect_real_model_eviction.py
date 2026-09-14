"""Canonical observed real-model KV-eviction evidence for ProspectEngine.

This module validates observations already produced by a real-model backend. It
never executes a model, derives quality from evicted bytes, or turns missing
telemetry into zero. The envelope binds exact model/runtime/trace provenance to
KVLab's replayable logical eviction contract plus baseline/candidate output
artefact digests and explicitly named observed numerical or quality metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import re
from typing import Any, Iterable, Mapping

from .prospect_eviction_handoff import ProspectKvEvictionHandoffV1


PROSPECT_KV_REAL_MODEL_EVIDENCE_SCHEMA_V1 = "kvlab.prospect-kv-real-model-eviction/v1"

_METRIC_KINDS = frozenset({"numerical", "quality"})
_PREFERENCES = frozenset({"higher_is_better", "lower_is_better", "none"})
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FLOAT_ABS_TOLERANCE = 1.0e-12
_FLOAT_REL_TOLERANCE = 1.0e-12


class ProspectKvRealModelEvidenceError(ValueError):
    """Raised when observed real-model evidence is malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class ObservedMetric:
    """One observed baseline/candidate metric with explicit interpretation."""

    name: str
    kind: str
    unit: str
    preference: str
    baseline_value: float
    candidate_value: float
    delta: float

    @classmethod
    def capture(
        cls,
        *,
        name: str,
        kind: str,
        unit: str,
        preference: str,
        baseline_value: float,
        candidate_value: float,
    ) -> "ObservedMetric":
        metric = cls(
            name=name,
            kind=kind,
            unit=unit,
            preference=preference,
            baseline_value=float(baseline_value),
            candidate_value=float(candidate_value),
            delta=float(candidate_value) - float(baseline_value),
        )
        metric.validate()
        return metric

    def validate(self) -> None:
        _require_text("metric.name", self.name)
        _require_text("metric.unit", self.unit)
        if self.kind not in _METRIC_KINDS:
            raise ProspectKvRealModelEvidenceError(
                f"unsupported metric kind: {self.kind!r}"
            )
        if self.preference not in _PREFERENCES:
            raise ProspectKvRealModelEvidenceError(
                f"unsupported metric preference: {self.preference!r}"
            )
        for field_name, value in (
            ("baseline_value", self.baseline_value),
            ("candidate_value", self.candidate_value),
            ("delta", self.delta),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ProspectKvRealModelEvidenceError(
                    f"metric.{field_name} must be numeric"
                )
            if not math.isfinite(float(value)):
                raise ProspectKvRealModelEvidenceError(
                    f"metric.{field_name} must be finite"
                )
        expected_delta = float(self.candidate_value) - float(self.baseline_value)
        if not _nearly_equal(float(self.delta), expected_delta):
            raise ProspectKvRealModelEvidenceError(
                "metric.delta must equal candidate_value - baseline_value"
            )


@dataclass(frozen=True, slots=True)
class ProspectKvRealModelEvictionEvidenceV1:
    """Observed paired baseline/candidate evidence for one exact logical eviction."""

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
    eviction: ProspectKvEvictionHandoffV1
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
        eviction: ProspectKvEvictionHandoffV1,
        baseline_output_sha256: str,
        candidate_output_sha256: str,
        metrics: Iterable[ObservedMetric],
    ) -> "ProspectKvRealModelEvictionEvidenceV1":
        evidence = cls(
            schema=PROSPECT_KV_REAL_MODEL_EVIDENCE_SCHEMA_V1,
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
            eviction=eviction,
            baseline_output_sha256=baseline_output_sha256,
            candidate_output_sha256=candidate_output_sha256,
            baseline_logical_kv_bytes=eviction.logical_input_bytes,
            candidate_logical_kv_bytes=eviction.logical_retained_bytes,
            metrics=tuple(metrics),
        )
        evidence.validate()
        return evidence

    def validate(self) -> None:
        if self.schema != PROSPECT_KV_REAL_MODEL_EVIDENCE_SCHEMA_V1:
            raise ProspectKvRealModelEvidenceError(
                "unsupported real-model eviction evidence schema"
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
            raise ProspectKvRealModelEvidenceError(
                "run_repository_revision must be a lowercase full Git SHA"
            )
        _require_sha256("trace_sha256", self.trace_sha256)
        _require_sha256("baseline_output_sha256", self.baseline_output_sha256)
        _require_sha256("candidate_output_sha256", self.candidate_output_sha256)
        if type(self.seed) is not int or self.seed < 0:
            raise ProspectKvRealModelEvidenceError(
                "seed must be a non-negative integer"
            )

        self.eviction.validate_replay()
        if type(self.baseline_logical_kv_bytes) is not int or self.baseline_logical_kv_bytes < 0:
            raise ProspectKvRealModelEvidenceError(
                "baseline_logical_kv_bytes must be a non-negative integer"
            )
        if type(self.candidate_logical_kv_bytes) is not int or self.candidate_logical_kv_bytes < 0:
            raise ProspectKvRealModelEvidenceError(
                "candidate_logical_kv_bytes must be a non-negative integer"
            )
        if self.baseline_logical_kv_bytes != self.eviction.logical_input_bytes:
            raise ProspectKvRealModelEvidenceError(
                "baseline logical KV bytes do not match the embedded eviction input"
            )
        if self.candidate_logical_kv_bytes != self.eviction.logical_retained_bytes:
            raise ProspectKvRealModelEvidenceError(
                "candidate logical KV bytes do not match the embedded eviction retention"
            )

        if not self.metrics:
            raise ProspectKvRealModelEvidenceError(
                "real-model eviction evidence must contain at least one observed metric"
            )
        seen_metric_names: set[str] = set()
        for metric in self.metrics:
            metric.validate()
            if metric.name in seen_metric_names:
                raise ProspectKvRealModelEvidenceError(
                    f"duplicate observed metric name: {metric.name}"
                )
            seen_metric_names.add(metric.name)

    def canonical_json(self) -> str:
        self.validate()
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_canonical_json(
        cls, payload: str
    ) -> "ProspectKvRealModelEvictionEvidenceV1":
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ProspectKvRealModelEvidenceError("invalid JSON") from error
        if not isinstance(raw, dict):
            raise ProspectKvRealModelEvidenceError(
                "real-model eviction evidence must be a JSON object"
            )
        if json.dumps(raw, sort_keys=True, separators=(",", ":")) != payload:
            raise ProspectKvRealModelEvidenceError(
                "real-model eviction evidence JSON is not canonical"
            )

        expected_fields = {
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
            "eviction",
            "baseline_output_sha256",
            "candidate_output_sha256",
            "baseline_logical_kv_bytes",
            "candidate_logical_kv_bytes",
            "metrics",
        }
        if set(raw) != expected_fields:
            raise ProspectKvRealModelEvidenceError(
                "real-model eviction evidence fields do not match schema v1"
            )
        if not isinstance(raw["eviction"], dict):
            raise ProspectKvRealModelEvidenceError("eviction must be an object")
        eviction_payload = json.dumps(
            raw["eviction"], sort_keys=True, separators=(",", ":")
        )
        try:
            eviction = ProspectKvEvictionHandoffV1.from_canonical_json(
                eviction_payload
            )
        except ValueError as error:
            raise ProspectKvRealModelEvidenceError(
                "invalid embedded eviction handoff"
            ) from error

        metrics_raw = raw["metrics"]
        if not isinstance(metrics_raw, list):
            raise ProspectKvRealModelEvidenceError("metrics must be an array")
        metrics = tuple(_metric_from_mapping(metric) for metric in metrics_raw)

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
            runtime_backend=_require_text(
                "runtime_backend", raw["runtime_backend"]
            ),
            runtime_revision=_require_text(
                "runtime_revision", raw["runtime_revision"]
            ),
            evaluation_id=_require_text("evaluation_id", raw["evaluation_id"]),
            trace_sha256=_require_text("trace_sha256", raw["trace_sha256"]),
            seed=_require_non_negative_int("seed", raw["seed"]),
            eviction=eviction,
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
        raise ProspectKvRealModelEvidenceError("metric must be an object")
    expected_fields = {
        "name",
        "kind",
        "unit",
        "preference",
        "baseline_value",
        "candidate_value",
        "delta",
    }
    if set(raw) != expected_fields:
        raise ProspectKvRealModelEvidenceError(
            "observed metric fields do not match schema v1"
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
    metric.validate()
    return metric


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProspectKvRealModelEvidenceError(f"{name} must be a non-empty string")
    return value


def _require_non_negative_int(name: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ProspectKvRealModelEvidenceError(
            f"{name} must be a non-negative integer"
        )
    return value


def _require_finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProspectKvRealModelEvidenceError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ProspectKvRealModelEvidenceError(f"{name} must be finite")
    return value


def _require_sha256(name: str, value: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise ProspectKvRealModelEvidenceError(
            f"{name} must be a lowercase 64-character SHA-256 digest"
        )


def _nearly_equal(left: float, right: float) -> bool:
    scale = max(abs(left), abs(right))
    return abs(left - right) <= _FLOAT_ABS_TOLERANCE + _FLOAT_REL_TOLERANCE * scale
