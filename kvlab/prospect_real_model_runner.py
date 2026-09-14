"""Execute observed KV-selection experiments through an external model backend.

The runner is intentionally backend-agnostic.  It sends one canonical JSON
request per execution to an argv-style command, never through a shell.  The
backend returns a canonical JSON response containing an opaque base64 artefact
and explicitly named observed metrics.  KVLab hashes the decoded artefact bytes
itself and pairs baseline/candidate metrics before constructing the existing
``kvlab.prospect-kv-real-model-selection/v1`` evidence envelope.

This module provides execution plumbing, not a model implementation and not a
performance claim.  Logical KV bytes are never converted into HBM, traffic,
latency, throughput, or quality claims.  A selection's policy label remains
provenance only; it does not prove that the named heuristic generated it.
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import binascii
import hashlib
import json
import math
import re
import subprocess
from typing import Any, Iterable, Sequence

from .prospect_real_model_eviction import ObservedMetric
from .prospect_real_model_selection import ProspectKvRealModelSelectionEvidenceV1
from .prospect_selection_handoff import ProspectKvSelectionHandoffV1


PROSPECT_KV_BACKEND_REQUEST_SCHEMA_V1 = "kvlab.prospect-kv-backend-request/v1"
PROSPECT_KV_BACKEND_RESPONSE_SCHEMA_V1 = "kvlab.prospect-kv-backend-response/v1"

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_METRIC_KINDS = frozenset({"numerical", "quality"})
_PREFERENCES = frozenset({"higher_is_better", "lower_is_better", "none"})


class ProspectKvRealModelRunnerError(RuntimeError):
    """Raised when runner inputs or an external backend response are invalid."""


@dataclass(frozen=True, slots=True)
class RealModelRunContext:
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

    def validate(self) -> None:
        for name, value in (
            ("experiment_id", self.experiment_id),
            ("model_id", self.model_id),
            ("model_revision", self.model_revision),
            ("tokenizer_revision", self.tokenizer_revision),
            ("runtime_backend", self.runtime_backend),
            ("runtime_revision", self.runtime_revision),
            ("evaluation_id", self.evaluation_id),
        ):
            _require_text(name, value)
        if not _GIT_SHA_RE.fullmatch(self.run_repository_revision):
            raise ProspectKvRealModelRunnerError(
                "run_repository_revision must be a lowercase full Git SHA"
            )
        if not _SHA256_RE.fullmatch(self.trace_sha256):
            raise ProspectKvRealModelRunnerError(
                "trace_sha256 must be a lowercase SHA-256 digest"
            )
        if type(self.seed) is not int or not 0 <= self.seed <= (2**64 - 1):
            raise ProspectKvRealModelRunnerError("seed must be an unsigned 64-bit integer")


@dataclass(frozen=True, slots=True)
class BackendMetricValue:
    name: str
    kind: str
    unit: str
    preference: str
    value: float

    def validate(self) -> None:
        _require_text("metric.name", self.name)
        _require_text("metric.unit", self.unit)
        if self.kind not in _METRIC_KINDS:
            raise ProspectKvRealModelRunnerError(
                f"unsupported backend metric kind: {self.kind!r}"
            )
        if self.preference not in _PREFERENCES:
            raise ProspectKvRealModelRunnerError(
                f"unsupported backend metric preference: {self.preference!r}"
            )
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ProspectKvRealModelRunnerError("backend metric value must be numeric")
        if not math.isfinite(float(self.value)):
            raise ProspectKvRealModelRunnerError("backend metric value must be finite")


@dataclass(frozen=True, slots=True)
class BackendObservation:
    artifact_bytes: bytes
    artifact_sha256: str
    metrics: tuple[BackendMetricValue, ...]


@dataclass(frozen=True, slots=True)
class ExternalJsonBackend:
    """External argv-style backend using canonical JSON over stdin/stdout."""

    command: tuple[str, ...]
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if not self.command or any(not isinstance(part, str) or not part for part in self.command):
            raise ProspectKvRealModelRunnerError(
                "backend command must contain non-empty argv strings"
            )
        if not isinstance(self.timeout_seconds, (int, float)) or isinstance(
            self.timeout_seconds, bool
        ):
            raise ProspectKvRealModelRunnerError("backend timeout must be numeric")
        if not math.isfinite(float(self.timeout_seconds)) or self.timeout_seconds <= 0:
            raise ProspectKvRealModelRunnerError("backend timeout must be positive and finite")

    def execute(self, request: dict[str, Any]) -> BackendObservation:
        payload = _canonical_json(request)
        try:
            completed = subprocess.run(
                self.command,
                input=payload,
                capture_output=True,
                text=True,
                timeout=float(self.timeout_seconds),
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ProspectKvRealModelRunnerError("external backend execution failed") from error
        if completed.returncode != 0:
            raise ProspectKvRealModelRunnerError(
                f"external backend exited with status {completed.returncode}"
            )
        return _parse_backend_response(completed.stdout)


def run_real_model_selection_campaign(
    *,
    context: RealModelRunContext,
    selections: Sequence[ProspectKvSelectionHandoffV1],
    backend: ExternalJsonBackend,
) -> tuple[ProspectKvRealModelSelectionEvidenceV1, ...]:
    """Execute one paired baseline and one candidate per explicit selection."""

    context.validate()
    if not selections:
        raise ProspectKvRealModelRunnerError("selection campaign must not be empty")

    first = selections[0]
    first.validate_replay()
    input_token_ids = first.input_token_ids
    bytes_per_token = first.bytes_per_token
    seen_policies: set[str] = set()
    for selection in selections:
        selection.validate_replay()
        if selection.input_token_ids != input_token_ids:
            raise ProspectKvRealModelRunnerError(
                "all selections must share the same input_token_ids"
            )
        if selection.bytes_per_token != bytes_per_token:
            raise ProspectKvRealModelRunnerError(
                "all selections must share the same bytes_per_token"
            )
        if selection.retained_token_ids == selection.input_token_ids:
            raise ProspectKvRealModelRunnerError(
                "candidate selection duplicates the full-cache baseline"
            )
        if selection.policy in seen_policies:
            raise ProspectKvRealModelRunnerError(
                f"duplicate selection policy: {selection.policy}"
            )
        seen_policies.add(selection.policy)

    baseline = backend.execute(
        _backend_request(
            context=context,
            input_token_ids=input_token_ids,
            retained_token_ids=input_token_ids,
            bytes_per_token=bytes_per_token,
            policy=None,
        )
    )

    evidence: list[ProspectKvRealModelSelectionEvidenceV1] = []
    for selection in selections:
        candidate = backend.execute(
            _backend_request(
                context=context,
                input_token_ids=input_token_ids,
                retained_token_ids=selection.retained_token_ids,
                bytes_per_token=bytes_per_token,
                policy=selection.policy,
            )
        )
        paired_metrics = _pair_metrics(baseline.metrics, candidate.metrics)
        evidence.append(
            ProspectKvRealModelSelectionEvidenceV1.capture(
                experiment_id=context.experiment_id,
                run_repository_revision=context.run_repository_revision,
                model_id=context.model_id,
                model_revision=context.model_revision,
                tokenizer_revision=context.tokenizer_revision,
                runtime_backend=context.runtime_backend,
                runtime_revision=context.runtime_revision,
                evaluation_id=context.evaluation_id,
                trace_sha256=context.trace_sha256,
                seed=context.seed,
                selection=selection,
                baseline_output_sha256=baseline.artifact_sha256,
                candidate_output_sha256=candidate.artifact_sha256,
                metrics=paired_metrics,
            )
        )
    return tuple(evidence)


def run_budget_matched_policy_campaign(
    *,
    context: RealModelRunContext,
    selections: Sequence[ProspectKvSelectionHandoffV1],
    backend: ExternalJsonBackend,
) -> tuple[ProspectKvRealModelSelectionEvidenceV1, ...]:
    """Run multiple policies only when their logical retained-byte budget matches."""

    if not selections:
        raise ProspectKvRealModelRunnerError("selection campaign must not be empty")
    budget = selections[0].logical_retained_bytes
    if any(selection.logical_retained_bytes != budget for selection in selections[1:]):
        raise ProspectKvRealModelRunnerError(
            "budget-matched campaign requires equal logical_retained_bytes"
        )
    return run_real_model_selection_campaign(
        context=context,
        selections=selections,
        backend=backend,
    )


def _backend_request(
    *,
    context: RealModelRunContext,
    input_token_ids: Sequence[int],
    retained_token_ids: Sequence[int],
    bytes_per_token: int,
    policy: str | None,
) -> dict[str, Any]:
    return {
        "schema": PROSPECT_KV_BACKEND_REQUEST_SCHEMA_V1,
        "mode": "baseline" if policy is None else "candidate",
        "experiment_id": context.experiment_id,
        "run_repository_revision": context.run_repository_revision,
        "model_id": context.model_id,
        "model_revision": context.model_revision,
        "tokenizer_revision": context.tokenizer_revision,
        "runtime_backend": context.runtime_backend,
        "runtime_revision": context.runtime_revision,
        "evaluation_id": context.evaluation_id,
        "trace_sha256": context.trace_sha256,
        "seed": context.seed,
        "input_token_ids": list(input_token_ids),
        "retained_token_ids": list(retained_token_ids),
        "bytes_per_token": bytes_per_token,
        "policy": policy,
    }


def _parse_backend_response(payload: str) -> BackendObservation:
    try:
        raw = json.loads(payload, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ProspectKvRealModelRunnerError) as error:
        raise ProspectKvRealModelRunnerError("backend returned invalid JSON") from error
    if not isinstance(raw, dict):
        raise ProspectKvRealModelRunnerError("backend response must be a JSON object")
    if _canonical_json(raw) != payload:
        raise ProspectKvRealModelRunnerError("backend response JSON must be canonical")
    if set(raw) != {"schema", "output_artifact_base64", "metrics"}:
        raise ProspectKvRealModelRunnerError("backend response fields do not match schema v1")
    if raw["schema"] != PROSPECT_KV_BACKEND_RESPONSE_SCHEMA_V1:
        raise ProspectKvRealModelRunnerError("unsupported backend response schema")
    encoded = raw["output_artifact_base64"]
    if not isinstance(encoded, str) or not encoded:
        raise ProspectKvRealModelRunnerError(
            "output_artifact_base64 must be a non-empty string"
        )
    try:
        artifact = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ProspectKvRealModelRunnerError("invalid base64 output artefact") from error
    if not artifact:
        raise ProspectKvRealModelRunnerError("decoded output artefact must not be empty")

    metrics_raw = raw["metrics"]
    if not isinstance(metrics_raw, list) or not metrics_raw:
        raise ProspectKvRealModelRunnerError("backend metrics must be a non-empty array")
    metrics: list[BackendMetricValue] = []
    seen: set[str] = set()
    for item in metrics_raw:
        if not isinstance(item, dict):
            raise ProspectKvRealModelRunnerError("backend metric must be an object")
        if set(item) != {"name", "kind", "unit", "preference", "value"}:
            raise ProspectKvRealModelRunnerError("backend metric fields do not match schema v1")
        metric = BackendMetricValue(
            name=_require_text("metric.name", item["name"]),
            kind=_require_text("metric.kind", item["kind"]),
            unit=_require_text("metric.unit", item["unit"]),
            preference=_require_text("metric.preference", item["preference"]),
            value=_require_finite_number("metric.value", item["value"]),
        )
        metric.validate()
        if metric.name in seen:
            raise ProspectKvRealModelRunnerError(
                f"duplicate backend metric name: {metric.name}"
            )
        seen.add(metric.name)
        metrics.append(metric)
    metrics.sort(key=lambda metric: metric.name)
    return BackendObservation(
        artifact_bytes=artifact,
        artifact_sha256=hashlib.sha256(artifact).hexdigest(),
        metrics=tuple(metrics),
    )


def _pair_metrics(
    baseline: Sequence[BackendMetricValue],
    candidate: Sequence[BackendMetricValue],
) -> tuple[ObservedMetric, ...]:
    baseline_by_name = {metric.name: metric for metric in baseline}
    candidate_by_name = {metric.name: metric for metric in candidate}
    if baseline_by_name.keys() != candidate_by_name.keys():
        raise ProspectKvRealModelRunnerError(
            "baseline and candidate metric names do not match"
        )

    paired: list[ObservedMetric] = []
    for name in sorted(baseline_by_name):
        left = baseline_by_name[name]
        right = candidate_by_name[name]
        if (left.kind, left.unit, left.preference) != (
            right.kind,
            right.unit,
            right.preference,
        ):
            raise ProspectKvRealModelRunnerError(
                f"baseline/candidate metric metadata mismatch for {name}"
            )
        paired.append(
            ObservedMetric.capture(
                name=name,
                kind=left.kind,
                unit=left.unit,
                preference=left.preference,
                baseline_value=left.value,
                candidate_value=right.value,
            )
        )
    return tuple(paired)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ProspectKvRealModelRunnerError("value is not canonical JSON data") from error


def _reject_json_constant(value: str) -> None:
    raise ProspectKvRealModelRunnerError(f"non-finite JSON constant is forbidden: {value}")


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProspectKvRealModelRunnerError(f"{name} must be a non-empty string")
    return value


def _require_finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProspectKvRealModelRunnerError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ProspectKvRealModelRunnerError(f"{name} must be finite")
    return converted
