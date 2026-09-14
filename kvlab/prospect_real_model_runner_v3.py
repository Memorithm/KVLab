"""Real-model KV-selection runner protocol with explicit model-token traces.

Protocol v3 separates three concepts that v2 could not safely distinguish:

* logical KV row identities, which are unique and drive selection evidence;
* actual model vocabulary token ids, which are position-aligned and may repeat;
* a non-empty teacher-forced evaluation continuation applied after selection.

The complete trace is canonicalized and SHA-256 bound to ``trace_sha256`` in the
run context.  This makes a real-model execution request self-contained enough
for a backend to map retained logical identities to physical KV rows without
pretending vocabulary token ids are unique.

This protocol still does not establish remote attestation of backend internals,
physical memory release, or performance/quality claims beyond the metrics a
real backend actually reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
import binascii
import hashlib
import json
import math
import subprocess
from typing import Any, Sequence

from .prospect_real_model_runner import (
    BackendMetricValue,
    BackendObservation,
    ProspectKvRealModelRunnerError,
    RealModelRunContext,
    _pair_metrics,
    _parse_backend_metrics,
)
from .prospect_real_model_selection import ProspectKvRealModelSelectionEvidenceV1
from .prospect_selection_handoff import ProspectKvSelectionHandoffV1


PROSPECT_KV_REAL_MODEL_TRACE_SCHEMA_V1 = "kvlab.prospect-kv-real-model-trace/v1"
PROSPECT_KV_BACKEND_REQUEST_SCHEMA_V3 = "kvlab.prospect-kv-backend-request/v3"
PROSPECT_KV_BACKEND_RESPONSE_SCHEMA_V3 = "kvlab.prospect-kv-backend-response/v3"


@dataclass(frozen=True, slots=True)
class RealModelEvaluationTraceV1:
    """Exact logical/model-token mapping plus teacher-forced continuation."""

    schema: str
    logical_input_token_ids: tuple[int, ...]
    model_input_token_ids: tuple[int, ...]
    evaluation_token_ids: tuple[int, ...]

    @classmethod
    def capture(
        cls,
        *,
        logical_input_token_ids: Sequence[int],
        model_input_token_ids: Sequence[int],
        evaluation_token_ids: Sequence[int],
    ) -> "RealModelEvaluationTraceV1":
        trace = cls(
            schema=PROSPECT_KV_REAL_MODEL_TRACE_SCHEMA_V1,
            logical_input_token_ids=_logical_ids(
                "logical_input_token_ids", logical_input_token_ids
            ),
            model_input_token_ids=_model_token_ids(
                "model_input_token_ids", model_input_token_ids, allow_empty=False
            ),
            evaluation_token_ids=_model_token_ids(
                "evaluation_token_ids", evaluation_token_ids, allow_empty=False
            ),
        )
        trace.validate()
        return trace

    def validate(self) -> None:
        if self.schema != PROSPECT_KV_REAL_MODEL_TRACE_SCHEMA_V1:
            raise ProspectKvRealModelRunnerError("unsupported real-model trace schema")
        logical = _logical_ids("logical_input_token_ids", self.logical_input_token_ids)
        model = _model_token_ids(
            "model_input_token_ids", self.model_input_token_ids, allow_empty=False
        )
        _model_token_ids(
            "evaluation_token_ids", self.evaluation_token_ids, allow_empty=False
        )
        if len(logical) != len(model):
            raise ProspectKvRealModelRunnerError(
                "logical_input_token_ids and model_input_token_ids must be position-aligned"
            )

    def canonical_json(self) -> str:
        self.validate()
        return _canonical_json(asdict(self))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_canonical_json(cls, payload: str) -> "RealModelEvaluationTraceV1":
        try:
            raw = json.loads(payload, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ProspectKvRealModelRunnerError) as error:
            raise ProspectKvRealModelRunnerError("invalid real-model trace JSON") from error
        if not isinstance(raw, dict):
            raise ProspectKvRealModelRunnerError("real-model trace must be an object")
        if _canonical_json(raw) != payload:
            raise ProspectKvRealModelRunnerError("real-model trace JSON must be canonical")
        expected = {
            "schema",
            "logical_input_token_ids",
            "model_input_token_ids",
            "evaluation_token_ids",
        }
        if set(raw) != expected:
            raise ProspectKvRealModelRunnerError(
                "real-model trace fields do not match schema v1"
            )
        trace = cls(
            schema=_text("trace.schema", raw["schema"]),
            logical_input_token_ids=_logical_ids(
                "logical_input_token_ids", raw["logical_input_token_ids"]
            ),
            model_input_token_ids=_model_token_ids(
                "model_input_token_ids", raw["model_input_token_ids"], allow_empty=False
            ),
            evaluation_token_ids=_model_token_ids(
                "evaluation_token_ids", raw["evaluation_token_ids"], allow_empty=False
            ),
        )
        trace.validate()
        return trace


@dataclass(frozen=True, slots=True)
class ExternalJsonBackendV3:
    """External argv backend for canonical request/response protocol v3."""

    command: tuple[str, ...]
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if not self.command or any(not isinstance(part, str) or not part for part in self.command):
            raise ProspectKvRealModelRunnerError(
                "backend command must contain non-empty argv strings"
            )
        if isinstance(self.timeout_seconds, bool) or not isinstance(
            self.timeout_seconds, (int, float)
        ):
            raise ProspectKvRealModelRunnerError("backend timeout must be numeric")
        if not math.isfinite(float(self.timeout_seconds)) or self.timeout_seconds <= 0:
            raise ProspectKvRealModelRunnerError("backend timeout must be positive and finite")

    def execute(self, request: dict[str, Any]) -> BackendObservation:
        payload = _canonical_json(request)
        request_sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()
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
        return _parse_backend_response_v3(
            completed.stdout,
            expected_request_sha256=request_sha256,
            expected_mode=request["mode"],
            expected_policy=request["policy"],
            expected_retained_logical_token_ids=request[
                "retained_logical_token_ids"
            ],
        )


def run_real_model_selection_campaign_v3(
    *,
    context: RealModelRunContext,
    trace: RealModelEvaluationTraceV1,
    selections: Sequence[ProspectKvSelectionHandoffV1],
    backend: ExternalJsonBackendV3,
) -> tuple[ProspectKvRealModelSelectionEvidenceV1, ...]:
    """Execute one full-history baseline and explicit selected-history candidates."""

    context.validate()
    trace.validate()
    if context.trace_sha256 != trace.sha256:
        raise ProspectKvRealModelRunnerError(
            "context trace_sha256 does not match the canonical real-model trace"
        )
    if not selections:
        raise ProspectKvRealModelRunnerError("selection campaign must not be empty")

    first = selections[0]
    first.validate_replay()
    logical_ids = trace.logical_input_token_ids
    if first.input_token_ids != logical_ids:
        raise ProspectKvRealModelRunnerError(
            "selection input_token_ids do not match trace logical_input_token_ids"
        )
    bytes_per_token = first.bytes_per_token
    seen_policies: set[str] = set()
    for selection in selections:
        selection.validate_replay()
        if selection.input_token_ids != logical_ids:
            raise ProspectKvRealModelRunnerError(
                "all selections must match trace logical_input_token_ids"
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
        _backend_request_v3(
            context=context,
            trace=trace,
            retained_logical_token_ids=logical_ids,
            bytes_per_token=bytes_per_token,
            policy=None,
        )
    )

    evidence: list[ProspectKvRealModelSelectionEvidenceV1] = []
    for selection in selections:
        candidate = backend.execute(
            _backend_request_v3(
                context=context,
                trace=trace,
                retained_logical_token_ids=selection.retained_token_ids,
                bytes_per_token=bytes_per_token,
                policy=selection.policy,
            )
        )
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
                metrics=_pair_metrics(baseline.metrics, candidate.metrics),
            )
        )
    return tuple(evidence)


def run_budget_matched_policy_campaign_v3(
    *,
    context: RealModelRunContext,
    trace: RealModelEvaluationTraceV1,
    selections: Sequence[ProspectKvSelectionHandoffV1],
    backend: ExternalJsonBackendV3,
) -> tuple[ProspectKvRealModelSelectionEvidenceV1, ...]:
    """Execute v3 candidates only when their logical retained-byte budgets match."""

    if not selections:
        raise ProspectKvRealModelRunnerError("selection campaign must not be empty")
    budget = selections[0].logical_retained_bytes
    if any(selection.logical_retained_bytes != budget for selection in selections[1:]):
        raise ProspectKvRealModelRunnerError(
            "budget-matched campaign requires equal logical_retained_bytes"
        )
    return run_real_model_selection_campaign_v3(
        context=context,
        trace=trace,
        selections=selections,
        backend=backend,
    )


def _backend_request_v3(
    *,
    context: RealModelRunContext,
    trace: RealModelEvaluationTraceV1,
    retained_logical_token_ids: Sequence[int],
    bytes_per_token: int,
    policy: str | None,
) -> dict[str, Any]:
    return {
        "schema": PROSPECT_KV_BACKEND_REQUEST_SCHEMA_V3,
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
        "logical_input_token_ids": list(trace.logical_input_token_ids),
        "model_input_token_ids": list(trace.model_input_token_ids),
        "retained_logical_token_ids": list(retained_logical_token_ids),
        "evaluation_token_ids": list(trace.evaluation_token_ids),
        "bytes_per_token": bytes_per_token,
        "policy": policy,
    }


def _parse_backend_response_v3(
    payload: str,
    *,
    expected_request_sha256: str,
    expected_mode: str,
    expected_policy: str | None,
    expected_retained_logical_token_ids: Sequence[int],
) -> BackendObservation:
    try:
        raw = json.loads(payload, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ProspectKvRealModelRunnerError) as error:
        raise ProspectKvRealModelRunnerError("backend returned invalid JSON") from error
    if not isinstance(raw, dict):
        raise ProspectKvRealModelRunnerError("backend response must be a JSON object")
    if _canonical_json(raw) != payload:
        raise ProspectKvRealModelRunnerError("backend response JSON must be canonical")
    expected_fields = {
        "schema",
        "request_sha256",
        "applied_mode",
        "applied_policy",
        "applied_retained_logical_token_ids",
        "output_artifact_base64",
        "metrics",
    }
    if set(raw) != expected_fields:
        raise ProspectKvRealModelRunnerError(
            "backend response fields do not match schema v3"
        )
    if raw["schema"] != PROSPECT_KV_BACKEND_RESPONSE_SCHEMA_V3:
        raise ProspectKvRealModelRunnerError("unsupported backend response schema")
    if raw["request_sha256"] != expected_request_sha256:
        raise ProspectKvRealModelRunnerError(
            "backend response does not attest the exact request SHA-256"
        )
    if raw["applied_mode"] != expected_mode:
        raise ProspectKvRealModelRunnerError("backend applied_mode does not match request")
    if raw["applied_policy"] != expected_policy:
        raise ProspectKvRealModelRunnerError("backend applied_policy does not match request")
    applied = _logical_ids(
        "applied_retained_logical_token_ids",
        raw["applied_retained_logical_token_ids"],
        allow_empty=True,
    )
    if applied != tuple(expected_retained_logical_token_ids):
        raise ProspectKvRealModelRunnerError(
            "backend applied retained logical token ids do not match request"
        )

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

    metrics: tuple[BackendMetricValue, ...] = _parse_backend_metrics(raw["metrics"])
    return BackendObservation(
        request_sha256=expected_request_sha256,
        applied_mode=expected_mode,
        applied_policy=expected_policy,
        applied_retained_token_ids=applied,
        artifact_bytes=artifact,
        artifact_sha256=hashlib.sha256(artifact).hexdigest(),
        metrics=metrics,
    )


def _logical_ids(
    name: str,
    value: Sequence[int] | Any,
    *,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ProspectKvRealModelRunnerError(f"{name} must be an array")
    result = tuple(value)
    if not allow_empty and not result:
        raise ProspectKvRealModelRunnerError(f"{name} must not be empty")
    if any(type(token_id) is not int or token_id < 0 for token_id in result):
        raise ProspectKvRealModelRunnerError(
            f"{name} must contain non-negative integer logical ids"
        )
    if len(result) != len(set(result)):
        raise ProspectKvRealModelRunnerError(f"{name} logical ids must be unique")
    return result


def _model_token_ids(
    name: str,
    value: Sequence[int] | Any,
    *,
    allow_empty: bool,
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ProspectKvRealModelRunnerError(f"{name} must be an array")
    result = tuple(value)
    if not allow_empty and not result:
        raise ProspectKvRealModelRunnerError(f"{name} must not be empty")
    if any(type(token_id) is not int or token_id < 0 for token_id in result):
        raise ProspectKvRealModelRunnerError(
            f"{name} must contain non-negative integer vocabulary token ids"
        )
    return result


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProspectKvRealModelRunnerError(f"{name} must be a non-empty string")
    return value


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
