"""Real-model KV-selection protocol using physical sequence positions.

Protocol v4 is the position-native companion to the earlier logical-id v3
protocol. Model vocabulary ids are preserved exactly and may repeat. KV row
occurrence identity is expressed only through zero-based sequence positions,
which maps directly to runtimes such as NNIS that compact an active KV prefix
by retained row indices.

The request SHA-256 attests the exact canonical request/response pair. It does
not prove backend internals, allocator release, HBM residency, traffic,
latency, throughput, or quality beyond metrics actually returned by a backend.
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
    ProspectKvRealModelRunnerError,
    RealModelRunContext,
    _pair_metrics,
    _parse_backend_metrics,
)
from .prospect_real_model_selection_v2 import ProspectKvRealModelSelectionEvidenceV2
from .prospect_selection_handoff_v2 import ProspectKvSelectionHandoffV2


PROSPECT_KV_REAL_MODEL_POSITION_TRACE_SCHEMA_V1 = (
    "kvlab.prospect-kv-real-model-position-trace/v1"
)
PROSPECT_KV_BACKEND_REQUEST_SCHEMA_V4 = "kvlab.prospect-kv-backend-request/v4"
PROSPECT_KV_BACKEND_RESPONSE_SCHEMA_V4 = "kvlab.prospect-kv-backend-response/v4"


@dataclass(frozen=True, slots=True)
class PositionModelEvaluationTraceV1:
    schema: str
    model_input_token_ids: tuple[int, ...]
    evaluation_token_ids: tuple[int, ...]

    @classmethod
    def capture(
        cls,
        *,
        model_input_token_ids: Sequence[int],
        evaluation_token_ids: Sequence[int],
    ) -> "PositionModelEvaluationTraceV1":
        trace = cls(
            schema=PROSPECT_KV_REAL_MODEL_POSITION_TRACE_SCHEMA_V1,
            model_input_token_ids=_token_ids(
                "model_input_token_ids", model_input_token_ids, allow_empty=False
            ),
            evaluation_token_ids=_token_ids(
                "evaluation_token_ids", evaluation_token_ids, allow_empty=False
            ),
        )
        trace.validate()
        return trace

    def validate(self) -> None:
        if self.schema != PROSPECT_KV_REAL_MODEL_POSITION_TRACE_SCHEMA_V1:
            raise ProspectKvRealModelRunnerError(
                "unsupported position-identified real-model trace schema"
            )
        _token_ids("model_input_token_ids", self.model_input_token_ids, allow_empty=False)
        _token_ids("evaluation_token_ids", self.evaluation_token_ids, allow_empty=False)

    def canonical_json(self) -> str:
        self.validate()
        return _canonical_json(asdict(self))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_canonical_json(cls, payload: str) -> "PositionModelEvaluationTraceV1":
        try:
            raw = json.loads(payload, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ProspectKvRealModelRunnerError) as error:
            raise ProspectKvRealModelRunnerError("invalid position trace JSON") from error
        if not isinstance(raw, dict):
            raise ProspectKvRealModelRunnerError("position trace must be an object")
        if _canonical_json(raw) != payload:
            raise ProspectKvRealModelRunnerError("position trace JSON must be canonical")
        if set(raw) != {"schema", "model_input_token_ids", "evaluation_token_ids"}:
            raise ProspectKvRealModelRunnerError(
                "position trace fields do not match schema v1"
            )
        trace = cls(
            schema=_text("trace.schema", raw["schema"]),
            model_input_token_ids=_token_ids(
                "model_input_token_ids", raw["model_input_token_ids"], allow_empty=False
            ),
            evaluation_token_ids=_token_ids(
                "evaluation_token_ids", raw["evaluation_token_ids"], allow_empty=False
            ),
        )
        trace.validate()
        return trace


@dataclass(frozen=True, slots=True)
class BackendObservationV4:
    request_sha256: str
    applied_mode: str
    applied_policy: str | None
    applied_retained_positions: tuple[int, ...]
    artifact_bytes: bytes
    artifact_sha256: str
    metrics: tuple[BackendMetricValue, ...]


@dataclass(frozen=True, slots=True)
class ExternalJsonBackendV4:
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

    def execute(self, request: dict[str, Any]) -> BackendObservationV4:
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
        return _parse_backend_response_v4(
            completed.stdout,
            expected_request_sha256=request_sha256,
            expected_mode=request["mode"],
            expected_policy=request["policy"],
            expected_retained_positions=request["retained_positions"],
        )


def run_real_model_selection_campaign_v4(
    *,
    context: RealModelRunContext,
    trace: PositionModelEvaluationTraceV1,
    selections: Sequence[ProspectKvSelectionHandoffV2],
    backend: Any,
) -> tuple[ProspectKvRealModelSelectionEvidenceV2, ...]:
    """Execute one full-history baseline and exact position-selected candidates."""

    context.validate()
    trace.validate()
    if context.trace_sha256 != trace.sha256:
        raise ProspectKvRealModelRunnerError(
            "context trace_sha256 does not match the canonical position trace"
        )
    if not selections:
        raise ProspectKvRealModelRunnerError("selection campaign must not be empty")

    model_tokens = trace.model_input_token_ids
    full_positions = tuple(range(len(model_tokens)))
    bytes_per_token = selections[0].bytes_per_token
    seen_policies: set[str] = set()
    for selection in selections:
        selection.validate_replay()
        if selection.input_token_ids != model_tokens:
            raise ProspectKvRealModelRunnerError(
                "all selections must match trace model_input_token_ids"
            )
        if selection.bytes_per_token != bytes_per_token:
            raise ProspectKvRealModelRunnerError(
                "all selections must share the same bytes_per_token"
            )
        if selection.retained_positions == full_positions:
            raise ProspectKvRealModelRunnerError(
                "candidate selection duplicates the full-cache baseline"
            )
        if selection.policy in seen_policies:
            raise ProspectKvRealModelRunnerError(
                f"duplicate selection policy: {selection.policy}"
            )
        seen_policies.add(selection.policy)

    baseline = backend.execute(
        _backend_request_v4(
            context=context,
            trace=trace,
            retained_positions=full_positions,
            bytes_per_token=bytes_per_token,
            policy=None,
        )
    )

    evidence: list[ProspectKvRealModelSelectionEvidenceV2] = []
    for selection in selections:
        candidate = backend.execute(
            _backend_request_v4(
                context=context,
                trace=trace,
                retained_positions=selection.retained_positions,
                bytes_per_token=bytes_per_token,
                policy=selection.policy,
            )
        )
        evidence.append(
            ProspectKvRealModelSelectionEvidenceV2.capture(
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


def run_budget_matched_policy_campaign_v4(
    *,
    context: RealModelRunContext,
    trace: PositionModelEvaluationTraceV1,
    selections: Sequence[ProspectKvSelectionHandoffV2],
    backend: Any,
) -> tuple[ProspectKvRealModelSelectionEvidenceV2, ...]:
    if not selections:
        raise ProspectKvRealModelRunnerError("selection campaign must not be empty")
    budget = selections[0].logical_retained_bytes
    if any(selection.logical_retained_bytes != budget for selection in selections[1:]):
        raise ProspectKvRealModelRunnerError(
            "budget-matched campaign requires equal logical_retained_bytes"
        )
    return run_real_model_selection_campaign_v4(
        context=context, trace=trace, selections=selections, backend=backend
    )


def _backend_request_v4(
    *,
    context: RealModelRunContext,
    trace: PositionModelEvaluationTraceV1,
    retained_positions: Sequence[int],
    bytes_per_token: int,
    policy: str | None,
) -> dict[str, Any]:
    positions = _positions(
        "retained_positions", retained_positions, input_len=len(trace.model_input_token_ids)
    )
    return {
        "schema": PROSPECT_KV_BACKEND_REQUEST_SCHEMA_V4,
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
        "model_input_token_ids": list(trace.model_input_token_ids),
        "retained_positions": list(positions),
        "evaluation_token_ids": list(trace.evaluation_token_ids),
        "bytes_per_token": bytes_per_token,
        "policy": policy,
    }


def _parse_backend_response_v4(
    payload: str,
    *,
    expected_request_sha256: str,
    expected_mode: str,
    expected_policy: str | None,
    expected_retained_positions: Sequence[int],
) -> BackendObservationV4:
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
        "applied_retained_positions",
        "output_artifact_base64",
        "metrics",
    }
    if set(raw) != expected_fields:
        raise ProspectKvRealModelRunnerError(
            "backend response fields do not match schema v4"
        )
    if raw["schema"] != PROSPECT_KV_BACKEND_RESPONSE_SCHEMA_V4:
        raise ProspectKvRealModelRunnerError("unsupported backend response schema")
    if raw["request_sha256"] != expected_request_sha256:
        raise ProspectKvRealModelRunnerError(
            "backend response does not attest the exact request SHA-256"
        )
    if raw["applied_mode"] != expected_mode:
        raise ProspectKvRealModelRunnerError("backend applied_mode does not match request")
    if raw["applied_policy"] != expected_policy:
        raise ProspectKvRealModelRunnerError("backend applied_policy does not match request")

    expected_positions = tuple(expected_retained_positions)
    applied = _positions_unbounded(
        "applied_retained_positions", raw["applied_retained_positions"]
    )
    if applied != expected_positions:
        raise ProspectKvRealModelRunnerError(
            "backend applied retained positions do not match request"
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

    metrics = _parse_backend_metrics(raw["metrics"])
    return BackendObservationV4(
        request_sha256=expected_request_sha256,
        applied_mode=expected_mode,
        applied_policy=expected_policy,
        applied_retained_positions=applied,
        artifact_bytes=artifact,
        artifact_sha256=hashlib.sha256(artifact).hexdigest(),
        metrics=metrics,
    )


def _positions(name: str, value: Sequence[int], *, input_len: int) -> tuple[int, ...]:
    positions = _positions_unbounded(name, value)
    if any(position >= input_len for position in positions):
        raise ProspectKvRealModelRunnerError(f"{name} contains out-of-range positions")
    return positions


def _positions_unbounded(name: str, value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ProspectKvRealModelRunnerError(f"{name} must be an array")
    positions = tuple(value)
    previous = None
    for position in positions:
        if type(position) is not int or position < 0:
            raise ProspectKvRealModelRunnerError(
                f"{name} must contain non-negative integer positions"
            )
        if previous is not None and position <= previous:
            raise ProspectKvRealModelRunnerError(
                f"{name} must be strictly increasing and unique"
            )
        previous = position
    return positions


def _token_ids(name: str, value: Any, *, allow_empty: bool) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ProspectKvRealModelRunnerError(f"{name} must be an array")
    tokens = tuple(value)
    if not allow_empty and not tokens:
        raise ProspectKvRealModelRunnerError(f"{name} must not be empty")
    if any(type(token_id) is not int or token_id < 0 for token_id in tokens):
        raise ProspectKvRealModelRunnerError(
            f"{name} must contain non-negative integer model token ids"
        )
    return tokens


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProspectKvRealModelRunnerError(f"{name} must be a non-empty string")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _reject_json_constant(value: str) -> None:
    raise ProspectKvRealModelRunnerError(f"non-finite JSON constant is forbidden: {value}")
