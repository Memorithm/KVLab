import base64
import hashlib
import json

import pytest

from kvlab.prospect_real_model_runner import (
    BackendMetricValue,
    ProspectKvRealModelRunnerError,
    RealModelRunContext,
)
from kvlab.prospect_real_model_runner_v4 import (
    BackendObservationV4,
    PositionModelEvaluationTraceV1,
    PROSPECT_KV_BACKEND_RESPONSE_SCHEMA_V4,
    _backend_request_v4,
    _parse_backend_response_v4,
    run_real_model_selection_campaign_v4,
)
from kvlab.prospect_selection_handoff_v2 import ProspectKvSelectionHandoffV2


def _context(trace: PositionModelEvaluationTraceV1) -> RealModelRunContext:
    return RealModelRunContext(
        experiment_id="position-campaign",
        run_repository_revision="a" * 40,
        model_id="example/model",
        model_revision="model-r1",
        tokenizer_revision="tok-r1",
        runtime_backend="nnis",
        runtime_revision="b" * 40,
        evaluation_id="eval-001",
        trace_sha256=trace.sha256,
        seed=7,
    )


def _metric(value: float) -> BackendMetricValue:
    metric = BackendMetricValue(
        name="token_accuracy",
        kind="quality",
        unit="ratio",
        preference="higher_is_better",
        value=value,
    )
    metric.validate()
    return metric


class _Backend:
    def __init__(self) -> None:
        self.requests = []

    def execute(self, request):
        self.requests.append(request)
        positions = tuple(request["retained_positions"])
        artifact = (
            b"baseline" if request["mode"] == "baseline" else b"candidate:" + bytes(positions)
        )
        return BackendObservationV4(
            request_sha256="0" * 64,
            applied_mode=request["mode"],
            applied_policy=request["policy"],
            applied_retained_positions=positions,
            artifact_bytes=artifact,
            artifact_sha256=hashlib.sha256(artifact).hexdigest(),
            metrics=(_metric(1.0 if request["mode"] == "baseline" else 0.75),),
        )


def test_position_trace_preserves_duplicate_model_token_values():
    trace = PositionModelEvaluationTraceV1.capture(
        model_input_token_ids=[7, 11, 7, 7, 19],
        evaluation_token_ids=[23, 7],
    )

    assert trace.model_input_token_ids == (7, 11, 7, 7, 19)
    assert PositionModelEvaluationTraceV1.from_canonical_json(
        trace.canonical_json()
    ) == trace
    assert len(trace.sha256) == 64


def test_campaign_uses_positions_as_identity_and_emits_v2_evidence():
    trace = PositionModelEvaluationTraceV1.capture(
        model_input_token_ids=[7, 11, 7, 7, 19],
        evaluation_token_ids=[23, 7],
    )
    selection = ProspectKvSelectionHandoffV2.capture(
        token_ids=trace.model_input_token_ids,
        bytes_per_token=64,
        policy="fixture",
        retained_positions=[0, 2, 4],
    )
    backend = _Backend()

    evidence = run_real_model_selection_campaign_v4(
        context=_context(trace),
        trace=trace,
        selections=[selection],
        backend=backend,
    )

    assert len(evidence) == 1
    assert evidence[0].selection.retained_positions == (0, 2, 4)
    assert evidence[0].selection.retained_token_ids == (7, 7, 19)
    assert evidence[0].metrics[0].baseline_value == 1.0
    assert evidence[0].metrics[0].candidate_value == 0.75
    assert backend.requests[0]["retained_positions"] == [0, 1, 2, 3, 4]
    assert backend.requests[1]["retained_positions"] == [0, 2, 4]
    assert backend.requests[1]["model_input_token_ids"] == [7, 11, 7, 7, 19]


def test_response_parser_rejects_position_attestation_drift():
    trace = PositionModelEvaluationTraceV1.capture(
        model_input_token_ids=[7, 11, 7, 7, 19],
        evaluation_token_ids=[23],
    )
    request = _backend_request_v4(
        context=_context(trace),
        trace=trace,
        retained_positions=[0, 2, 4],
        bytes_per_token=64,
        policy="fixture",
    )
    payload = json.dumps(request, sort_keys=True, separators=(",", ":"))
    request_sha = hashlib.sha256(payload.encode()).hexdigest()
    response = {
        "schema": PROSPECT_KV_BACKEND_RESPONSE_SCHEMA_V4,
        "request_sha256": request_sha,
        "applied_mode": "candidate",
        "applied_policy": "fixture",
        "applied_retained_positions": [0, 3, 4],
        "output_artifact_base64": base64.b64encode(b"artifact").decode(),
        "metrics": [
            {
                "name": "token_accuracy",
                "kind": "quality",
                "unit": "ratio",
                "preference": "higher_is_better",
                "value": 0.75,
            }
        ],
    }
    response_payload = json.dumps(response, sort_keys=True, separators=(",", ":"))

    with pytest.raises(ProspectKvRealModelRunnerError, match="positions do not match"):
        _parse_backend_response_v4(
            response_payload,
            expected_request_sha256=request_sha,
            expected_mode="candidate",
            expected_policy="fixture",
            expected_retained_positions=[0, 2, 4],
        )


def test_campaign_fails_closed_when_trace_hash_or_tokens_drift():
    trace = PositionModelEvaluationTraceV1.capture(
        model_input_token_ids=[7, 11, 7],
        evaluation_token_ids=[23],
    )
    selection = ProspectKvSelectionHandoffV2.capture(
        token_ids=[7, 11, 7],
        bytes_per_token=64,
        policy="fixture",
        retained_positions=[0, 2],
    )
    context = _context(trace)
    context = RealModelRunContext(
        experiment_id=context.experiment_id,
        run_repository_revision=context.run_repository_revision,
        model_id=context.model_id,
        model_revision=context.model_revision,
        tokenizer_revision=context.tokenizer_revision,
        runtime_backend=context.runtime_backend,
        runtime_revision=context.runtime_revision,
        evaluation_id=context.evaluation_id,
        trace_sha256="0" * 64,
        seed=context.seed,
    )

    with pytest.raises(ProspectKvRealModelRunnerError, match="trace_sha256"):
        run_real_model_selection_campaign_v4(
            context=context,
            trace=trace,
            selections=[selection],
            backend=_Backend(),
        )
