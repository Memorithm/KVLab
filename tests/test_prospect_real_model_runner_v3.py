import hashlib
import sys
import unittest

from kvlab.prospect_real_model_runner import (
    ProspectKvRealModelRunnerError,
    RealModelRunContext,
)
from kvlab.prospect_real_model_runner_v3 import (
    ExternalJsonBackendV3,
    RealModelEvaluationTraceV1,
    run_budget_matched_policy_campaign_v3,
    run_real_model_selection_campaign_v3,
)
from kvlab.prospect_real_model_selection import ProspectKvRealModelSelectionEvidenceV1
from kvlab.prospect_selection_handoff import ProspectKvSelectionHandoffV1


_FAKE_BACKEND_V3 = r'''
import base64
import hashlib
import json
import sys

payload = sys.stdin.read()
request = json.loads(payload)
mode = request["mode"]
policy = request["policy"]
retained = request["retained_logical_token_ids"]
assert request["model_input_token_ids"] == [42, 7, 42, 9, 7]
assert request["evaluation_token_ids"] == [5, 5, 6]
quality = 0.9 if mode == "baseline" else {"lru": 0.8, "magnitude": 0.85}[policy]
nll = 0.2 if mode == "baseline" else {"lru": 0.4, "magnitude": 0.3}[policy]
artifact = (
    f"{mode}|{policy}|{','.join(map(str, retained))}|"
    f"{','.join(map(str, request['evaluation_token_ids']))}"
).encode()
response = {
    "schema": "kvlab.prospect-kv-backend-response/v3",
    "request_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    "applied_mode": mode,
    "applied_policy": policy,
    "applied_retained_logical_token_ids": retained,
    "output_artifact_base64": base64.b64encode(artifact).decode("ascii"),
    "metrics": [
        {
            "name": "mean_nll",
            "kind": "quality",
            "unit": "nat_per_token",
            "preference": "lower_is_better",
            "value": nll,
        },
        {
            "name": "token_accuracy",
            "kind": "quality",
            "unit": "ratio",
            "preference": "higher_is_better",
            "value": quality,
        },
    ],
}
sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":"), allow_nan=False))
'''

_BAD_RETAINED_V3 = r'''
import base64
import hashlib
import json
import sys
payload = sys.stdin.read()
request = json.loads(payload)
retained = list(request["retained_logical_token_ids"])
if retained:
    retained = retained[:-1]
response = {
    "schema": "kvlab.prospect-kv-backend-response/v3",
    "request_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    "applied_mode": request["mode"],
    "applied_policy": request["policy"],
    "applied_retained_logical_token_ids": retained,
    "output_artifact_base64": base64.b64encode(b"output").decode("ascii"),
    "metrics": [{
        "name": "mean_nll",
        "kind": "quality",
        "unit": "nat_per_token",
        "preference": "lower_is_better",
        "value": 0.2,
    }],
}
sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")))
'''


def _trace() -> RealModelEvaluationTraceV1:
    return RealModelEvaluationTraceV1.capture(
        logical_input_token_ids=(10, 11, 12, 13, 14),
        model_input_token_ids=(42, 7, 42, 9, 7),
        evaluation_token_ids=(5, 5, 6),
    )


def _context(trace: RealModelEvaluationTraceV1) -> RealModelRunContext:
    return RealModelRunContext(
        experiment_id="real-model-policy-comparison-v3",
        run_repository_revision="a" * 40,
        model_id="example/model",
        model_revision="model-r1",
        tokenizer_revision="tokenizer-r1",
        runtime_backend="fixture-runtime-v3",
        runtime_revision="runtime-r1",
        evaluation_id="teacher-forced-holdout-001",
        trace_sha256=trace.sha256,
        seed=7,
    )


def _selection(policy: str, retained: tuple[int, ...]) -> ProspectKvSelectionHandoffV1:
    return ProspectKvSelectionHandoffV1.capture(
        token_ids=(10, 11, 12, 13, 14),
        bytes_per_token=64,
        policy=policy,
        retained_token_ids=retained,
    )


class ProspectKvRealModelRunnerV3Tests(unittest.TestCase):
    def test_trace_separates_unique_logical_ids_from_repeated_model_tokens(self) -> None:
        trace = _trace()
        self.assertEqual(trace.logical_input_token_ids, (10, 11, 12, 13, 14))
        self.assertEqual(trace.model_input_token_ids, (42, 7, 42, 9, 7))
        self.assertEqual(trace.evaluation_token_ids, (5, 5, 6))
        payload = trace.canonical_json()
        self.assertEqual(RealModelEvaluationTraceV1.from_canonical_json(payload), trace)
        self.assertEqual(trace.sha256, hashlib.sha256(payload.encode("utf-8")).hexdigest())

    def test_runs_budget_matched_v3_campaign_and_binds_trace(self) -> None:
        trace = _trace()
        backend = ExternalJsonBackendV3((sys.executable, "-c", _FAKE_BACKEND_V3))
        evidence = run_budget_matched_policy_campaign_v3(
            context=_context(trace),
            trace=trace,
            selections=(
                _selection("lru", (10, 12, 14)),
                _selection("magnitude", (11, 13, 14)),
            ),
            backend=backend,
        )
        self.assertEqual(len(evidence), 2)
        self.assertEqual(evidence[0].trace_sha256, trace.sha256)
        self.assertEqual(
            evidence[0].baseline_output_sha256,
            evidence[1].baseline_output_sha256,
        )
        self.assertAlmostEqual(evidence[0].metrics[0].baseline_value, 0.2)
        self.assertAlmostEqual(evidence[0].metrics[0].candidate_value, 0.4)
        self.assertAlmostEqual(evidence[1].metrics[0].candidate_value, 0.3)
        for record in evidence:
            payload = record.canonical_json()
            self.assertEqual(
                ProspectKvRealModelSelectionEvidenceV1.from_canonical_json(payload),
                record,
            )

    def test_rejects_trace_hash_drift_before_backend_execution(self) -> None:
        trace = _trace()
        context = RealModelRunContext(
            experiment_id="experiment",
            run_repository_revision="a" * 40,
            model_id="model",
            model_revision="model-r1",
            tokenizer_revision="tok-r1",
            runtime_backend="runtime",
            runtime_revision="runtime-r1",
            evaluation_id="eval",
            trace_sha256="0" * 64,
            seed=0,
        )
        backend = ExternalJsonBackendV3(("definitely-not-a-real-command",))
        with self.assertRaisesRegex(ProspectKvRealModelRunnerError, "canonical real-model trace"):
            run_real_model_selection_campaign_v3(
                context=context,
                trace=trace,
                selections=(_selection("lru", (10, 12, 14)),),
                backend=backend,
            )

    def test_rejects_misaligned_model_trace_and_empty_evaluation(self) -> None:
        with self.assertRaisesRegex(ProspectKvRealModelRunnerError, "position-aligned"):
            RealModelEvaluationTraceV1.capture(
                logical_input_token_ids=(10, 11, 12),
                model_input_token_ids=(42, 7),
                evaluation_token_ids=(5,),
            )
        with self.assertRaisesRegex(ProspectKvRealModelRunnerError, "must not be empty"):
            RealModelEvaluationTraceV1.capture(
                logical_input_token_ids=(10, 11, 12),
                model_input_token_ids=(42, 7, 42),
                evaluation_token_ids=(),
            )

    def test_rejects_backend_retained_logical_identity_drift(self) -> None:
        trace = _trace()
        backend = ExternalJsonBackendV3((sys.executable, "-c", _BAD_RETAINED_V3))
        with self.assertRaisesRegex(
            ProspectKvRealModelRunnerError,
            "retained logical token ids do not match request",
        ):
            run_real_model_selection_campaign_v3(
                context=_context(trace),
                trace=trace,
                selections=(_selection("lru", (10, 12, 14)),),
                backend=backend,
            )

    def test_rejects_budget_mismatch_before_backend_execution(self) -> None:
        trace = _trace()
        backend = ExternalJsonBackendV3(("definitely-not-a-real-command",))
        with self.assertRaisesRegex(
            ProspectKvRealModelRunnerError,
            "equal logical_retained_bytes",
        ):
            run_budget_matched_policy_campaign_v3(
                context=_context(trace),
                trace=trace,
                selections=(
                    _selection("lru", (10, 12, 14)),
                    _selection("magnitude", (13, 14)),
                ),
                backend=backend,
            )


if __name__ == "__main__":
    unittest.main()
