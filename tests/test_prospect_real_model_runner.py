import hashlib
import sys
import unittest

from kvlab.prospect_real_model_runner import (
    ExternalJsonBackend,
    ProspectKvRealModelRunnerError,
    RealModelRunContext,
    run_budget_matched_policy_campaign,
    run_real_model_selection_campaign,
)
from kvlab.prospect_real_model_selection import (
    ProspectKvRealModelSelectionEvidenceV1,
)
from kvlab.prospect_selection_handoff import ProspectKvSelectionHandoffV1


_FAKE_BACKEND = r'''
import base64
import hashlib
import json
import sys

payload = sys.stdin.read()
request = json.loads(payload)
policy = request["policy"]
retained = request["retained_token_ids"]
mode = request["mode"]
if mode == "baseline":
    accuracy = 0.8
else:
    accuracy = {"lru": 0.76, "magnitude": 0.78}.get(policy, 0.70)
artifact = f"{mode}|{policy}|{','.join(str(token) for token in retained)}".encode()
response = {
    "schema": "kvlab.prospect-kv-backend-response/v2",
    "request_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    "applied_mode": mode,
    "applied_policy": policy,
    "applied_retained_token_ids": retained,
    "output_artifact_base64": base64.b64encode(artifact).decode("ascii"),
    "metrics": [
        {
            "name": "token_accuracy",
            "kind": "quality",
            "unit": "ratio",
            "preference": "higher_is_better",
            "value": accuracy,
        },
        {
            "name": "logit_l2",
            "kind": "numerical",
            "unit": "l2",
            "preference": "lower_is_better",
            "value": 0.0 if mode == "baseline" else 0.125,
        },
    ],
}
sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":"), allow_nan=False))
'''

_NONCANONICAL_BACKEND = r'''
import base64
import hashlib
import json
import sys
payload = sys.stdin.read()
request = json.loads(payload)
response = {
    "schema": "kvlab.prospect-kv-backend-response/v2",
    "request_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    "applied_mode": request["mode"],
    "applied_policy": request["policy"],
    "applied_retained_token_ids": request["retained_token_ids"],
    "output_artifact_base64": base64.b64encode(b"output").decode("ascii"),
    "metrics": [{
        "name": "quality",
        "kind": "quality",
        "unit": "ratio",
        "preference": "higher_is_better",
        "value": 1.0,
    }],
}
sys.stdout.write(json.dumps(response, sort_keys=True, indent=2))
'''

_MISMATCHED_METRIC_BACKEND = r'''
import base64
import hashlib
import json
import sys
payload = sys.stdin.read()
request = json.loads(payload)
mode = request["mode"]
response = {
    "schema": "kvlab.prospect-kv-backend-response/v2",
    "request_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    "applied_mode": mode,
    "applied_policy": request["policy"],
    "applied_retained_token_ids": request["retained_token_ids"],
    "output_artifact_base64": base64.b64encode(mode.encode()).decode("ascii"),
    "metrics": [{
        "name": "quality",
        "kind": "quality",
        "unit": "ratio" if mode == "baseline" else "percent",
        "preference": "higher_is_better",
        "value": 0.8,
    }],
}
sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")))
'''

_BAD_REQUEST_HASH_BACKEND = r'''
import base64
import json
import sys
request = json.loads(sys.stdin.read())
response = {
    "schema": "kvlab.prospect-kv-backend-response/v2",
    "request_sha256": "0" * 64,
    "applied_mode": request["mode"],
    "applied_policy": request["policy"],
    "applied_retained_token_ids": request["retained_token_ids"],
    "output_artifact_base64": base64.b64encode(b"output").decode("ascii"),
    "metrics": [{
        "name": "quality",
        "kind": "quality",
        "unit": "ratio",
        "preference": "higher_is_better",
        "value": 1.0,
    }],
}
sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")))
'''

_BAD_RETAINED_BACKEND = r'''
import base64
import hashlib
import json
import sys
payload = sys.stdin.read()
request = json.loads(payload)
applied = list(request["retained_token_ids"])
if len(applied) > 1:
    applied = applied[1:]
response = {
    "schema": "kvlab.prospect-kv-backend-response/v2",
    "request_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    "applied_mode": request["mode"],
    "applied_policy": request["policy"],
    "applied_retained_token_ids": applied,
    "output_artifact_base64": base64.b64encode(b"output").decode("ascii"),
    "metrics": [{
        "name": "quality",
        "kind": "quality",
        "unit": "ratio",
        "preference": "higher_is_better",
        "value": 1.0,
    }],
}
sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")))
'''


def _context() -> RealModelRunContext:
    return RealModelRunContext(
        experiment_id="real-model-policy-comparison-c1",
        run_repository_revision="a" * 40,
        model_id="example/model",
        model_revision="model-r1",
        tokenizer_revision="tokenizer-r1",
        runtime_backend="fixture-runtime",
        runtime_revision="runtime-r1",
        evaluation_id="holdout-001",
        trace_sha256="1" * 64,
        seed=7,
    )


def _selection(policy: str, retained: tuple[int, ...]) -> ProspectKvSelectionHandoffV1:
    return ProspectKvSelectionHandoffV1.capture(
        token_ids=(10, 11, 12, 13, 14),
        bytes_per_token=64,
        policy=policy,
        retained_token_ids=retained,
    )


class ProspectKvRealModelRunnerTests(unittest.TestCase):
    def test_runs_one_paired_baseline_and_budget_matched_candidates(self) -> None:
        backend = ExternalJsonBackend((sys.executable, "-c", _FAKE_BACKEND))
        evidence = run_budget_matched_policy_campaign(
            context=_context(),
            selections=(
                _selection("lru", (10, 12, 14)),
                _selection("magnitude", (11, 13, 14)),
            ),
            backend=backend,
        )

        self.assertEqual(len(evidence), 2)
        self.assertEqual(evidence[0].selection.policy, "lru")
        self.assertEqual(evidence[1].selection.policy, "magnitude")
        self.assertEqual(
            evidence[0].baseline_output_sha256,
            evidence[1].baseline_output_sha256,
        )
        expected_baseline = hashlib.sha256(b"baseline|None|10,11,12,13,14").hexdigest()
        self.assertEqual(evidence[0].baseline_output_sha256, expected_baseline)
        self.assertAlmostEqual(evidence[0].metrics[0].candidate_value, 0.125)
        self.assertAlmostEqual(evidence[0].metrics[1].candidate_value, 0.76)
        self.assertAlmostEqual(evidence[1].metrics[1].candidate_value, 0.78)

        for record in evidence:
            payload = record.canonical_json()
            self.assertEqual(
                ProspectKvRealModelSelectionEvidenceV1.from_canonical_json(payload),
                record,
            )

    def test_rejects_budget_mismatch_before_backend_execution(self) -> None:
        backend = ExternalJsonBackend(("definitely-not-a-real-command",))
        with self.assertRaisesRegex(
            ProspectKvRealModelRunnerError,
            "equal logical_retained_bytes",
        ):
            run_budget_matched_policy_campaign(
                context=_context(),
                selections=(
                    _selection("lru", (10, 12, 14)),
                    _selection("magnitude", (13, 14)),
                ),
                backend=backend,
            )

    def test_rejects_noncanonical_backend_response(self) -> None:
        backend = ExternalJsonBackend((sys.executable, "-c", _NONCANONICAL_BACKEND))
        with self.assertRaisesRegex(ProspectKvRealModelRunnerError, "canonical"):
            run_real_model_selection_campaign(
                context=_context(),
                selections=(_selection("lru", (10, 12, 14)),),
                backend=backend,
            )

    def test_rejects_baseline_candidate_metric_metadata_drift(self) -> None:
        backend = ExternalJsonBackend((sys.executable, "-c", _MISMATCHED_METRIC_BACKEND))
        with self.assertRaisesRegex(ProspectKvRealModelRunnerError, "metadata mismatch"):
            run_real_model_selection_campaign(
                context=_context(),
                selections=(_selection("lru", (10, 12, 14)),),
                backend=backend,
            )

    def test_rejects_response_for_different_request(self) -> None:
        backend = ExternalJsonBackend((sys.executable, "-c", _BAD_REQUEST_HASH_BACKEND))
        with self.assertRaisesRegex(ProspectKvRealModelRunnerError, "exact request SHA-256"):
            run_real_model_selection_campaign(
                context=_context(),
                selections=(_selection("lru", (10, 12, 14)),),
                backend=backend,
            )

    def test_rejects_backend_retained_set_drift(self) -> None:
        backend = ExternalJsonBackend((sys.executable, "-c", _BAD_RETAINED_BACKEND))
        with self.assertRaisesRegex(ProspectKvRealModelRunnerError, "do not match request"):
            run_real_model_selection_campaign(
                context=_context(),
                selections=(_selection("lru", (10, 12, 14)),),
                backend=backend,
            )

    def test_rejects_invalid_context_before_backend_execution(self) -> None:
        bad_context = RealModelRunContext(
            experiment_id="experiment",
            run_repository_revision="deadbeef",
            model_id="model",
            model_revision="model-r1",
            tokenizer_revision="tok-r1",
            runtime_backend="runtime",
            runtime_revision="runtime-r1",
            evaluation_id="eval",
            trace_sha256="1" * 64,
            seed=0,
        )
        backend = ExternalJsonBackend(("definitely-not-a-real-command",))
        with self.assertRaisesRegex(ProspectKvRealModelRunnerError, "full Git SHA"):
            run_real_model_selection_campaign(
                context=bad_context,
                selections=(_selection("lru", (10, 12, 14)),),
                backend=backend,
            )


if __name__ == "__main__":
    unittest.main()
