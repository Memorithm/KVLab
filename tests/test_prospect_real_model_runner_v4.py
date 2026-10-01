import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

from kvlab.prospect_real_model_runner import (
    BackendMetricValue,
    ProspectKvRealModelRunnerError,
    RealModelRunContext,
)
from kvlab.prospect_real_model_runner_v4 import (
    BackendObservationV4,
    ExternalJsonBackendV4,
    PositionModelEvaluationTraceV1,
    PROSPECT_KV_BACKEND_RESPONSE_SCHEMA_V4,
    _MAX_BACKEND_STDOUT_BYTES,
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
            b"baseline"
            if request["mode"] == "baseline"
            else b"candidate:" + bytes(positions)
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


class ProspectKvRealModelRunnerV4Tests(unittest.TestCase):
    def test_position_trace_preserves_duplicate_model_token_values(self):
        trace = PositionModelEvaluationTraceV1.capture(
            model_input_token_ids=[7, 11, 7, 7, 19],
            evaluation_token_ids=[23, 7],
        )

        self.assertEqual(trace.model_input_token_ids, (7, 11, 7, 7, 19))
        self.assertEqual(
            PositionModelEvaluationTraceV1.from_canonical_json(trace.canonical_json()),
            trace,
        )
        self.assertEqual(len(trace.sha256), 64)

    def test_campaign_uses_positions_as_identity_and_emits_v2_evidence(self):
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

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].selection.retained_positions, (0, 2, 4))
        self.assertEqual(evidence[0].selection.retained_token_ids, (7, 7, 19))
        self.assertEqual(evidence[0].metrics[0].baseline_value, 1.0)
        self.assertEqual(evidence[0].metrics[0].candidate_value, 0.75)
        self.assertEqual(backend.requests[0]["retained_positions"], [0, 1, 2, 3, 4])
        self.assertEqual(backend.requests[1]["retained_positions"], [0, 2, 4])
        self.assertEqual(
            backend.requests[1]["model_input_token_ids"], [7, 11, 7, 7, 19]
        )

    def test_response_parser_rejects_position_attestation_drift(self):
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

        with self.assertRaisesRegex(
            ProspectKvRealModelRunnerError, "positions do not match"
        ):
            _parse_backend_response_v4(
                response_payload,
                expected_request_sha256=request_sha,
                expected_mode="candidate",
                expected_policy="fixture",
                expected_retained_positions=[0, 2, 4],
            )

    @unittest.skipUnless(os.name == "posix", "requires POSIX process groups")
    def test_external_backend_starts_in_own_process_group(self):
        script = (
            "import base64,hashlib,json,os,sys;"
            "request=json.load(sys.stdin);"
            "payload=json.dumps(request,sort_keys=True,separators=(',',':'));"
            "artifact=f'{os.getpid()}:{os.getpgrp()}'.encode();"
            "response={'schema':'kvlab.prospect-kv-backend-response/v4',"
            "'request_sha256':hashlib.sha256(payload.encode()).hexdigest(),"
            "'applied_mode':request['mode'],'applied_policy':request['policy'],"
            "'applied_retained_positions':request['retained_positions'],"
            "'output_artifact_base64':base64.b64encode(artifact).decode(),"
            "'metrics':[{'name':'token_accuracy','kind':'quality','unit':'ratio',"
            "'preference':'higher_is_better','value':1.0}]};"
            "sys.stdout.write(json.dumps(response,sort_keys=True,separators=(',',':')))"
        )
        backend = ExternalJsonBackendV4(
            command=(sys.executable, "-c", script), timeout_seconds=5.0
        )
        observation = backend.execute(
            {"mode": "baseline", "policy": None, "retained_positions": []}
        )
        process_id, process_group_id = observation.artifact_bytes.decode().split(":")
        self.assertEqual(process_id, process_group_id)

    @unittest.skipUnless(os.name == "posix", "requires POSIX process groups")
    def test_external_backend_timeout_kills_descendant_group_and_returns_boundedly(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_path = Path(directory) / "descendant.pid"
            script = (
                "import subprocess,sys,time;"
                "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
                f"open({str(pid_path)!r},'w',encoding='utf-8').write(str(child.pid));"
                "time.sleep(60)"
            )
            backend = ExternalJsonBackendV4(
                command=(sys.executable, "-c", script), timeout_seconds=0.2
            )
            started = time.monotonic()
            with self.assertRaisesRegex(
                ProspectKvRealModelRunnerError, "wall-clock deadline"
            ):
                backend.execute(
                    {"mode": "baseline", "policy": None, "retained_positions": []}
                )
            self.assertLess(time.monotonic() - started, 3.0)

            descendant_pid = int(pid_path.read_text(encoding="utf-8"))
            state_path = Path(f"/proc/{descendant_pid}/stat")
            for _ in range(50):
                if not state_path.exists():
                    break
                fields = state_path.read_text(encoding="utf-8").split()
                if len(fields) >= 3 and fields[2] == "Z":
                    break
                time.sleep(0.02)
            else:
                self.fail("backend descendant survived process-group timeout")

    @unittest.skipUnless(os.name == "posix", "requires POSIX process groups")
    def test_timeout_does_not_block_on_escaped_descendant_pipe(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_path = Path(directory) / "escaped.pid"
            script = (
                "import subprocess,sys,time;"
                "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],"
                "start_new_session=True);"
                f"open({str(pid_path)!r},'w',encoding='utf-8').write(str(child.pid));"
                "time.sleep(60)"
            )
            backend = ExternalJsonBackendV4(
                command=(sys.executable, "-c", script), timeout_seconds=0.2
            )
            started = time.monotonic()
            try:
                with self.assertRaisesRegex(
                    ProspectKvRealModelRunnerError, "wall-clock deadline"
                ):
                    backend.execute(
                        {"mode": "baseline", "policy": None, "retained_positions": []}
                    )
                self.assertLess(time.monotonic() - started, 3.0)
            finally:
                if pid_path.exists():
                    try:
                        os.kill(int(pid_path.read_text(encoding="utf-8")), 9)
                    except ProcessLookupError:
                        pass

    @unittest.skipUnless(os.name == "posix", "requires POSIX process groups")
    def test_large_backend_stdin_uses_zero_copy_memoryview_slices(self):
        script = (
            "import base64,hashlib,json,sys,time;"
            "time.sleep(0.05);"
            "request=json.load(sys.stdin);"
            "payload=json.dumps(request,sort_keys=True,separators=(',',':'));"
            "response={'schema':'kvlab.prospect-kv-backend-response/v4',"
            "'request_sha256':hashlib.sha256(payload.encode()).hexdigest(),"
            "'applied_mode':request['mode'],'applied_policy':request['policy'],"
            "'applied_retained_positions':request['retained_positions'],"
            "'output_artifact_base64':base64.b64encode(b'ok').decode(),"
            "'metrics':[{'name':'token_accuracy','kind':'quality','unit':'ratio',"
            "'preference':'higher_is_better','value':1.0}]};"
            "sys.stdout.write(json.dumps(response,sort_keys=True,separators=(',',':')))"
        )
        backend = ExternalJsonBackendV4(
            command=(sys.executable, "-c", script), timeout_seconds=5.0
        )
        original_write = os.write
        large_write_types = []

        def recording_write(file_descriptor, data):
            if len(data) > 4096:
                large_write_types.append(type(data))
            return original_write(file_descriptor, data)

        request = {
            "mode": "baseline",
            "policy": None,
            "retained_positions": [],
            "padding": "x" * (256 * 1024),
        }
        with mock.patch(
            "kvlab.prospect_real_model_runner_v4.os.write",
            side_effect=recording_write,
        ):
            observation = backend.execute(request)

        self.assertEqual(observation.artifact_bytes, b"ok")
        self.assertGreater(len(large_write_types), 1)
        self.assertTrue(
            all(payload_type is memoryview for payload_type in large_write_types)
        )

    @unittest.skipUnless(os.name == "posix", "requires POSIX process groups")
    def test_external_backend_stdout_is_capped_while_drained(self):
        script = (
            "import sys;"
            f"sys.stdout.buffer.write(b'x'*({_MAX_BACKEND_STDOUT_BYTES}+1));"
            "sys.stdout.buffer.flush()"
        )
        backend = ExternalJsonBackendV4(
            command=(sys.executable, "-c", script), timeout_seconds=5.0
        )
        with self.assertRaisesRegex(
            ProspectKvRealModelRunnerError, "stdout exceeded the byte limit"
        ):
            backend.execute(
                {"mode": "baseline", "policy": None, "retained_positions": []}
            )

    def test_campaign_fails_closed_when_trace_hash_drifts(self):
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
        original = _context(trace)
        context = RealModelRunContext(
            experiment_id=original.experiment_id,
            run_repository_revision=original.run_repository_revision,
            model_id=original.model_id,
            model_revision=original.model_revision,
            tokenizer_revision=original.tokenizer_revision,
            runtime_backend=original.runtime_backend,
            runtime_revision=original.runtime_revision,
            evaluation_id=original.evaluation_id,
            trace_sha256="0" * 64,
            seed=original.seed,
        )

        with self.assertRaisesRegex(ProspectKvRealModelRunnerError, "trace_sha256"):
            run_real_model_selection_campaign_v4(
                context=context,
                trace=trace,
                selections=[selection],
                backend=_Backend(),
            )


if __name__ == "__main__":
    unittest.main()
