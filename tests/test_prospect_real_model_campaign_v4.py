import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from kvlab.prospect_real_model_campaign_v4 import (
    PROSPECT_KV_POSITION_CAMPAIGN_RESULT_SCHEMA_V1,
    PROSPECT_KV_POSITION_CAMPAIGN_SCHEMA_V1,
    PositionCampaignSpecV1,
    ProspectKvPositionCampaignError,
    execute_position_campaign_v4,
)
from kvlab.prospect_real_model_runner import (
    BackendMetricValue,
    ProspectKvRealModelRunnerError,
)
from kvlab.prospect_real_model_runner_v4 import BackendObservationV4
from kvlab.prospect_real_model_selection_v2 import (
    PROSPECT_KV_REAL_MODEL_SELECTION_SCHEMA_V2,
    ProspectKvRealModelSelectionEvidenceV2,
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _spec_payload():
    return _canonical(
        {
            "schema": PROSPECT_KV_POSITION_CAMPAIGN_SCHEMA_V1,
            "experiment_id": "tiny-position-campaign",
            "run_repository_revision": "a" * 40,
            "model_id": "example/model",
            "model_revision": "model-r1",
            "tokenizer_revision": "tok-r1",
            "runtime_backend": "nnis-kvlab-v4",
            "runtime_revision": "b" * 40,
            "evaluation_id": "teacher-forced-001",
            "seed": 7,
            "bytes_per_token": 64,
            "model_input_token_ids": [7, 11, 7, 7, 19],
            "evaluation_token_ids": [23, 7],
            "selections": [
                {"policy": "lru", "retained_positions": [0, 2, 4]},
                {"policy": "magnitude", "retained_positions": [1, 2, 4]},
            ],
        }
    )


def _metric(value):
    metric = BackendMetricValue(
        name="token_accuracy",
        kind="quality",
        unit="ratio",
        preference="higher_is_better",
        value=value,
    )
    metric.validate()
    return metric


class _SuccessfulBackend:
    def __init__(self):
        self.requests = []

    def execute(self, request):
        self.requests.append(request)
        mode = request["mode"]
        positions = tuple(request["retained_positions"])
        artifact = (
            b"baseline-artifact"
            if mode == "baseline"
            else f"candidate:{request['policy']}:{positions}".encode()
        )
        return BackendObservationV4(
            request_sha256="0" * 64,
            applied_mode=mode,
            applied_policy=request["policy"],
            applied_retained_positions=positions,
            artifact_bytes=artifact,
            artifact_sha256=hashlib.sha256(artifact).hexdigest(),
            metrics=(_metric(1.0 if mode == "baseline" else 0.75),),
        )


class _FailAfterBaselineBackend(_SuccessfulBackend):
    def execute(self, request):
        if self.requests:
            raise ProspectKvRealModelRunnerError("injected candidate failure")
        return super().execute(request)


class ProspectPositionCampaignV4Tests(unittest.TestCase):
    def test_spec_preserves_repeated_model_tokens_and_builds_position_selections(self):
        spec = PositionCampaignSpecV1.from_canonical_json(_spec_payload())
        context, trace, selections = spec.build_execution_inputs()

        self.assertEqual(trace.model_input_token_ids, (7, 11, 7, 7, 19))
        self.assertEqual(trace.evaluation_token_ids, (23, 7))
        self.assertEqual(context.trace_sha256, trace.sha256)
        self.assertEqual(selections[0].retained_positions, (0, 2, 4))
        self.assertEqual(selections[0].retained_token_ids, (7, 7, 19))
        self.assertEqual(selections[1].retained_token_ids, (11, 7, 19))
        self.assertEqual(spec.canonical_json(), _spec_payload())

    def test_successful_campaign_publishes_replayable_records_and_manifest(self):
        spec = PositionCampaignSpecV1.from_canonical_json(_spec_payload())
        backend = _SuccessfulBackend()

        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "evidence"
            manifest = execute_position_campaign_v4(
                spec=spec, backend=backend, output_dir=output
            )

            self.assertEqual(
                manifest.schema, PROSPECT_KV_POSITION_CAMPAIGN_RESULT_SCHEMA_V1
            )
            self.assertEqual(manifest.evidence_schema, PROSPECT_KV_REAL_MODEL_SELECTION_SCHEMA_V2)
            self.assertEqual(len(manifest.records), 2)
            self.assertTrue(output.is_dir())
            self.assertEqual(len(backend.requests), 3)
            self.assertEqual(
                backend.requests[0]["retained_positions"], [0, 1, 2, 3, 4]
            )
            self.assertEqual(backend.requests[1]["retained_positions"], [0, 2, 4])

            on_disk_manifest = (output / "manifest.json").read_text(encoding="utf-8")
            self.assertEqual(on_disk_manifest, manifest.canonical_json())
            for record in manifest.records:
                payload = (output / record.filename).read_text(encoding="utf-8")
                self.assertEqual(
                    hashlib.sha256(payload.encode("utf-8")).hexdigest(), record.sha256
                )
                replayed = ProspectKvRealModelSelectionEvidenceV2.from_canonical_json(
                    payload
                )
                self.assertEqual(replayed.selection.policy, record.policy)

    def test_backend_failure_leaves_no_observed_evidence_directory(self):
        spec = PositionCampaignSpecV1.from_canonical_json(_spec_payload())
        backend = _FailAfterBaselineBackend()

        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "evidence"
            with self.assertRaisesRegex(
                ProspectKvRealModelRunnerError, "injected candidate failure"
            ):
                execute_position_campaign_v4(
                    spec=spec, backend=backend, output_dir=output
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(parent).iterdir()), [])

    def test_existing_output_is_rejected_before_backend_execution(self):
        spec = PositionCampaignSpecV1.from_canonical_json(_spec_payload())
        backend = _SuccessfulBackend()

        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "evidence"
            output.mkdir()
            with self.assertRaisesRegex(
                ProspectKvPositionCampaignError, "already exists"
            ):
                execute_position_campaign_v4(
                    spec=spec, backend=backend, output_dir=output
                )
            self.assertEqual(backend.requests, [])

    def test_noncanonical_or_duplicate_policy_spec_fails_closed(self):
        raw = json.loads(_spec_payload())
        pretty = json.dumps(raw, indent=2, sort_keys=True)
        with self.assertRaisesRegex(ProspectKvPositionCampaignError, "canonical"):
            PositionCampaignSpecV1.from_canonical_json(pretty)

        raw["selections"][1]["policy"] = "lru"
        with self.assertRaisesRegex(ProspectKvPositionCampaignError, "duplicate"):
            PositionCampaignSpecV1.from_canonical_json(_canonical(raw))


if __name__ == "__main__":
    unittest.main()
