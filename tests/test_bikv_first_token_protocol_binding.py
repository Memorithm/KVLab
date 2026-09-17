from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from kvlab.bikv_evidence_bundle import BikvEvidenceBundleV1
from kvlab.bikv_evidence_receipt import BikvEvidenceReceiptV1
from kvlab.bikv_first_token import BikvFirstTokenObservationError
from kvlab.bikv_first_token_campaign import (
    BKV_K8_CAMPAIGN_BINDING_SCHEMA_V1,
    BikvK8CampaignBindingError,
    BikvK8FirstTokenCampaignBindingV1,
)
from kvlab.bikv_first_token_v2 import (
    BKV_K8_OBSERVATION_SCHEMA_V2,
    BikvK8FirstTokenObservationV2,
)
from kvlab.bikv_target_campaign import BikvTargetCampaignV1
from kvlab.bikv_target_protocol import (
    BASELINE_FULL_CACHE_NATIVE_PREFILL,
    BKV_TARGET_PROTOCOL_SCHEMA_V1,
    REQUIRED_METRICS,
    BikvTargetProtocolV1,
)
from kvlab.bikv_target_run import (
    BKV_TARGET_RUN_SCHEMA_V1,
    BikvMetricObservationV1,
    BikvTargetRunV1,
)
from kvlab.bikv_traffic import LOGICAL_PACKED_PAYLOAD, PHYSICAL_DRAM_COUNTER


FLAT_COMMIT = "1" * 40
HARDWARE_SHA256 = "2" * 64


def _bundle() -> BikvEvidenceBundleV1:
    receipt = BikvEvidenceReceiptV1.from_json_bytes(
        producer_repo="Memorithm/FLAT-ATTENTION",
        producer_commit=FLAT_COMMIT,
        payload=b'{"schema":"flat.bikv-test-evidence.v1"}',
    )
    return BikvEvidenceBundleV1.from_receipts((("candidate", receipt),))


def _protocol(bundle: BikvEvidenceBundleV1, **changes: object) -> BikvTargetProtocolV1:
    values: dict[str, object] = {
        "schema": BKV_TARGET_PROTOCOL_SCHEMA_V1,
        "campaign_id": "bkv-k8-binding-test",
        "phase": "development",
        "hypothesis_h0": "No first-token benefit under the frozen protocol",
        "hypothesis_h1": "The Boolean route improves the preregistered metric",
        "evidence_bundle_sha256": bundle.bundle_sha256(),
        "kvlab_commit": "3" * 40,
        "flat_commit": FLAT_COMMIT,
        "model_id": "model",
        "model_revision": "4" * 40,
        "tokenizer_id": "tokenizer",
        "tokenizer_revision": "5" * 40,
        "runtime_id": "runtime",
        "runtime_revision": "6" * 40,
        "hardware_fingerprint_sha256": HARDWARE_SHA256,
        "precision": "f32",
        "context_tokens": 128,
        "batch_size": 1,
        "dataset_id": "dataset",
        "dataset_revision": "7" * 40,
        "partition_id": "development",
        "partition_role": "development",
        "tuning_permitted": True,
        "seeds": (1,),
        "warmup_runs": 1,
        "repetitions": 2,
        "boolean_policy": "frozen-test-policy",
        "baseline_policy": BASELINE_FULL_CACHE_NATIVE_PREFILL,
        "timing_source": "device_timestamp",
        "byte_evidence_kind": LOGICAL_PACKED_PAYLOAD,
        "quality_metric": "exact-parity",
        "quality_rule": "must pass",
        "holdout_policy": "protected holdout remains closed",
        "required_metrics": REQUIRED_METRICS,
    }
    values.update(changes)
    return BikvTargetProtocolV1(**values)  # type: ignore[arg-type]


def _observation(bundle: BikvEvidenceBundleV1, **changes: object) -> BikvK8FirstTokenObservationV2:
    values: dict[str, object] = {
        "schema": BKV_K8_OBSERVATION_SCHEMA_V2,
        "evidence_bundle_sha256": bundle.bundle_sha256(),
        "producer_repo": "Memorithm/FLAT-ATTENTION",
        "producer_commit": FLAT_COMMIT,
        "hardware_fingerprint_sha256": HARDWARE_SHA256,
        "timing_source": "device_timestamp",
        "first_token_latency_ns": 1_000,
        "steady_state_latency_ns": (800, 810),
        "boolean_frontend_ns": 100,
        "numerical_kv_bytes_avoided": 4096,
        "numerical_evidence_kind": LOGICAL_PACKED_PAYLOAD,
        "boolean_kv_bytes_read": 256,
        "boolean_evidence_kind": LOGICAL_PACKED_PAYLOAD,
        "historical_signature_rebuilds": 0,
        "first_token_boolean_route_consumed": True,
    }
    values.update(changes)
    return BikvK8FirstTokenObservationV2(**values)  # type: ignore[arg-type]


def _run(protocol: BikvTargetProtocolV1, observation: BikvK8FirstTokenObservationV2, **changes: object) -> BikvTargetRunV1:
    shared = {
        "first_token_latency_ns": (observation.first_token_latency_ns, "ns"),
        "boolean_frontend_ns": (observation.boolean_frontend_ns, "ns"),
        "numerical_kv_bytes_avoided": (observation.numerical_kv_bytes_avoided, "bytes"),
        "boolean_kv_bytes_read": (observation.boolean_kv_bytes_read, "bytes"),
    }
    metrics = []
    for name in REQUIRED_METRICS:
        if name in shared:
            value, unit = shared[name]
            metrics.append(BikvMetricObservationV1(name, "measured", value, unit, None))
        else:
            metrics.append(BikvMetricObservationV1(name, "not_exposed", None, None, "not exposed by fixture"))
    values: dict[str, object] = {
        "schema": BKV_TARGET_RUN_SCHEMA_V1,
        "protocol_sha256": protocol.protocol_sha256(),
        "campaign_id": protocol.campaign_id,
        "attempt_id": "candidate-seed1-r0",
        "variant": "candidate",
        "seed": 1,
        "repetition_index": 0,
        "status": "completed",
        "failure_reason": None,
        "metrics": tuple(metrics),
    }
    values.update(changes)
    return BikvTargetRunV1(**values)  # type: ignore[arg-type]


def _campaign_runs(
    protocol: BikvTargetProtocolV1, observation: BikvK8FirstTokenObservationV2
) -> tuple[BikvTargetRunV1, ...]:
    return (
        _run(protocol, observation, attempt_id="baseline-seed1-r0", variant="baseline", repetition_index=0),
        _run(protocol, observation),
        _run(protocol, observation, attempt_id="baseline-seed1-r1", variant="baseline", repetition_index=1),
        _run(protocol, observation, attempt_id="candidate-seed1-r1", repetition_index=1),
    )


class BikvFirstTokenProtocolBindingTests(unittest.TestCase):
    def test_exact_bundle_protocol_and_observation_bind(self) -> None:
        bundle = _bundle()
        protocol = _protocol(bundle)
        observation = _observation(bundle)
        observation.validate_against(evidence_bundle=bundle, protocol=protocol)

    def test_bundle_substitution_fails_closed(self) -> None:
        bundle = _bundle()
        protocol = _protocol(bundle)
        foreign_receipt = BikvEvidenceReceiptV1.from_json_bytes(
            producer_repo="Memorithm/FLAT-ATTENTION",
            producer_commit=FLAT_COMMIT,
            payload=b'{"schema":"foreign"}',
        )
        foreign_bundle = BikvEvidenceBundleV1.from_receipts((("candidate", foreign_receipt),))
        with self.assertRaisesRegex(BikvFirstTokenObservationError, "retained evidence bundle"):
            _observation(bundle).validate_against(
                evidence_bundle=foreign_bundle,
                protocol=protocol,
            )

    def test_producer_commit_must_match_frozen_flat_revision(self) -> None:
        bundle = _bundle()
        protocol = _protocol(bundle)
        with self.assertRaisesRegex(BikvFirstTokenObservationError, "frozen FLAT revision"):
            _observation(bundle, producer_commit="8" * 40).validate_against(
                evidence_bundle=bundle,
                protocol=protocol,
            )

    def test_producer_identity_must_be_retained_in_bundle(self) -> None:
        bundle = _bundle()
        protocol = _protocol(bundle)
        with self.assertRaisesRegex(BikvFirstTokenObservationError, "not retained in evidence bundle"):
            _observation(bundle, producer_repo="Memorithm/other-producer").validate_against(
                evidence_bundle=bundle,
                protocol=protocol,
            )

    def test_hardware_or_timing_drift_fails_closed(self) -> None:
        bundle = _bundle()
        protocol = _protocol(bundle)
        with self.assertRaisesRegex(BikvFirstTokenObservationError, "hardware fingerprint"):
            _observation(bundle, hardware_fingerprint_sha256="8" * 64).validate_against(
                evidence_bundle=bundle,
                protocol=protocol,
            )
        with self.assertRaisesRegex(BikvFirstTokenObservationError, "timing source"):
            _observation(bundle, timing_source="host_wall_clock").validate_against(
                evidence_bundle=bundle,
                protocol=protocol,
            )

    def test_byte_evidence_kind_must_match_frozen_protocol(self) -> None:
        bundle = _bundle()
        protocol = _protocol(bundle, byte_evidence_kind=PHYSICAL_DRAM_COUNTER)
        with self.assertRaisesRegex(BikvFirstTokenObservationError, "byte evidence kind"):
            _observation(bundle).validate_against(
                evidence_bundle=bundle,
                protocol=protocol,
            )

    def test_target_run_binding_requires_exact_shared_metrics(self) -> None:
        bundle = _bundle()
        protocol = _protocol(bundle)
        observation = _observation(bundle)
        run = _run(protocol, observation)
        observation.validate_against_target_run(
            evidence_bundle=bundle,
            protocol=protocol,
            run=run,
        )

        metrics = list(run.metrics)
        index = REQUIRED_METRICS.index("first_token_latency_ns")
        metrics[index] = BikvMetricObservationV1(
            "first_token_latency_ns", "measured", 1001, "ns", None
        )
        drifted = _run(protocol, observation, metrics=tuple(metrics))
        with self.assertRaisesRegex(BikvFirstTokenObservationError, "does not match"):
            observation.validate_against_target_run(
                evidence_bundle=bundle,
                protocol=protocol,
                run=drifted,
            )

    def test_target_run_binding_rejects_noncandidate_or_unmeasured_metric(self) -> None:
        bundle = _bundle()
        protocol = _protocol(bundle)
        observation = _observation(bundle)
        with self.assertRaisesRegex(BikvFirstTokenObservationError, "candidate target run"):
            observation.validate_against_target_run(
                evidence_bundle=bundle,
                protocol=protocol,
                run=_run(protocol, observation, variant="baseline"),
            )
        run = _run(protocol, observation)
        metrics = list(run.metrics)
        index = REQUIRED_METRICS.index("boolean_kv_bytes_read")
        metrics[index] = BikvMetricObservationV1(
            "boolean_kv_bytes_read", "not_exposed", None, None, "counter unavailable"
        )
        with self.assertRaisesRegex(BikvFirstTokenObservationError, "must be measured"):
            observation.validate_against_target_run(
                evidence_bundle=bundle,
                protocol=protocol,
                run=_run(protocol, observation, metrics=tuple(metrics)),
            )

    def test_target_run_binding_cli_emits_content_identities(self) -> None:
        bundle = _bundle()
        protocol = _protocol(bundle)
        observation = _observation(bundle)
        run = _run(protocol, observation)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle_path = root / "bundle.json"
            protocol_path = root / "protocol.json"
            run_path = root / "run.json"
            observation_path = root / "observation.json"
            bundle_path.write_bytes(bundle.canonical_json_bytes())
            protocol_path.write_text(protocol.canonical_json(), encoding="utf-8")
            run_path.write_text(run.canonical_json(), encoding="utf-8")
            observation_path.write_text(observation.canonical_json(), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "tools/verify_bikv_first_token_target_run.py",
                    str(bundle_path),
                    str(protocol_path),
                    str(run_path),
                    str(observation_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = json.loads(completed.stdout)
            self.assertEqual(output["run_sha256"], run.run_sha256())
            self.assertEqual(output["observation_sha256"], observation.observation_sha256())
            self.assertEqual(output["attempt_id"], run.attempt_id)

    def test_cli_emits_only_verified_identity_and_observation_fields(self) -> None:
        bundle = _bundle()
        protocol = _protocol(bundle)
        observation = _observation(bundle)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle_path = root / "bundle.json"
            protocol_path = root / "protocol.json"
            observation_path = root / "observation.json"
            bundle_path.write_bytes(bundle.canonical_json_bytes())
            protocol_path.write_text(protocol.canonical_json(), encoding="utf-8")
            observation_path.write_text(observation.canonical_json(), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "tools/verify_bikv_first_token_observation.py",
                    str(bundle_path),
                    str(protocol_path),
                    str(observation_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = json.loads(completed.stdout)
            self.assertEqual(output["observation_sha256"], observation.observation_sha256())
            self.assertEqual(output["protocol_sha256"], protocol.protocol_sha256())
            self.assertEqual(output["numerical_to_boolean_bytes_ratio"], "16/1")

    def test_cli_rejects_protocol_drift(self) -> None:
        bundle = _bundle()
        protocol = _protocol(bundle, timing_source="host_wall_clock")
        observation = _observation(bundle)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle_path = root / "bundle.json"
            protocol_path = root / "protocol.json"
            observation_path = root / "observation.json"
            bundle_path.write_bytes(bundle.canonical_json_bytes())
            protocol_path.write_text(protocol.canonical_json(), encoding="utf-8")
            observation_path.write_text(observation.canonical_json(), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "tools/verify_bikv_first_token_observation.py",
                    str(bundle_path),
                    str(protocol_path),
                    str(observation_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("timing source", completed.stderr)

    def test_campaign_binding_requires_complete_retained_campaign_payloads(self) -> None:
        bundle = _bundle()
        protocol = _protocol(bundle)
        observation = _observation(bundle)
        runs = _campaign_runs(protocol, observation)
        campaign = BikvTargetCampaignV1.from_runs(protocol=protocol, runs=runs)
        candidate_run = runs[1]

        binding = BikvK8FirstTokenCampaignBindingV1.from_evidence(
            evidence_bundle=bundle,
            protocol=protocol,
            campaign=campaign,
            retained_runs=runs,
            candidate_run=candidate_run,
            observation=observation,
        )
        self.assertEqual(binding.schema, BKV_K8_CAMPAIGN_BINDING_SCHEMA_V1)
        self.assertEqual(binding.campaign_sha256, campaign.campaign_sha256())
        self.assertEqual(binding.run_sha256, candidate_run.run_sha256())
        self.assertEqual(binding.observation_sha256, observation.observation_sha256())
        self.assertEqual(
            BikvK8FirstTokenCampaignBindingV1.from_canonical_json(binding.canonical_json()),
            binding,
        )

        with self.assertRaisesRegex(
            BikvK8CampaignBindingError, "campaign slots do not match frozen protocol"
        ):
            BikvK8FirstTokenCampaignBindingV1.from_evidence(
                evidence_bundle=bundle,
                protocol=protocol,
                campaign=campaign,
                retained_runs=runs[:-1],
                candidate_run=candidate_run,
                observation=observation,
            )

    def test_campaign_binding_rejects_candidate_payload_not_in_manifest(self) -> None:
        bundle = _bundle()
        protocol = _protocol(bundle)
        observation = _observation(bundle)
        runs = _campaign_runs(protocol, observation)
        campaign = BikvTargetCampaignV1.from_runs(protocol=protocol, runs=runs)
        substituted = _run(
            protocol,
            observation,
            attempt_id="candidate-seed1-r0-substituted",
        )
        with self.assertRaisesRegex(
            BikvK8CampaignBindingError, "not retained in the supplied campaign manifest"
        ):
            BikvK8FirstTokenCampaignBindingV1.from_evidence(
                evidence_bundle=bundle,
                protocol=protocol,
                campaign=campaign,
                retained_runs=runs,
                candidate_run=substituted,
                observation=observation,
            )

    def test_campaign_binding_canonical_parser_rejects_duplicate_keys(self) -> None:
        payload = (
            '{"schema":"kvlab.bkv-k8-first-token-campaign-binding.v1",'
            '"schema":"kvlab.bkv-k8-first-token-campaign-binding.v1"}'
        )
        with self.assertRaisesRegex(BikvK8CampaignBindingError, "duplicate JSON key"):
            BikvK8FirstTokenCampaignBindingV1.from_canonical_json(payload)

    def test_campaign_binding_cli_emits_content_addressed_receipt(self) -> None:
        bundle = _bundle()
        protocol = _protocol(bundle)
        observation = _observation(bundle)
        runs = _campaign_runs(protocol, observation)
        campaign = BikvTargetCampaignV1.from_runs(protocol=protocol, runs=runs)
        candidate_run = runs[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle_path = root / "bundle.json"
            protocol_path = root / "protocol.json"
            campaign_path = root / "campaign.json"
            candidate_path = root / "candidate.json"
            observation_path = root / "observation.json"
            bundle_path.write_bytes(bundle.canonical_json_bytes())
            protocol_path.write_text(protocol.canonical_json(), encoding="utf-8")
            campaign_path.write_bytes(campaign.canonical_json_bytes())
            candidate_path.write_text(candidate_run.canonical_json(), encoding="utf-8")
            observation_path.write_text(observation.canonical_json(), encoding="utf-8")
            run_paths = []
            for index, run in enumerate(runs):
                path = root / f"run-{index}.json"
                path.write_text(run.canonical_json(), encoding="utf-8")
                run_paths.append(path)

            command = [
                sys.executable,
                "tools/verify_bikv_first_token_campaign.py",
                str(bundle_path),
                str(protocol_path),
                str(campaign_path),
                str(candidate_path),
                str(observation_path),
            ] + [str(path) for path in run_paths]
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = json.loads(completed.stdout)
            self.assertEqual(output["campaign_sha256"], campaign.campaign_sha256())
            self.assertEqual(output["run_sha256"], candidate_run.run_sha256())
            self.assertEqual(output["observation_sha256"], observation.observation_sha256())
            self.assertRegex(output["binding_sha256"], r"^[0-9a-f]{64}$")



if __name__ == "__main__":
    unittest.main()
