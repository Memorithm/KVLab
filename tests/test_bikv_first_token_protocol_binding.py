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
from kvlab.bikv_first_token_v2 import (
    BKV_K8_OBSERVATION_SCHEMA_V2,
    BikvK8FirstTokenObservationV2,
)
from kvlab.bikv_target_protocol import (
    BASELINE_FULL_CACHE_NATIVE_PREFILL,
    BKV_TARGET_PROTOCOL_SCHEMA_V1,
    REQUIRED_METRICS,
    BikvTargetProtocolV1,
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


if __name__ == "__main__":
    unittest.main()
