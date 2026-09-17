from fractions import Fraction
import unittest

from kvlab.bikv_first_token import BikvFirstTokenObservationError
from kvlab.bikv_first_token_v2 import (
    BKV_K8_OBSERVATION_SCHEMA_V2,
    BikvK8FirstTokenObservationV2,
)
from kvlab.bikv_traffic import LOGICAL_PACKED_PAYLOAD, PHYSICAL_DRAM_COUNTER


def _observation(**changes: object) -> BikvK8FirstTokenObservationV2:
    values: dict[str, object] = {
        "schema": BKV_K8_OBSERVATION_SCHEMA_V2,
        "evidence_bundle_sha256": "a" * 64,
        "producer_repo": "Memorithm/FLAT-ATTENTION",
        "producer_commit": "b" * 40,
        "hardware_fingerprint_sha256": "c" * 64,
        "timing_source": "device_timestamp",
        "first_token_latency_ns": 1_000,
        "steady_state_latency_ns": (800, 810, 790),
        "boolean_frontend_ns": 120,
        "numerical_kv_bytes_avoided": 4096,
        "numerical_evidence_kind": LOGICAL_PACKED_PAYLOAD,
        "boolean_kv_bytes_read": 256,
        "boolean_evidence_kind": LOGICAL_PACKED_PAYLOAD,
        "historical_signature_rebuilds": 0,
        "first_token_boolean_route_consumed": True,
    }
    values.update(changes)
    return BikvK8FirstTokenObservationV2(**values)  # type: ignore[arg-type]


class BikvFirstTokenObservationV2Tests(unittest.TestCase):
    def test_canonical_round_trip_retains_evidence_kind(self) -> None:
        observation = _observation()
        decoded = BikvK8FirstTokenObservationV2.from_canonical_json(observation.canonical_json())
        self.assertEqual(decoded, observation)
        self.assertEqual(decoded.traffic_evidence().evidence_kind, LOGICAL_PACKED_PAYLOAD)
        self.assertEqual(decoded.numerical_to_boolean_bytes_ratio(), Fraction(16, 1))

    def test_mixed_evidence_kinds_fail_closed_before_serialization(self) -> None:
        with self.assertRaisesRegex(BikvFirstTokenObservationError, "same evidence kind"):
            _observation(boolean_evidence_kind=PHYSICAL_DRAM_COUNTER).canonical_json()

    def test_like_for_like_physical_counter_ratio_is_labelled(self) -> None:
        observation = _observation(
            numerical_kv_bytes_avoided=600,
            numerical_evidence_kind=PHYSICAL_DRAM_COUNTER,
            boolean_kv_bytes_read=100,
            boolean_evidence_kind=PHYSICAL_DRAM_COUNTER,
        )
        self.assertEqual(observation.traffic_evidence().evidence_kind, PHYSICAL_DRAM_COUNTER)
        self.assertEqual(observation.numerical_to_boolean_bytes_ratio(), Fraction(6, 1))


if __name__ == "__main__":
    unittest.main()
