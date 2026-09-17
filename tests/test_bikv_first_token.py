from __future__ import annotations

from fractions import Fraction
import json
import unittest

from kvlab.bikv_first_token import (
    BKV_K8_OBSERVATION_SCHEMA_V1,
    BikvFirstTokenObservationError,
    BikvK8FirstTokenObservationV1,
)


def _observation(**changes: object) -> BikvK8FirstTokenObservationV1:
    values: dict[str, object] = {
        "schema": BKV_K8_OBSERVATION_SCHEMA_V1,
        "evidence_bundle_sha256": "a" * 64,
        "producer_repo": "Memorithm/FLAT-ATTENTION",
        "producer_commit": "b" * 40,
        "hardware_fingerprint_sha256": "c" * 64,
        "timing_source": "device_timestamp",
        "first_token_latency_ns": 1_000,
        "steady_state_latency_ns": (800, 810, 790),
        "boolean_frontend_ns": 120,
        "numerical_kv_bytes_avoided": 4096,
        "boolean_kv_bytes_read": 256,
        "historical_signature_rebuilds": 0,
        "first_token_boolean_route_consumed": True,
    }
    values.update(changes)
    return BikvK8FirstTokenObservationV1(**values)  # type: ignore[arg-type]


class BikvFirstTokenObservationTests(unittest.TestCase):
    def test_canonical_round_trip_and_exact_ratio(self) -> None:
        observation = _observation()
        encoded = observation.canonical_json()
        decoded = BikvK8FirstTokenObservationV1.from_canonical_json(encoded)
        self.assertEqual(decoded, observation)
        self.assertEqual(decoded.numerical_to_boolean_bytes_ratio(), Fraction(16, 1))
        self.assertEqual(len(decoded.observation_sha256()), 64)

    def test_zero_boolean_bytes_has_no_defined_ratio(self) -> None:
        observation = _observation(
            boolean_kv_bytes_read=0,
            numerical_kv_bytes_avoided=0,
        )
        self.assertIsNone(observation.numerical_to_boolean_bytes_ratio())

    def test_boolean_first_token_readiness_rejects_historical_rebuild(self) -> None:
        with self.assertRaisesRegex(
            BikvFirstTokenObservationError, "historical signature rebuilds"
        ):
            _observation(historical_signature_rebuilds=1).validate()

    def test_non_boolean_route_flag_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            BikvFirstTokenObservationError, "must be Boolean"
        ):
            _observation(first_token_boolean_route_consumed=1).validate()

    def test_boolean_frontend_interval_cannot_exceed_first_token(self) -> None:
        with self.assertRaisesRegex(
            BikvFirstTokenObservationError, "cannot exceed"
        ):
            _observation(boolean_frontend_ns=1_001).validate()

    def test_invalid_timing_source_and_empty_steady_samples_are_rejected(self) -> None:
        with self.assertRaisesRegex(BikvFirstTokenObservationError, "timing_source"):
            _observation(timing_source="guessed_clock").validate()
        with self.assertRaisesRegex(BikvFirstTokenObservationError, "non-empty tuple"):
            _observation(steady_state_latency_ns=()).validate()

    def test_alternate_json_encoding_is_not_canonical(self) -> None:
        canonical = _observation().canonical_json()
        alternate = json.dumps(json.loads(canonical), sort_keys=False, indent=2)
        self.assertNotEqual(canonical, alternate)
        with self.assertRaisesRegex(BikvFirstTokenObservationError, "not canonical"):
            BikvK8FirstTokenObservationV1.from_canonical_json(alternate)

    def test_bool_cannot_alias_integer_fields(self) -> None:
        for field in (
            "first_token_latency_ns",
            "boolean_frontend_ns",
            "numerical_kv_bytes_avoided",
            "boolean_kv_bytes_read",
            "historical_signature_rebuilds",
        ):
            with self.subTest(field=field), self.assertRaises(
                BikvFirstTokenObservationError
            ):
                _observation(**{field: True}).validate()
