"""BKV-K8 first-token readiness observation contract.

The record in this module is deliberately evidence-only. It binds one target-host
observation to an exact retained BIKV evidence bundle and records first-token
versus steady-state timing plus Boolean/numerical traffic accounting without
claiming speedup, physical DRAM traffic, model quality, or adaptive-placement
readiness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import re
from typing import Any

from .bikv_traffic import LOGICAL_PACKED_PAYLOAD, BikvTrafficEvidence


BKV_K8_OBSERVATION_SCHEMA_V1 = "kvlab.bkv-k8-first-token-observation.v1"
_TIMING_SOURCES = frozenset({"host_wall_clock", "device_timestamp"})
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REPO_RE = re.compile(r"^[^/\s]+/[^/\s]+$")
_FIELDS = frozenset(
    {
        "schema",
        "evidence_bundle_sha256",
        "producer_repo",
        "producer_commit",
        "hardware_fingerprint_sha256",
        "timing_source",
        "first_token_latency_ns",
        "steady_state_latency_ns",
        "boolean_frontend_ns",
        "numerical_kv_bytes_avoided",
        "boolean_kv_bytes_read",
        "historical_signature_rebuilds",
        "first_token_boolean_route_consumed",
    }
)


class BikvFirstTokenObservationError(ValueError):
    """Raised when a BKV-K8 observation is malformed or non-canonical."""


def _require_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BikvFirstTokenObservationError(f"{name} must be a non-negative integer")
    return value


def _require_positive_int(name: str, value: object) -> int:
    value = _require_nonnegative_int(name, value)
    if value == 0:
        raise BikvFirstTokenObservationError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class BikvK8FirstTokenObservationV1:
    """Canonical evidence record for one BKV-K8 first-token campaign unit."""

    schema: str
    evidence_bundle_sha256: str
    producer_repo: str
    producer_commit: str
    hardware_fingerprint_sha256: str
    timing_source: str
    first_token_latency_ns: int
    steady_state_latency_ns: tuple[int, ...]
    boolean_frontend_ns: int
    numerical_kv_bytes_avoided: int
    boolean_kv_bytes_read: int
    historical_signature_rebuilds: int
    first_token_boolean_route_consumed: bool

    def validate(self) -> None:
        """Validate provenance and bounded integer observation fields fail-closed."""

        if self.schema != BKV_K8_OBSERVATION_SCHEMA_V1:
            raise BikvFirstTokenObservationError("unsupported BKV-K8 observation schema")
        if not isinstance(self.evidence_bundle_sha256, str) or not _SHA256_RE.fullmatch(
            self.evidence_bundle_sha256
        ):
            raise BikvFirstTokenObservationError(
                "evidence_bundle_sha256 must be a lowercase 64-hex digest"
            )
        if not isinstance(self.producer_repo, str) or not _REPO_RE.fullmatch(self.producer_repo):
            raise BikvFirstTokenObservationError("producer_repo must use owner/name form")
        if not isinstance(self.producer_commit, str) or not _SHA1_RE.fullmatch(self.producer_commit):
            raise BikvFirstTokenObservationError(
                "producer_commit must be a lowercase 40-hex Git SHA"
            )
        if not isinstance(self.hardware_fingerprint_sha256, str) or not _SHA256_RE.fullmatch(
            self.hardware_fingerprint_sha256
        ):
            raise BikvFirstTokenObservationError(
                "hardware_fingerprint_sha256 must be a lowercase 64-hex digest"
            )
        if self.timing_source not in _TIMING_SOURCES:
            raise BikvFirstTokenObservationError(
                "timing_source must be host_wall_clock or device_timestamp"
            )
        _require_positive_int("first_token_latency_ns", self.first_token_latency_ns)
        if not isinstance(self.steady_state_latency_ns, tuple) or not self.steady_state_latency_ns:
            raise BikvFirstTokenObservationError(
                "steady_state_latency_ns must be a non-empty tuple"
            )
        if len(self.steady_state_latency_ns) > 1_000_000:
            raise BikvFirstTokenObservationError("steady-state sample budget exceeded")
        for value in self.steady_state_latency_ns:
            _require_positive_int("steady_state_latency_ns entry", value)
        _require_nonnegative_int("boolean_frontend_ns", self.boolean_frontend_ns)
        if self.boolean_frontend_ns > self.first_token_latency_ns:
            raise BikvFirstTokenObservationError(
                "boolean_frontend_ns cannot exceed first_token_latency_ns"
            )
        _require_nonnegative_int(
            "numerical_kv_bytes_avoided", self.numerical_kv_bytes_avoided
        )
        _require_nonnegative_int("boolean_kv_bytes_read", self.boolean_kv_bytes_read)
        _require_nonnegative_int(
            "historical_signature_rebuilds", self.historical_signature_rebuilds
        )
        if not isinstance(self.first_token_boolean_route_consumed, bool):
            raise BikvFirstTokenObservationError(
                "first_token_boolean_route_consumed must be Boolean"
            )
        if self.first_token_boolean_route_consumed and self.historical_signature_rebuilds != 0:
            raise BikvFirstTokenObservationError(
                "first-token Boolean routing cannot claim readiness with historical signature rebuilds"
            )

    def traffic_evidence(self) -> BikvTrafficEvidence:
        """Bind the v1 byte fields to their declared logical-payload semantics.

        The v1 schema predates the generic typed traffic-evidence contract, but
        its byte-ratio documentation has always declared logical accounting.
        Converting through this method makes that evidence kind executable
        without changing the canonical v1 JSON schema.
        """

        self.validate()
        evidence = BikvTrafficEvidence(
            numerical_kv_bytes_avoided=self.numerical_kv_bytes_avoided,
            numerical_evidence_kind=LOGICAL_PACKED_PAYLOAD,
            boolean_kv_bytes_read=self.boolean_kv_bytes_read,
            boolean_evidence_kind=LOGICAL_PACKED_PAYLOAD,
        )
        evidence.validate()
        return evidence

    def numerical_to_boolean_bytes_ratio(self) -> Fraction | None:
        """Return exact logical avoided/read byte ratio, or ``None`` at zero denominator.

        This ratio is logical packed-payload accounting. It is not a physical
        DRAM-bandwidth, allocator-residency or transfer measurement. Canonical
        v1 records historically permit a zero Boolean-byte denominator, so that
        retained case remains an undefined ratio rather than being reclassified
        as invalid by the stricter typed evidence helper introduced later.
        """

        self.validate()
        if self.boolean_kv_bytes_read == 0:
            return None
        return self.traffic_evidence().numerical_bytes_avoided_per_boolean_byte

    def canonical_json(self) -> str:
        """Return deterministic JSON for content-addressed retention."""

        self.validate()
        payload = asdict(self)
        payload["steady_state_latency_ns"] = list(self.steady_state_latency_ns)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def observation_sha256(self) -> str:
        """Hash the canonical observation JSON, not any upstream evidence payload."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_canonical_json(cls, payload: str) -> "BikvK8FirstTokenObservationV1":
        """Decode only exact canonical v1 JSON."""

        if not isinstance(payload, str) or not payload:
            raise BikvFirstTokenObservationError("observation JSON must be non-empty text")
        try:
            decoded: Any = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise BikvFirstTokenObservationError("observation payload must be valid JSON") from exc
        if not isinstance(decoded, dict) or frozenset(decoded) != _FIELDS:
            raise BikvFirstTokenObservationError("observation JSON fields do not match v1 schema")
        steady = decoded["steady_state_latency_ns"]
        if not isinstance(steady, list):
            raise BikvFirstTokenObservationError("steady_state_latency_ns must be a JSON array")
        observation = cls(
            schema=decoded["schema"],
            evidence_bundle_sha256=decoded["evidence_bundle_sha256"],
            producer_repo=decoded["producer_repo"],
            producer_commit=decoded["producer_commit"],
            hardware_fingerprint_sha256=decoded["hardware_fingerprint_sha256"],
            timing_source=decoded["timing_source"],
            first_token_latency_ns=decoded["first_token_latency_ns"],
            steady_state_latency_ns=tuple(steady),
            boolean_frontend_ns=decoded["boolean_frontend_ns"],
            numerical_kv_bytes_avoided=decoded["numerical_kv_bytes_avoided"],
            boolean_kv_bytes_read=decoded["boolean_kv_bytes_read"],
            historical_signature_rebuilds=decoded["historical_signature_rebuilds"],
            first_token_boolean_route_consumed=decoded["first_token_boolean_route_consumed"],
        )
        observation.validate()
        if observation.canonical_json() != payload:
            raise BikvFirstTokenObservationError("observation JSON is valid but not canonical")
        return observation