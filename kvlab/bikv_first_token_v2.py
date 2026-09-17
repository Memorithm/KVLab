"""BKV-K8 first-token observation v2 with explicit byte-evidence kinds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
from typing import Any

from .bikv_first_token import (
    BKV_K8_OBSERVATION_SCHEMA_V1,
    BikvFirstTokenObservationError,
    BikvK8FirstTokenObservationV1,
)
from .bikv_evidence_bundle import BikvEvidenceBundleV1
from .bikv_target_protocol import BikvTargetProtocolV1
from .bikv_target_run import BikvTargetRunError, BikvTargetRunV1
from .bikv_traffic import BikvTrafficEvidence, BikvTrafficEvidenceError


BKV_K8_OBSERVATION_SCHEMA_V2 = "kvlab.bkv-k8-first-token-observation.v2"
_FIELDS_V2 = frozenset(
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
        "numerical_evidence_kind",
        "boolean_kv_bytes_read",
        "boolean_evidence_kind",
        "historical_signature_rebuilds",
        "first_token_boolean_route_consumed",
    }
)


@dataclass(frozen=True, slots=True)
class BikvK8FirstTokenObservationV2:
    """Canonical K8 record whose byte ratio is evidence-kind explicit."""

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
    numerical_evidence_kind: str
    boolean_kv_bytes_read: int
    boolean_evidence_kind: str
    historical_signature_rebuilds: int
    first_token_boolean_route_consumed: bool

    def _v1_common_fields(self) -> BikvK8FirstTokenObservationV1:
        return BikvK8FirstTokenObservationV1(
            schema=BKV_K8_OBSERVATION_SCHEMA_V1,
            evidence_bundle_sha256=self.evidence_bundle_sha256,
            producer_repo=self.producer_repo,
            producer_commit=self.producer_commit,
            hardware_fingerprint_sha256=self.hardware_fingerprint_sha256,
            timing_source=self.timing_source,
            first_token_latency_ns=self.first_token_latency_ns,
            steady_state_latency_ns=self.steady_state_latency_ns,
            boolean_frontend_ns=self.boolean_frontend_ns,
            numerical_kv_bytes_avoided=self.numerical_kv_bytes_avoided,
            boolean_kv_bytes_read=self.boolean_kv_bytes_read,
            historical_signature_rebuilds=self.historical_signature_rebuilds,
            first_token_boolean_route_consumed=self.first_token_boolean_route_consumed,
        )

    def traffic_evidence(self) -> BikvTrafficEvidence:
        evidence = BikvTrafficEvidence(
            numerical_kv_bytes_avoided=self.numerical_kv_bytes_avoided,
            numerical_evidence_kind=self.numerical_evidence_kind,
            boolean_kv_bytes_read=self.boolean_kv_bytes_read,
            boolean_evidence_kind=self.boolean_evidence_kind,
        )
        evidence.validate()
        return evidence

    def validate(self) -> None:
        if self.schema != BKV_K8_OBSERVATION_SCHEMA_V2:
            raise BikvFirstTokenObservationError("unsupported BKV-K8 v2 observation schema")
        self._v1_common_fields().validate()
        try:
            self.traffic_evidence()
        except BikvTrafficEvidenceError as exc:
            raise BikvFirstTokenObservationError(str(exc)) from exc

    def validate_against(
        self,
        *,
        evidence_bundle: BikvEvidenceBundleV1,
        protocol: BikvTargetProtocolV1,
    ) -> None:
        """Bind one K8 observation to the exact preregistered campaign inputs.

        This check is provenance-only. It does not decide whether the observed
        latency or byte counters are scientifically favorable.
        """

        self.validate()
        evidence_bundle.validate()
        protocol.validate()

        bundle_sha256 = evidence_bundle.bundle_sha256()
        if self.evidence_bundle_sha256 != bundle_sha256:
            raise BikvFirstTokenObservationError(
                "observation evidence_bundle_sha256 does not match retained evidence bundle"
            )
        if protocol.evidence_bundle_sha256 != bundle_sha256:
            raise BikvFirstTokenObservationError(
                "protocol evidence_bundle_sha256 does not match retained evidence bundle"
            )
        if self.producer_commit != protocol.flat_commit:
            raise BikvFirstTokenObservationError(
                "observation producer commit does not match frozen FLAT revision"
            )
        if not any(
            entry.producer_repo == self.producer_repo
            and entry.producer_commit == self.producer_commit
            for entry in evidence_bundle.entries
        ):
            raise BikvFirstTokenObservationError(
                "observation producer identity is not retained in evidence bundle"
            )
        if self.hardware_fingerprint_sha256 != protocol.hardware_fingerprint_sha256:
            raise BikvFirstTokenObservationError(
                "observation hardware fingerprint does not match frozen target protocol"
            )
        if self.timing_source != protocol.timing_source:
            raise BikvFirstTokenObservationError(
                "observation timing source does not match frozen target protocol"
            )
        if self.numerical_evidence_kind != protocol.byte_evidence_kind:
            raise BikvFirstTokenObservationError(
                "numerical byte evidence kind does not match frozen target protocol"
            )
        if self.boolean_evidence_kind != protocol.byte_evidence_kind:
            raise BikvFirstTokenObservationError(
                "Boolean byte evidence kind does not match frozen target protocol"
            )

    def validate_against_target_run(
        self,
        *,
        evidence_bundle: BikvEvidenceBundleV1,
        protocol: BikvTargetProtocolV1,
        run: BikvTargetRunV1,
    ) -> None:
        """Bind this K8 observation to one exact completed candidate attempt.

        The shared first-token/Boolean-byte metrics must be explicitly measured
        by the retained target run with canonical units and must equal this
        observation exactly.  This is a provenance consistency check, not an
        estimator for TPOT, quality, or a performance decision.
        """

        self.validate_against(evidence_bundle=evidence_bundle, protocol=protocol)
        try:
            run.validate_against(protocol)
        except BikvTargetRunError as exc:
            raise BikvFirstTokenObservationError(str(exc)) from exc
        if run.variant != "candidate":
            raise BikvFirstTokenObservationError(
                "BKV-K8 first-token observation must bind to a candidate target run"
            )
        if run.status != "completed":
            raise BikvFirstTokenObservationError(
                "BKV-K8 first-token observation must bind to a completed target run"
            )

        measured = {metric.name: metric for metric in run.metrics}
        expected = {
            "first_token_latency_ns": (self.first_token_latency_ns, "ns"),
            "boolean_frontend_ns": (self.boolean_frontend_ns, "ns"),
            "numerical_kv_bytes_avoided": (self.numerical_kv_bytes_avoided, "bytes"),
            "boolean_kv_bytes_read": (self.boolean_kv_bytes_read, "bytes"),
        }
        for name, (value, unit) in expected.items():
            metric = measured[name]
            if metric.status != "measured":
                raise BikvFirstTokenObservationError(
                    f"target run metric {name} must be measured for BKV-K8 binding"
                )
            if metric.unit != unit:
                raise BikvFirstTokenObservationError(
                    f"target run metric {name} must use canonical unit {unit}"
                )
            if metric.value != value:
                raise BikvFirstTokenObservationError(
                    f"target run metric {name} does not match BKV-K8 observation"
                )

    def numerical_to_boolean_bytes_ratio(self) -> Fraction | None:
        """Return an exact ratio only for like-for-like byte evidence."""

        self.validate()
        return self.traffic_evidence().numerical_bytes_avoided_per_boolean_byte

    def canonical_json(self) -> str:
        self.validate()
        payload = asdict(self)
        payload["steady_state_latency_ns"] = list(self.steady_state_latency_ns)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def observation_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_canonical_json(cls, payload: str) -> "BikvK8FirstTokenObservationV2":
        if not isinstance(payload, str) or not payload:
            raise BikvFirstTokenObservationError("observation JSON must be non-empty text")
        try:
            decoded: Any = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise BikvFirstTokenObservationError("observation payload must be valid JSON") from exc
        if not isinstance(decoded, dict) or frozenset(decoded) != _FIELDS_V2:
            raise BikvFirstTokenObservationError("observation JSON fields do not match v2 schema")
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
            numerical_evidence_kind=decoded["numerical_evidence_kind"],
            boolean_kv_bytes_read=decoded["boolean_kv_bytes_read"],
            boolean_evidence_kind=decoded["boolean_evidence_kind"],
            historical_signature_rebuilds=decoded["historical_signature_rebuilds"],
            first_token_boolean_route_consumed=decoded["first_token_boolean_route_consumed"],
        )
        observation.validate()
        if observation.canonical_json() != payload:
            raise BikvFirstTokenObservationError("observation JSON is valid but not canonical")
        return observation
