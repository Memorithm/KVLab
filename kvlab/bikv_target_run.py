"""Canonical per-attempt result retention for frozen BIKV target protocols.

The record is deliberately descriptive. It binds one baseline or candidate
attempt to an exact :class:`BikvTargetProtocolV1`, preserves every frozen metric
obligation as measured/not-exposed/failed evidence, and retains failed attempts.
It does not decide BKV-K9 promotion or infer performance from unavailable data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping

from .bikv_target_protocol import REQUIRED_METRICS, BikvTargetProtocolV1


BKV_TARGET_RUN_SCHEMA_V1 = "kvlab.bikv-target-run.v1"
RUN_VARIANTS = frozenset({"baseline", "candidate"})
RUN_STATUSES = frozenset({"completed", "failed"})
METRIC_STATUSES = frozenset({"measured", "not_exposed", "failed"})
_BOOLEAN_METRICS = frozenset({"reset_reuse_correctness"})
_SIGNED_NUMERIC_METRICS = frozenset({"downstream_quality"})
_INTEGER_METRICS = frozenset(
    {
        "boolean_bits_per_token",
        "boolean_bits_per_page",
        "boolean_index_bytes",
        "boolean_metadata_overhead_bytes",
        "numerical_kv_bytes_touched",
        "numerical_kv_bytes_avoided",
        "boolean_kv_bytes_read",
        "host_device_transfer_bytes",
        "numa_traffic_bytes_when_measurable",
        "fragmentation_bytes",
        "allocator_overhead_bytes",
        "query_signature_transfer_bytes",
        "candidate_bitmap_transfer_bytes",
        "synchronization_wait_ns",
        "dispatch_count",
        "backpressure_wait_ns",
        "boolean_search_latency_ns",
        "boolean_frontend_ns",
        "first_token_latency_ns",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_FIELDS = frozenset(
    {
        "schema",
        "protocol_sha256",
        "campaign_id",
        "attempt_id",
        "variant",
        "seed",
        "repetition_index",
        "status",
        "failure_reason",
        "metrics",
    }
)
_METRIC_FIELDS = frozenset({"name", "status", "value", "unit", "reason"})


class BikvTargetRunError(ValueError):
    """Raised when retained target-run evidence is malformed or inconsistent."""


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise BikvTargetRunError(f"{name} must be non-empty trimmed text")
    return value


def _enum(name: str, value: object, allowed: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise BikvTargetRunError(f"{name} must be one of {sorted(allowed)}")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BikvTargetRunError(f"{name} must be a non-negative integer")
    return value


def _measured_scalar(name: str, value: object) -> int | float | bool:
    if name in _BOOLEAN_METRICS:
        if not isinstance(value, bool):
            raise BikvTargetRunError(f"{name} measured value must be Boolean")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BikvTargetRunError(f"{name} measured value must be numeric")
    if name in _INTEGER_METRICS and not isinstance(value, int):
        raise BikvTargetRunError(f"{name} measured value must be an integer")
    if isinstance(value, float) and not math.isfinite(value):
        raise BikvTargetRunError(f"{name} measured value must be finite")
    if name not in _SIGNED_NUMERIC_METRICS and value < 0:
        raise BikvTargetRunError(f"{name} measured value must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class BikvMetricObservationV1:
    """One frozen metric obligation and its observed availability state."""

    name: str
    status: str
    value: int | float | bool | None
    unit: str | None
    reason: str | None

    def validate(self) -> None:
        if self.name not in REQUIRED_METRICS:
            raise BikvTargetRunError(f"unknown BIKV target metric: {self.name!r}")
        _enum("metric status", self.status, METRIC_STATUSES)
        if self.status == "measured":
            _measured_scalar(self.name, self.value)
            _text("metric unit", self.unit)
            if self.reason is not None:
                raise BikvTargetRunError("measured metric must not carry a reason")
            return
        if self.value is not None or self.unit is not None:
            raise BikvTargetRunError(
                f"{self.status} metric must not carry a value or unit"
            )
        _text("metric reason", self.reason)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BikvMetricObservationV1":
        if not isinstance(raw, Mapping) or frozenset(raw) != _METRIC_FIELDS:
            raise BikvTargetRunError("metric fields do not match v1 schema")
        metric = cls(
            name=raw["name"],
            status=raw["status"],
            value=raw["value"],
            unit=raw["unit"],
            reason=raw["reason"],
        )
        metric.validate()
        return metric


def _measured_metrics(
    metrics: tuple["BikvMetricObservationV1", ...],
) -> dict[str, int | float | bool]:
    return {
        metric.name: metric.value
        for metric in metrics
        if metric.status == "measured" and metric.value is not None
    }


def _validate_cross_metric_invariants(
    metrics: tuple["BikvMetricObservationV1", ...],
) -> None:
    measured = _measured_metrics(metrics)
    frontend = measured.get("boolean_frontend_ns")
    first_token = measured.get("first_token_latency_ns")
    if frontend is not None and first_token is not None and frontend > first_token:
        raise BikvTargetRunError(
            "boolean_frontend_ns cannot exceed first_token_latency_ns"
        )

    avoided = measured.get("numerical_kv_bytes_avoided")
    boolean_read = measured.get("boolean_kv_bytes_read")
    if avoided is not None and boolean_read is not None and avoided > 0 and boolean_read == 0:
        raise BikvTargetRunError(
            "non-zero numerical KV bytes avoided requires non-zero Boolean KV bytes read"
        )


@dataclass(frozen=True, slots=True)
class BikvTargetRunV1:
    """One content-addressed baseline/candidate attempt under a frozen protocol."""

    schema: str
    protocol_sha256: str
    campaign_id: str
    attempt_id: str
    variant: str
    seed: int
    repetition_index: int
    status: str
    failure_reason: str | None
    metrics: tuple[BikvMetricObservationV1, ...]

    def validate(self) -> None:
        if self.schema != BKV_TARGET_RUN_SCHEMA_V1:
            raise BikvTargetRunError("unsupported BIKV target-run schema")
        if not isinstance(self.protocol_sha256, str) or not _SHA256_RE.fullmatch(
            self.protocol_sha256
        ):
            raise BikvTargetRunError("protocol_sha256 must be lowercase 64-hex")
        _text("campaign_id", self.campaign_id)
        _text("attempt_id", self.attempt_id)
        _enum("variant", self.variant, RUN_VARIANTS)
        _nonnegative_int("seed", self.seed)
        _nonnegative_int("repetition_index", self.repetition_index)
        _enum("run status", self.status, RUN_STATUSES)

        if self.status == "completed":
            if self.failure_reason is not None:
                raise BikvTargetRunError("completed run must not carry failure_reason")
        else:
            _text("failure_reason", self.failure_reason)

        if not isinstance(self.metrics, tuple):
            raise BikvTargetRunError("metrics must be a tuple")
        metric_names = tuple(metric.name for metric in self.metrics)
        if metric_names != REQUIRED_METRICS:
            raise BikvTargetRunError(
                "metrics must exactly match the frozen required-metric order"
            )
        for metric in self.metrics:
            metric.validate()
        _validate_cross_metric_invariants(self.metrics)

        failed_metrics = [metric for metric in self.metrics if metric.status == "failed"]
        if self.status == "completed" and failed_metrics:
            raise BikvTargetRunError("completed run cannot contain failed metrics")
        if self.status == "failed" and not failed_metrics:
            raise BikvTargetRunError("failed run must retain at least one failed metric")

    def validate_against(self, protocol: BikvTargetProtocolV1) -> None:
        """Bind this attempt to exact protocol identity and repetition membership."""

        self.validate()
        protocol.validate()
        if self.protocol_sha256 != protocol.protocol_sha256():
            raise BikvTargetRunError("run protocol_sha256 does not match supplied protocol")
        if self.campaign_id != protocol.campaign_id:
            raise BikvTargetRunError("run campaign_id does not match supplied protocol")
        if self.seed not in protocol.seeds:
            raise BikvTargetRunError("run seed is not frozen by supplied protocol")
        if self.repetition_index >= protocol.repetitions:
            raise BikvTargetRunError("run repetition_index exceeds supplied protocol")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "protocol_sha256": self.protocol_sha256,
            "campaign_id": self.campaign_id,
            "attempt_id": self.attempt_id,
            "variant": self.variant,
            "seed": self.seed,
            "repetition_index": self.repetition_index,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "metrics": [metric.to_dict() for metric in self.metrics],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    def run_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BikvTargetRunV1":
        if not isinstance(raw, Mapping) or frozenset(raw) != _RUN_FIELDS:
            raise BikvTargetRunError("target-run fields do not match v1 schema")
        raw_metrics = raw["metrics"]
        if not isinstance(raw_metrics, list):
            raise BikvTargetRunError("metrics must be a JSON array")
        run = cls(
            schema=raw["schema"],
            protocol_sha256=raw["protocol_sha256"],
            campaign_id=raw["campaign_id"],
            attempt_id=raw["attempt_id"],
            variant=raw["variant"],
            seed=raw["seed"],
            repetition_index=raw["repetition_index"],
            status=raw["status"],
            failure_reason=raw["failure_reason"],
            metrics=tuple(BikvMetricObservationV1.from_mapping(item) for item in raw_metrics),
        )
        run.validate()
        return run

    @classmethod
    def from_canonical_json(cls, payload: str) -> "BikvTargetRunV1":
        if not isinstance(payload, str) or not payload:
            raise BikvTargetRunError("target-run JSON must be non-empty text")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise BikvTargetRunError("target-run payload must be valid JSON") from exc
        run = cls.from_mapping(decoded)
        if run.canonical_json() != payload:
            raise BikvTargetRunError("target-run JSON is valid but not canonical")
        return run
