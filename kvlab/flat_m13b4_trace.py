"""Fail-closed consumer for FLAT-ATTENTION M13B.4 trace evidence.

KVLab owns measurement/evidence interpretation for the Boolean KV programme, while
FLAT-ATTENTION owns the producer-side M13B.4 trace contract.  This module accepts
only the canonical ``flat.m13b4-trace.v1`` byte encoding emitted by the pinned
FLAT producer revision and mirrors its structural/order validation without
inferring overlap, concurrency, traffic, latency improvement, or quality.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

FLAT_M13B4_TRACE_SCHEMA = "flat.m13b4-trace.v1"
FLAT_M13B4_REFERENCE_REVISION = "29c18275b5687b71599ef11f5badcc4571a52a41"

_TIMING_SOURCES = {"host_wall_clock", "device_timestamp"}
_SCHEDULING_VARIANTS = {
    "serial_matched",
    "multi_dispatch_overlap_candidate",
    "same_dispatch_fused_candidate",
}
_SCOPES = {"prefill", "first_decode", "steady_state_decode"}
_EVENT_KINDS = {
    "query_representation_ready",
    "q_signature_start",
    "q_signature_end",
    "boolean_routing_start",
    "boolean_routing_end",
    "survivor_metadata_ready",
    "numerical_kv_staging_start",
    "numerical_attention_start",
    "numerical_attention_end",
    "synchronization_wait_start",
    "synchronization_wait_end",
    "output_ready",
    "numerical_kv_commit",
    "boolean_signature_commit",
    "decode_visible",
}
_TRACE_FIELDS = {
    "schema",
    "timing_source",
    "scheduling_variant",
    "scope",
    "events",
}
_EVENT_FIELDS = {"kind", "timestamp_ns"}
_U64_MAX = (1 << 64) - 1
_DECODE_REQUIRED = (
    "query_representation_ready",
    "q_signature_start",
    "q_signature_end",
    "boolean_routing_start",
    "boolean_routing_end",
    "survivor_metadata_ready",
    "numerical_kv_staging_start",
    "numerical_attention_start",
    "numerical_attention_end",
    "output_ready",
)


class FlatM13B4TraceError(ValueError):
    """Raised when producer evidence does not satisfy the pinned FLAT contract."""


@dataclass(frozen=True, slots=True)
class FlatM13B4TraceEventV1:
    """One timestamped event from ``flat.m13b4-trace.v1``."""

    kind: str
    timestamp_ns: int

    def validate(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in _EVENT_KINDS:
            raise FlatM13B4TraceError(f"unsupported M13B.4 event kind: {self.kind!r}")
        if isinstance(self.timestamp_ns, bool) or not isinstance(self.timestamp_ns, int):
            raise FlatM13B4TraceError("timestamp_ns must be an integer u64")
        if not 0 <= self.timestamp_ns <= _U64_MAX:
            raise FlatM13B4TraceError("timestamp_ns must be within the u64 range")


@dataclass(frozen=True, slots=True)
class FlatM13B4TraceV1:
    """Canonical, producer-owned M13B.4 trace retained by KVLab as evidence."""

    schema: str
    timing_source: str
    scheduling_variant: str
    scope: str
    events: tuple[FlatM13B4TraceEventV1, ...]

    def validate(self) -> None:
        """Mirror the producer's fail-closed structural and event-order contract."""

        if self.schema != FLAT_M13B4_TRACE_SCHEMA:
            raise FlatM13B4TraceError(f"unsupported trace schema: {self.schema!r}")
        if not isinstance(self.timing_source, str) or self.timing_source not in _TIMING_SOURCES:
            raise FlatM13B4TraceError(f"unsupported timing_source: {self.timing_source!r}")
        if (
            not isinstance(self.scheduling_variant, str)
            or self.scheduling_variant not in _SCHEDULING_VARIANTS
        ):
            raise FlatM13B4TraceError(
                f"unsupported scheduling_variant: {self.scheduling_variant!r}"
            )
        if not isinstance(self.scope, str) or self.scope not in _SCOPES:
            raise FlatM13B4TraceError(f"unsupported trace scope: {self.scope!r}")
        if not isinstance(self.events, tuple) or not self.events:
            raise FlatM13B4TraceError("events must be a non-empty tuple")

        timestamps: dict[str, int] = {}
        previous_ns: int | None = None
        for event in self.events:
            if not isinstance(event, FlatM13B4TraceEventV1):
                raise FlatM13B4TraceError(
                    "events must contain FlatM13B4TraceEventV1 values"
                )
            event.validate()
            if previous_ns is not None and event.timestamp_ns < previous_ns:
                raise FlatM13B4TraceError("event timestamps must be non-decreasing")
            if event.kind in timestamps:
                raise FlatM13B4TraceError(f"duplicate M13B.4 event: {event.kind}")
            timestamps[event.kind] = event.timestamp_ns
            previous_ns = event.timestamp_ns

        if self.scope == "prefill":
            numerical = self._required_timestamp(timestamps, "numerical_kv_commit")
            boolean = self._required_timestamp(timestamps, "boolean_signature_commit")
            visible = self._required_timestamp(timestamps, "decode_visible")
            if numerical > visible:
                raise FlatM13B4TraceError(
                    "numerical_kv_commit must not follow decode_visible"
                )
            if boolean > visible:
                raise FlatM13B4TraceError(
                    "boolean_signature_commit must not follow decode_visible"
                )
            return

        for before, after in zip(_DECODE_REQUIRED, _DECODE_REQUIRED[1:]):
            if self._required_timestamp(timestamps, before) > self._required_timestamp(
                timestamps, after
            ):
                raise FlatM13B4TraceError(f"{before} must not follow {after}")

        wait_start = timestamps.get("synchronization_wait_start")
        wait_end = timestamps.get("synchronization_wait_end")
        if (wait_start is None) != (wait_end is None):
            raise FlatM13B4TraceError("synchronization wait evidence must be a complete pair")
        if wait_start is not None and wait_end is not None:
            if wait_start > wait_end:
                raise FlatM13B4TraceError(
                    "synchronization_wait_start must not follow synchronization_wait_end"
                )
            query_ready = self._required_timestamp(timestamps, "query_representation_ready")
            output_ready = self._required_timestamp(timestamps, "output_ready")
            if wait_start < query_ready:
                raise FlatM13B4TraceError(
                    "synchronization_wait_start must not precede query_representation_ready"
                )
            if wait_end > output_ready:
                raise FlatM13B4TraceError(
                    "synchronization_wait_end must not follow output_ready"
                )
        elif self.scheduling_variant == "multi_dispatch_overlap_candidate":
            raise FlatM13B4TraceError(
                "multi-dispatch candidate requires explicit synchronization evidence"
            )

    @staticmethod
    def _required_timestamp(timestamps: dict[str, int], kind: str) -> int:
        try:
            return timestamps[kind]
        except KeyError as exc:
            raise FlatM13B4TraceError(f"missing required M13B.4 event: {kind}") from exc

    def canonical_json_bytes(self) -> bytes:
        """Return exactly the field order/encoding emitted by the FLAT producer."""

        self.validate()
        payload = {
            "schema": self.schema,
            "timing_source": self.timing_source,
            "scheduling_variant": self.scheduling_variant,
            "scope": self.scope,
            "events": [
                {"kind": event.kind, "timestamp_ns": event.timestamp_ns}
                for event in self.events
            ],
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

    def trace_sha256(self) -> str:
        """Hash the exact canonical producer bytes retained by KVLab."""

        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()

    @classmethod
    def from_canonical_json_bytes(cls, payload: bytes) -> "FlatM13B4TraceV1":
        """Parse only exact canonical ``flat.m13b4-trace.v1`` evidence bytes."""

        if not isinstance(payload, bytes) or not payload:
            raise FlatM13B4TraceError("trace payload must be non-empty bytes")
        try:
            parsed: Any = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FlatM13B4TraceError("trace payload must be UTF-8 JSON") from exc
        if not isinstance(parsed, dict) or set(parsed) != _TRACE_FIELDS:
            raise FlatM13B4TraceError("trace payload fields do not match schema")
        raw_events = parsed["events"]
        if not isinstance(raw_events, list):
            raise FlatM13B4TraceError("events must be a JSON array")

        events: list[FlatM13B4TraceEventV1] = []
        for raw_event in raw_events:
            if not isinstance(raw_event, dict) or set(raw_event) != _EVENT_FIELDS:
                raise FlatM13B4TraceError("trace event fields do not match schema")
            events.append(
                FlatM13B4TraceEventV1(
                    kind=raw_event["kind"],
                    timestamp_ns=raw_event["timestamp_ns"],
                )
            )

        trace = cls(
            schema=parsed["schema"],
            timing_source=parsed["timing_source"],
            scheduling_variant=parsed["scheduling_variant"],
            scope=parsed["scope"],
            events=tuple(events),
        )
        trace.validate()
        if trace.canonical_json_bytes() != payload:
            raise FlatM13B4TraceError("trace payload is not in canonical FLAT JSON form")
        return trace
