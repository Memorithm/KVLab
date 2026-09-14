"""Canonical logical KV-eviction handoff for ProspectEngine.

This contract serializes and replays KVLab's exact logical eviction semantics.
It intentionally does not infer allocator release, GPU/HBM residency, transfer
volume, latency, or downstream numerical/model quality from evicted bytes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Sequence

from .eviction import EvictionOrder, EvictionPolicy, EvictionResult, apply_eviction


PROSPECT_KV_EVICTION_HANDOFF_SCHEMA_V1 = "kvlab.prospect-kv-eviction/v1"


class ProspectKvEvictionHandoffError(ValueError):
    """Raised when a handoff is malformed, non-canonical, or non-replayable."""


@dataclass(frozen=True)
class ProspectKvEvictionHandoffV1:
    schema: str
    order: str
    max_tokens: int
    input_token_ids: tuple[int, ...]
    bytes_per_token: int
    retained_token_ids: tuple[int, ...]
    evicted_token_ids: tuple[int, ...]
    logical_input_bytes: int
    logical_retained_bytes: int
    logical_evicted_bytes: int

    @classmethod
    def capture(
        cls,
        *,
        token_ids: Sequence[int],
        bytes_per_token: int,
        policy: EvictionPolicy,
    ) -> "ProspectKvEvictionHandoffV1":
        result = apply_eviction(
            token_ids=token_ids,
            bytes_per_token=bytes_per_token,
            policy=policy,
        )
        handoff = cls._from_result(policy=policy, result=result)
        handoff.validate_replay()
        return handoff

    @classmethod
    def _from_result(
        cls,
        *,
        policy: EvictionPolicy,
        result: EvictionResult,
    ) -> "ProspectKvEvictionHandoffV1":
        return cls(
            schema=PROSPECT_KV_EVICTION_HANDOFF_SCHEMA_V1,
            order=policy.order.value,
            max_tokens=policy.max_tokens,
            input_token_ids=result.input_token_ids,
            bytes_per_token=result.bytes_per_token,
            retained_token_ids=result.retained_token_ids,
            evicted_token_ids=result.evicted_token_ids,
            logical_input_bytes=result.logical_input_bytes,
            logical_retained_bytes=result.logical_retained_bytes,
            logical_evicted_bytes=result.logical_evicted_bytes,
        )

    def canonical_json(self) -> str:
        self.validate_replay()
        payload = asdict(self)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_canonical_json(cls, payload: str) -> "ProspectKvEvictionHandoffV1":
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ProspectKvEvictionHandoffError("invalid JSON") from error
        if not isinstance(raw, dict):
            raise ProspectKvEvictionHandoffError("handoff must be a JSON object")
        if json.dumps(raw, sort_keys=True, separators=(",", ":")) != payload:
            raise ProspectKvEvictionHandoffError("handoff JSON is not canonical")

        expected_keys = {
            "schema",
            "order",
            "max_tokens",
            "input_token_ids",
            "bytes_per_token",
            "retained_token_ids",
            "evicted_token_ids",
            "logical_input_bytes",
            "logical_retained_bytes",
            "logical_evicted_bytes",
        }
        if set(raw) != expected_keys:
            raise ProspectKvEvictionHandoffError("handoff fields do not match schema v1")

        handoff = cls(
            schema=_require_string(raw, "schema"),
            order=_require_string(raw, "order"),
            max_tokens=_require_positive_int(raw, "max_tokens"),
            input_token_ids=_require_token_ids(raw, "input_token_ids"),
            bytes_per_token=_require_positive_int(raw, "bytes_per_token"),
            retained_token_ids=_require_token_ids(raw, "retained_token_ids"),
            evicted_token_ids=_require_token_ids(raw, "evicted_token_ids"),
            logical_input_bytes=_require_non_negative_int(raw, "logical_input_bytes"),
            logical_retained_bytes=_require_non_negative_int(raw, "logical_retained_bytes"),
            logical_evicted_bytes=_require_non_negative_int(raw, "logical_evicted_bytes"),
        )
        handoff.validate_replay()
        return handoff

    def validate_replay(self) -> None:
        if self.schema != PROSPECT_KV_EVICTION_HANDOFF_SCHEMA_V1:
            raise ProspectKvEvictionHandoffError("unsupported eviction handoff schema")
        try:
            order = EvictionOrder(self.order)
        except ValueError as error:
            raise ProspectKvEvictionHandoffError("unsupported eviction order") from error
        try:
            policy = EvictionPolicy(max_tokens=self.max_tokens, order=order)
            replay = apply_eviction(
                token_ids=self.input_token_ids,
                bytes_per_token=self.bytes_per_token,
                policy=policy,
            )
        except (TypeError, ValueError) as error:
            raise ProspectKvEvictionHandoffError("invalid eviction inputs") from error

        expected = self._from_result(policy=policy, result=replay)
        if self != expected:
            raise ProspectKvEvictionHandoffError(
                "recorded eviction outcome does not match replayed KVLab semantics"
            )


def _require_string(raw: dict[str, Any], field: str) -> str:
    value = raw[field]
    if not isinstance(value, str) or not value:
        raise ProspectKvEvictionHandoffError(f"{field} must be a non-empty string")
    return value


def _require_positive_int(raw: dict[str, Any], field: str) -> int:
    value = raw[field]
    if type(value) is not int or value <= 0:
        raise ProspectKvEvictionHandoffError(f"{field} must be a positive integer")
    return value


def _require_non_negative_int(raw: dict[str, Any], field: str) -> int:
    value = raw[field]
    if type(value) is not int or value < 0:
        raise ProspectKvEvictionHandoffError(
            f"{field} must be a non-negative integer"
        )
    return value


def _require_token_ids(raw: dict[str, Any], field: str) -> tuple[int, ...]:
    value = raw[field]
    if not isinstance(value, list):
        raise ProspectKvEvictionHandoffError(f"{field} must be an array")
    if any(type(token_id) is not int or token_id < 0 for token_id in value):
        raise ProspectKvEvictionHandoffError(
            f"{field} must contain non-negative integer token ids"
        )
    if len(set(value)) != len(value):
        raise ProspectKvEvictionHandoffError(f"{field} token ids must be unique")
    return tuple(value)
