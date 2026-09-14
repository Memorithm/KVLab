"""Canonical explicit logical KV-selection handoff for ProspectEngine.

Unlike ``prospect-kv-eviction/v1``, this contract is not tied to one eviction
algorithm.  It records an exact retained/evicted partition plus a policy label
and revalidates only the structural/logical facts that are actually present.
The policy label is provenance, not proof that a named heuristic was executed
correctly; heuristic-specific replay must be supplied separately.

Logical byte accounting is not allocator release, freed HBM, avoided physical
traffic, latency reduction, or model-quality evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Sequence


PROSPECT_KV_SELECTION_HANDOFF_SCHEMA_V1 = "kvlab.prospect-kv-selection/v1"


class ProspectKvSelectionHandoffError(ValueError):
    """Raised when an explicit selection handoff is malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class ProspectKvSelectionHandoffV1:
    schema: str
    policy: str
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
        policy: str,
        retained_token_ids: Sequence[int],
    ) -> "ProspectKvSelectionHandoffV1":
        inputs = _validate_token_sequence("token_ids", token_ids, allow_empty=False)
        retained = _validate_token_sequence(
            "retained_token_ids", retained_token_ids, allow_empty=True
        )
        _require_positive_int_value("bytes_per_token", bytes_per_token)
        policy = _require_text_value("policy", policy)

        input_set = set(inputs)
        if any(token_id not in input_set for token_id in retained):
            raise ProspectKvSelectionHandoffError(
                "retained_token_ids must be a subset of input_token_ids"
            )
        retained_set = set(retained)
        canonical_retained = tuple(token_id for token_id in inputs if token_id in retained_set)
        if retained != canonical_retained:
            raise ProspectKvSelectionHandoffError(
                "retained_token_ids must preserve input token order"
            )
        evicted = tuple(token_id for token_id in inputs if token_id not in retained_set)

        logical_input_bytes = _checked_bytes(len(inputs), bytes_per_token)
        logical_retained_bytes = _checked_bytes(len(retained), bytes_per_token)
        handoff = cls(
            schema=PROSPECT_KV_SELECTION_HANDOFF_SCHEMA_V1,
            policy=policy,
            input_token_ids=inputs,
            bytes_per_token=bytes_per_token,
            retained_token_ids=retained,
            evicted_token_ids=evicted,
            logical_input_bytes=logical_input_bytes,
            logical_retained_bytes=logical_retained_bytes,
            logical_evicted_bytes=logical_input_bytes - logical_retained_bytes,
        )
        handoff.validate_replay()
        return handoff

    def canonical_json(self) -> str:
        self.validate_replay()
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_canonical_json(cls, payload: str) -> "ProspectKvSelectionHandoffV1":
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ProspectKvSelectionHandoffError("invalid JSON") from error
        if not isinstance(raw, dict):
            raise ProspectKvSelectionHandoffError("handoff must be a JSON object")
        if json.dumps(raw, sort_keys=True, separators=(",", ":")) != payload:
            raise ProspectKvSelectionHandoffError("handoff JSON is not canonical")

        expected = {
            "schema",
            "policy",
            "input_token_ids",
            "bytes_per_token",
            "retained_token_ids",
            "evicted_token_ids",
            "logical_input_bytes",
            "logical_retained_bytes",
            "logical_evicted_bytes",
        }
        if set(raw) != expected:
            raise ProspectKvSelectionHandoffError(
                "handoff fields do not match schema v1"
            )

        handoff = cls(
            schema=_require_text(raw, "schema"),
            policy=_require_text(raw, "policy"),
            input_token_ids=_require_tokens(raw, "input_token_ids", allow_empty=False),
            bytes_per_token=_require_positive_int(raw, "bytes_per_token"),
            retained_token_ids=_require_tokens(raw, "retained_token_ids", allow_empty=True),
            evicted_token_ids=_require_tokens(raw, "evicted_token_ids", allow_empty=True),
            logical_input_bytes=_require_non_negative_int(raw, "logical_input_bytes"),
            logical_retained_bytes=_require_non_negative_int(raw, "logical_retained_bytes"),
            logical_evicted_bytes=_require_non_negative_int(raw, "logical_evicted_bytes"),
        )
        handoff.validate_replay()
        return handoff

    def validate_replay(self) -> None:
        if self.schema != PROSPECT_KV_SELECTION_HANDOFF_SCHEMA_V1:
            raise ProspectKvSelectionHandoffError("unsupported selection handoff schema")
        _require_text_value("policy", self.policy)
        inputs = _validate_token_sequence(
            "input_token_ids", self.input_token_ids, allow_empty=False
        )
        retained = _validate_token_sequence(
            "retained_token_ids", self.retained_token_ids, allow_empty=True
        )
        evicted = _validate_token_sequence(
            "evicted_token_ids", self.evicted_token_ids, allow_empty=True
        )
        _require_positive_int_value("bytes_per_token", self.bytes_per_token)

        input_set = set(inputs)
        retained_set = set(retained)
        evicted_set = set(evicted)
        if retained_set & evicted_set:
            raise ProspectKvSelectionHandoffError(
                "retained and evicted token sets must be disjoint"
            )
        if retained_set | evicted_set != input_set:
            raise ProspectKvSelectionHandoffError(
                "retained and evicted token sets must partition input_token_ids"
            )

        expected_retained = tuple(token_id for token_id in inputs if token_id in retained_set)
        expected_evicted = tuple(token_id for token_id in inputs if token_id in evicted_set)
        if retained != expected_retained:
            raise ProspectKvSelectionHandoffError(
                "retained_token_ids must preserve input token order"
            )
        if evicted != expected_evicted:
            raise ProspectKvSelectionHandoffError(
                "evicted_token_ids must preserve input token order"
            )

        expected_input_bytes = _checked_bytes(len(inputs), self.bytes_per_token)
        expected_retained_bytes = _checked_bytes(len(retained), self.bytes_per_token)
        expected_evicted_bytes = expected_input_bytes - expected_retained_bytes
        if self.logical_input_bytes != expected_input_bytes:
            raise ProspectKvSelectionHandoffError("logical_input_bytes mismatch")
        if self.logical_retained_bytes != expected_retained_bytes:
            raise ProspectKvSelectionHandoffError("logical_retained_bytes mismatch")
        if self.logical_evicted_bytes != expected_evicted_bytes:
            raise ProspectKvSelectionHandoffError("logical_evicted_bytes mismatch")


def _checked_bytes(count: int, bytes_per_token: int) -> int:
    value = count * bytes_per_token
    if value < 0:
        raise ProspectKvSelectionHandoffError("logical byte accounting overflow")
    return value


def _require_text(raw: dict[str, Any], field: str) -> str:
    return _require_text_value(field, raw[field])


def _require_text_value(field: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProspectKvSelectionHandoffError(f"{field} must be a non-empty string")
    return value


def _require_positive_int(raw: dict[str, Any], field: str) -> int:
    value = raw[field]
    _require_positive_int_value(field, value)
    return value


def _require_positive_int_value(field: str, value: Any) -> None:
    if type(value) is not int or value <= 0:
        raise ProspectKvSelectionHandoffError(f"{field} must be a positive integer")


def _require_non_negative_int(raw: dict[str, Any], field: str) -> int:
    value = raw[field]
    if type(value) is not int or value < 0:
        raise ProspectKvSelectionHandoffError(
            f"{field} must be a non-negative integer"
        )
    return value


def _require_tokens(
    raw: dict[str, Any], field: str, *, allow_empty: bool
) -> tuple[int, ...]:
    value = raw[field]
    if not isinstance(value, list):
        raise ProspectKvSelectionHandoffError(f"{field} must be an array")
    return _validate_token_sequence(field, value, allow_empty=allow_empty)


def _validate_token_sequence(
    field: str, values: Sequence[int], *, allow_empty: bool
) -> tuple[int, ...]:
    values = tuple(values)
    if not allow_empty and not values:
        raise ProspectKvSelectionHandoffError(f"{field} must not be empty")
    if any(type(token_id) is not int or token_id < 0 for token_id in values):
        raise ProspectKvSelectionHandoffError(
            f"{field} must contain non-negative integer token ids"
        )
    if len(values) != len(set(values)):
        raise ProspectKvSelectionHandoffError(f"{field} token ids must be unique")
    return values
