"""Canonical explicit KV-selection handoff with sequence-position identity.

Version 1 used vocabulary token values as set identities and therefore could
not represent ordinary model inputs containing repeated token values. Version 2
separates occurrence identity from token value: `input_token_ids` stores the
actual model tokens and retained/evicted partitions are expressed as sequence
positions.

Logical byte accounting is not allocator release, freed HBM, avoided physical
traffic, latency reduction, or model-quality evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Sequence


PROSPECT_KV_SELECTION_HANDOFF_SCHEMA_V2 = "kvlab.prospect-kv-selection/v2"


class ProspectKvSelectionHandoffV2Error(ValueError):
    """Raised when a position-identified selection handoff is inconsistent."""


@dataclass(frozen=True, slots=True)
class ProspectKvSelectionHandoffV2:
    schema: str
    policy: str
    input_token_ids: tuple[int, ...]
    bytes_per_token: int
    retained_positions: tuple[int, ...]
    evicted_positions: tuple[int, ...]
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
        retained_positions: Sequence[int],
    ) -> "ProspectKvSelectionHandoffV2":
        inputs = _validate_token_values("token_ids", token_ids, allow_empty=False)
        retained = _validate_positions(
            "retained_positions",
            retained_positions,
            input_len=len(inputs),
        )
        _require_positive_int_value("bytes_per_token", bytes_per_token)
        policy = _require_text_value("policy", policy)

        retained_set = set(retained)
        evicted = tuple(position for position in range(len(inputs)) if position not in retained_set)
        logical_input_bytes = _checked_bytes(len(inputs), bytes_per_token)
        logical_retained_bytes = _checked_bytes(len(retained), bytes_per_token)
        handoff = cls(
            schema=PROSPECT_KV_SELECTION_HANDOFF_SCHEMA_V2,
            policy=policy,
            input_token_ids=inputs,
            bytes_per_token=bytes_per_token,
            retained_positions=retained,
            evicted_positions=evicted,
            logical_input_bytes=logical_input_bytes,
            logical_retained_bytes=logical_retained_bytes,
            logical_evicted_bytes=logical_input_bytes - logical_retained_bytes,
        )
        handoff.validate_replay()
        return handoff

    @property
    def retained_token_ids(self) -> tuple[int, ...]:
        """Token values corresponding to retained occurrences, duplicates preserved."""
        return tuple(self.input_token_ids[position] for position in self.retained_positions)

    @property
    def evicted_token_ids(self) -> tuple[int, ...]:
        """Token values corresponding to evicted occurrences, duplicates preserved."""
        return tuple(self.input_token_ids[position] for position in self.evicted_positions)

    def canonical_json(self) -> str:
        self.validate_replay()
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_canonical_json(cls, payload: str) -> "ProspectKvSelectionHandoffV2":
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ProspectKvSelectionHandoffV2Error("invalid JSON") from error
        if not isinstance(raw, dict):
            raise ProspectKvSelectionHandoffV2Error("handoff must be a JSON object")
        if json.dumps(raw, sort_keys=True, separators=(",", ":")) != payload:
            raise ProspectKvSelectionHandoffV2Error("handoff JSON is not canonical")

        expected = {
            "schema",
            "policy",
            "input_token_ids",
            "bytes_per_token",
            "retained_positions",
            "evicted_positions",
            "logical_input_bytes",
            "logical_retained_bytes",
            "logical_evicted_bytes",
        }
        if set(raw) != expected:
            raise ProspectKvSelectionHandoffV2Error(
                "handoff fields do not match schema v2"
            )

        inputs = _require_token_values(raw, "input_token_ids", allow_empty=False)
        handoff = cls(
            schema=_require_text(raw, "schema"),
            policy=_require_text(raw, "policy"),
            input_token_ids=inputs,
            bytes_per_token=_require_positive_int(raw, "bytes_per_token"),
            retained_positions=_require_positions(
                raw, "retained_positions", input_len=len(inputs)
            ),
            evicted_positions=_require_positions(
                raw, "evicted_positions", input_len=len(inputs)
            ),
            logical_input_bytes=_require_non_negative_int(raw, "logical_input_bytes"),
            logical_retained_bytes=_require_non_negative_int(
                raw, "logical_retained_bytes"
            ),
            logical_evicted_bytes=_require_non_negative_int(raw, "logical_evicted_bytes"),
        )
        handoff.validate_replay()
        return handoff

    def validate_replay(self) -> None:
        if self.schema != PROSPECT_KV_SELECTION_HANDOFF_SCHEMA_V2:
            raise ProspectKvSelectionHandoffV2Error(
                "unsupported selection handoff schema"
            )
        _require_text_value("policy", self.policy)
        inputs = _validate_token_values(
            "input_token_ids", self.input_token_ids, allow_empty=False
        )
        retained = _validate_positions(
            "retained_positions", self.retained_positions, input_len=len(inputs)
        )
        evicted = _validate_positions(
            "evicted_positions", self.evicted_positions, input_len=len(inputs)
        )
        _require_positive_int_value("bytes_per_token", self.bytes_per_token)

        retained_set = set(retained)
        evicted_set = set(evicted)
        if retained_set & evicted_set:
            raise ProspectKvSelectionHandoffV2Error(
                "retained and evicted positions must be disjoint"
            )
        expected_positions = set(range(len(inputs)))
        if retained_set | evicted_set != expected_positions:
            raise ProspectKvSelectionHandoffV2Error(
                "retained and evicted positions must partition the input sequence"
            )
        expected_evicted = tuple(
            position for position in range(len(inputs)) if position not in retained_set
        )
        if evicted != expected_evicted:
            raise ProspectKvSelectionHandoffV2Error(
                "evicted_positions must be the canonical complement of retained_positions"
            )

        expected_input_bytes = _checked_bytes(len(inputs), self.bytes_per_token)
        expected_retained_bytes = _checked_bytes(len(retained), self.bytes_per_token)
        expected_evicted_bytes = expected_input_bytes - expected_retained_bytes
        if self.logical_input_bytes != expected_input_bytes:
            raise ProspectKvSelectionHandoffV2Error("logical_input_bytes mismatch")
        if self.logical_retained_bytes != expected_retained_bytes:
            raise ProspectKvSelectionHandoffV2Error("logical_retained_bytes mismatch")
        if self.logical_evicted_bytes != expected_evicted_bytes:
            raise ProspectKvSelectionHandoffV2Error("logical_evicted_bytes mismatch")


def _checked_bytes(count: int, bytes_per_token: int) -> int:
    return count * bytes_per_token


def _require_text(raw: dict[str, Any], field: str) -> str:
    return _require_text_value(field, raw[field])


def _require_text_value(field: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProspectKvSelectionHandoffV2Error(f"{field} must be a non-empty string")
    return value


def _require_positive_int(raw: dict[str, Any], field: str) -> int:
    value = raw[field]
    _require_positive_int_value(field, value)
    return value


def _require_positive_int_value(field: str, value: Any) -> None:
    if type(value) is not int or value <= 0:
        raise ProspectKvSelectionHandoffV2Error(f"{field} must be a positive integer")


def _require_non_negative_int(raw: dict[str, Any], field: str) -> int:
    value = raw[field]
    if type(value) is not int or value < 0:
        raise ProspectKvSelectionHandoffV2Error(
            f"{field} must be a non-negative integer"
        )
    return value


def _require_token_values(
    raw: dict[str, Any], field: str, *, allow_empty: bool
) -> tuple[int, ...]:
    value = raw[field]
    if not isinstance(value, list):
        raise ProspectKvSelectionHandoffV2Error(f"{field} must be an array")
    return _validate_token_values(field, value, allow_empty=allow_empty)


def _validate_token_values(
    field: str, values: Sequence[int], *, allow_empty: bool
) -> tuple[int, ...]:
    values = tuple(values)
    if not allow_empty and not values:
        raise ProspectKvSelectionHandoffV2Error(f"{field} must not be empty")
    if any(type(token_id) is not int or token_id < 0 for token_id in values):
        raise ProspectKvSelectionHandoffV2Error(
            f"{field} must contain non-negative integer token ids"
        )
    # Repeated vocabulary values are valid. Occurrence identity is positional.
    return values


def _require_positions(
    raw: dict[str, Any], field: str, *, input_len: int
) -> tuple[int, ...]:
    value = raw[field]
    if not isinstance(value, list):
        raise ProspectKvSelectionHandoffV2Error(f"{field} must be an array")
    return _validate_positions(field, value, input_len=input_len)


def _validate_positions(
    field: str, values: Sequence[int], *, input_len: int
) -> tuple[int, ...]:
    positions = tuple(values)
    previous = None
    for position in positions:
        if type(position) is not int or not 0 <= position < input_len:
            raise ProspectKvSelectionHandoffV2Error(
                f"{field} must contain in-range non-negative sequence positions"
            )
        if previous is not None and position <= previous:
            raise ProspectKvSelectionHandoffV2Error(
                f"{field} must be strictly increasing and unique"
            )
        previous = position
    return positions
