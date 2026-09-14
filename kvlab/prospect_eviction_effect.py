"""Replayable synthetic numerical effect for ProspectEngine KV eviction.

This module connects KVLab's exact logical oldest-first eviction contract to the
existing additive synthetic oracle. It is deliberately a synthetic numerical
experiment: it does not infer real-model quality, latency, allocator release,
GPU/HBM residency, or physical traffic from logical eviction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isfinite
from typing import Any, Mapping

from .baselines import BudgetSelection
from .evaluation import evaluate_selection
from .prospect_eviction_handoff import (
    ProspectKvEvictionHandoffError,
    ProspectKvEvictionHandoffV1,
)
from .synthetic_trace import KvRegion, SyntheticKvTrace


PROSPECT_KV_EVICTION_EFFECT_SCHEMA_V1 = "kvlab.prospect-kv-eviction-effect/v1"


class ProspectKvEvictionEffectError(ValueError):
    """Raised when synthetic eviction-effect evidence cannot be replayed."""


@dataclass(frozen=True)
class BoundKvRegion:
    """One synthetic region explicitly bound to one logical KV token id."""

    token_id: int
    region_id: str
    storage_bytes: int
    contribution: tuple[float, ...]


@dataclass(frozen=True)
class ProspectKvEvictionEffectV1:
    """Self-contained synthetic numerical consequence of one KV eviction."""

    schema: str
    trace_id: str
    regions: tuple[BoundKvRegion, ...]
    eviction: ProspectKvEvictionHandoffV1
    retained_region_ids: tuple[str, ...]
    evicted_region_ids: tuple[str, ...]
    full_cache_output: tuple[float, ...]
    retained_output: tuple[float, ...]
    output_l2_delta: float
    logical_evicted_bytes: int

    @classmethod
    def capture(
        cls,
        *,
        trace: SyntheticKvTrace,
        eviction: ProspectKvEvictionHandoffV1,
        token_to_region: Mapping[int, str],
    ) -> "ProspectKvEvictionEffectV1":
        try:
            eviction.validate_replay()
        except ProspectKvEvictionHandoffError as error:
            raise ProspectKvEvictionEffectError("invalid eviction handoff") from error

        regions = _bind_regions(trace, eviction, token_to_region)
        replay = _evaluate_bound_regions(trace.trace_id, regions, eviction)
        effect = cls(
            schema=PROSPECT_KV_EVICTION_EFFECT_SCHEMA_V1,
            trace_id=trace.trace_id,
            regions=regions,
            eviction=eviction,
            retained_region_ids=replay.retained_region_ids,
            evicted_region_ids=replay.evicted_region_ids,
            full_cache_output=replay.full_cache_output,
            retained_output=replay.retained_output,
            output_l2_delta=replay.output_l2_delta,
            logical_evicted_bytes=eviction.logical_evicted_bytes,
        )
        effect.validate_replay()
        return effect

    def canonical_json(self) -> str:
        self.validate_replay()
        return json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_canonical_json(cls, payload: str) -> "ProspectKvEvictionEffectV1":
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ProspectKvEvictionEffectError("invalid JSON") from error
        if not isinstance(raw, dict):
            raise ProspectKvEvictionEffectError("effect evidence must be a JSON object")
        if json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False) != payload:
            raise ProspectKvEvictionEffectError("effect evidence JSON is not canonical")

        expected_keys = {
            "schema",
            "trace_id",
            "regions",
            "eviction",
            "retained_region_ids",
            "evicted_region_ids",
            "full_cache_output",
            "retained_output",
            "output_l2_delta",
            "logical_evicted_bytes",
        }
        if set(raw) != expected_keys:
            raise ProspectKvEvictionEffectError("effect fields do not match schema v1")

        eviction_raw = raw["eviction"]
        if not isinstance(eviction_raw, dict):
            raise ProspectKvEvictionEffectError("eviction must be an object")
        eviction_json = json.dumps(
            eviction_raw,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        try:
            eviction = ProspectKvEvictionHandoffV1.from_canonical_json(eviction_json)
        except ProspectKvEvictionHandoffError as error:
            raise ProspectKvEvictionEffectError("invalid embedded eviction") from error

        regions_raw = raw["regions"]
        if not isinstance(regions_raw, list) or not regions_raw:
            raise ProspectKvEvictionEffectError("regions must be a non-empty array")
        regions = tuple(_parse_region(region) for region in regions_raw)

        effect = cls(
            schema=_require_string(raw, "schema"),
            trace_id=_require_string(raw, "trace_id"),
            regions=regions,
            eviction=eviction,
            retained_region_ids=_require_string_array(raw, "retained_region_ids"),
            evicted_region_ids=_require_string_array(raw, "evicted_region_ids"),
            full_cache_output=_require_float_array(raw, "full_cache_output"),
            retained_output=_require_float_array(raw, "retained_output"),
            output_l2_delta=_require_finite_float(raw, "output_l2_delta"),
            logical_evicted_bytes=_require_non_negative_int(raw, "logical_evicted_bytes"),
        )
        effect.validate_replay()
        return effect

    def validate_replay(self) -> None:
        if self.schema != PROSPECT_KV_EVICTION_EFFECT_SCHEMA_V1:
            raise ProspectKvEvictionEffectError("unsupported eviction-effect schema")
        if not self.trace_id:
            raise ProspectKvEvictionEffectError("trace_id must be non-empty")
        try:
            self.eviction.validate_replay()
        except ProspectKvEvictionHandoffError as error:
            raise ProspectKvEvictionEffectError("invalid embedded eviction") from error

        replay = _evaluate_bound_regions(self.trace_id, self.regions, self.eviction)
        if self.retained_region_ids != replay.retained_region_ids:
            raise ProspectKvEvictionEffectError("retained regions do not match replay")
        if self.evicted_region_ids != replay.evicted_region_ids:
            raise ProspectKvEvictionEffectError("evicted regions do not match replay")
        if self.full_cache_output != replay.full_cache_output:
            raise ProspectKvEvictionEffectError("full-cache output does not match replay")
        if self.retained_output != replay.retained_output:
            raise ProspectKvEvictionEffectError("retained output does not match replay")
        if self.output_l2_delta != replay.output_l2_delta:
            raise ProspectKvEvictionEffectError("L2 delta does not match replay")
        if self.logical_evicted_bytes != self.eviction.logical_evicted_bytes:
            raise ProspectKvEvictionEffectError("logical evicted bytes do not match eviction")


@dataclass(frozen=True)
class _Replay:
    retained_region_ids: tuple[str, ...]
    evicted_region_ids: tuple[str, ...]
    full_cache_output: tuple[float, ...]
    retained_output: tuple[float, ...]
    output_l2_delta: float


def _bind_regions(
    trace: SyntheticKvTrace,
    eviction: ProspectKvEvictionHandoffV1,
    token_to_region: Mapping[int, str],
) -> tuple[BoundKvRegion, ...]:
    token_ids = eviction.input_token_ids
    if set(token_to_region) != set(token_ids):
        raise ProspectKvEvictionEffectError(
            "token_to_region keys must exactly cover eviction input token ids"
        )
    region_ids = tuple(token_to_region[token_id] for token_id in token_ids)
    if len(set(region_ids)) != len(region_ids):
        raise ProspectKvEvictionEffectError("token_to_region values must be unique")

    by_id = {region.region_id: region for region in trace.regions}
    if set(region_ids) != set(by_id):
        raise ProspectKvEvictionEffectError(
            "token_to_region values must exactly cover synthetic trace regions"
        )

    bound = []
    for token_id, region_id in zip(token_ids, region_ids, strict=True):
        region = by_id[region_id]
        if region.storage_bytes != eviction.bytes_per_token:
            raise ProspectKvEvictionEffectError(
                "every bound synthetic region must match eviction bytes_per_token"
            )
        if any(not isfinite(value) for value in region.contribution):
            raise ProspectKvEvictionEffectError("region contributions must be finite")
        bound.append(
            BoundKvRegion(
                token_id=token_id,
                region_id=region.region_id,
                storage_bytes=region.storage_bytes,
                contribution=region.contribution,
            )
        )
    return tuple(bound)


def _evaluate_bound_regions(
    trace_id: str,
    regions: tuple[BoundKvRegion, ...],
    eviction: ProspectKvEvictionHandoffV1,
) -> _Replay:
    if not regions:
        raise ProspectKvEvictionEffectError("regions must not be empty")
    token_ids = tuple(region.token_id for region in regions)
    if token_ids != eviction.input_token_ids:
        raise ProspectKvEvictionEffectError(
            "bound region token order must match eviction input token order"
        )
    region_ids = tuple(region.region_id for region in regions)
    if len(set(region_ids)) != len(region_ids):
        raise ProspectKvEvictionEffectError("bound region ids must be unique")
    if any(region.storage_bytes != eviction.bytes_per_token for region in regions):
        raise ProspectKvEvictionEffectError(
            "bound region storage must match eviction bytes_per_token"
        )

    try:
        trace = SyntheticKvTrace(
            trace_id,
            tuple(
                KvRegion(region.region_id, region.storage_bytes, region.contribution)
                for region in regions
            ),
        )
    except ValueError as error:
        raise ProspectKvEvictionEffectError("invalid synthetic trace") from error

    by_token = {region.token_id: region.region_id for region in regions}
    retained_region_ids = tuple(by_token[token_id] for token_id in eviction.retained_token_ids)
    evicted_region_ids = tuple(by_token[token_id] for token_id in eviction.evicted_token_ids)
    selection = BudgetSelection(
        policy="oldest_first",
        budget_bytes=eviction.logical_retained_bytes,
        retained_region_ids=retained_region_ids,
        retained_bytes=eviction.logical_retained_bytes,
    )
    try:
        evaluation = evaluate_selection(trace, selection)
    except ValueError as error:
        raise ProspectKvEvictionEffectError("synthetic selection replay failed") from error

    return _Replay(
        retained_region_ids=retained_region_ids,
        evicted_region_ids=evicted_region_ids,
        full_cache_output=evaluation.full_cache_output,
        retained_output=evaluation.selected_output,
        output_l2_delta=evaluation.output_l2_delta,
    )


def _parse_region(raw: Any) -> BoundKvRegion:
    if not isinstance(raw, dict):
        raise ProspectKvEvictionEffectError("region entries must be objects")
    expected = {"token_id", "region_id", "storage_bytes", "contribution"}
    if set(raw) != expected:
        raise ProspectKvEvictionEffectError("region fields do not match schema v1")
    return BoundKvRegion(
        token_id=_require_non_negative_int(raw, "token_id"),
        region_id=_require_string(raw, "region_id"),
        storage_bytes=_require_positive_int(raw, "storage_bytes"),
        contribution=_require_float_array(raw, "contribution"),
    )


def _require_string(raw: dict[str, Any], field: str) -> str:
    value = raw[field]
    if not isinstance(value, str) or not value:
        raise ProspectKvEvictionEffectError(f"{field} must be a non-empty string")
    return value


def _require_string_array(raw: dict[str, Any], field: str) -> tuple[str, ...]:
    value = raw[field]
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ProspectKvEvictionEffectError(f"{field} must contain non-empty strings")
    if len(set(value)) != len(value):
        raise ProspectKvEvictionEffectError(f"{field} values must be unique")
    return tuple(value)


def _require_float_array(raw: dict[str, Any], field: str) -> tuple[float, ...]:
    value = raw[field]
    if not isinstance(value, list) or not value:
        raise ProspectKvEvictionEffectError(f"{field} must be a non-empty numeric array")
    parsed = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ProspectKvEvictionEffectError(f"{field} must contain numbers")
        converted = float(item)
        if not isfinite(converted):
            raise ProspectKvEvictionEffectError(f"{field} must contain finite numbers")
        parsed.append(converted)
    return tuple(parsed)


def _require_finite_float(raw: dict[str, Any], field: str) -> float:
    value = raw[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProspectKvEvictionEffectError(f"{field} must be numeric")
    converted = float(value)
    if not isfinite(converted) or converted < 0.0:
        raise ProspectKvEvictionEffectError(f"{field} must be a finite non-negative number")
    return converted


def _require_positive_int(raw: dict[str, Any], field: str) -> int:
    value = raw[field]
    if type(value) is not int or value <= 0:
        raise ProspectKvEvictionEffectError(f"{field} must be a positive integer")
    return value


def _require_non_negative_int(raw: dict[str, Any], field: str) -> int:
    value = raw[field]
    if type(value) is not int or value < 0:
        raise ProspectKvEvictionEffectError(f"{field} must be a non-negative integer")
    return value
