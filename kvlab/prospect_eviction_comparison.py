"""Replayable budget-matched heuristic comparison for synthetic KV eviction.

This module compares one exact oldest-first eviction against KVLab's existing
synthetic calibration baselines under the same byte budget. The comparison is
limited to the additive synthetic oracle and carries no real-model or hardware
claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isfinite
from typing import Any

from .baselines import (
    BudgetSelection,
    select_lru_baseline,
    select_magnitude_baseline,
    select_random_baseline,
    select_sensitivity_per_byte_baseline,
)
from .evaluation import SelectionEvaluation, evaluate_selection
from .prospect_eviction_effect import (
    ProspectKvEvictionEffectError,
    ProspectKvEvictionEffectV1,
)
from .synthetic_trace import KvRegion, SyntheticKvTrace


PROSPECT_KV_HEURISTIC_COMPARISON_SCHEMA_V1 = (
    "kvlab.prospect-kv-heuristic-comparison/v1"
)


class ProspectKvHeuristicComparisonError(ValueError):
    """Raised when a comparison record cannot be reproduced exactly."""


@dataclass(frozen=True)
class HeuristicResult:
    policy: str
    retained_region_ids: tuple[str, ...]
    retained_bytes: int
    unused_bytes: int
    output_l2_delta: float


@dataclass(frozen=True)
class ProspectKvHeuristicComparisonV1:
    schema: str
    effect: ProspectKvEvictionEffectV1
    budget_bytes: int
    random_seed: int
    results: tuple[HeuristicResult, ...]
    best_policy: str

    @classmethod
    def capture(
        cls,
        *,
        effect: ProspectKvEvictionEffectV1,
        random_seed: int,
    ) -> "ProspectKvHeuristicComparisonV1":
        if type(random_seed) is not int or random_seed < 0 or random_seed > (2**64 - 1):
            raise ProspectKvHeuristicComparisonError(
                "random_seed must be an unsigned 64-bit integer"
            )
        try:
            effect.validate_replay()
        except ProspectKvEvictionEffectError as error:
            raise ProspectKvHeuristicComparisonError("invalid eviction effect") from error

        budget_bytes = effect.eviction.logical_retained_bytes
        results = _run_comparison(effect, random_seed)
        comparison = cls(
            schema=PROSPECT_KV_HEURISTIC_COMPARISON_SCHEMA_V1,
            effect=effect,
            budget_bytes=budget_bytes,
            random_seed=random_seed,
            results=results,
            best_policy=_best_policy(results),
        )
        comparison.validate_replay()
        return comparison

    def canonical_json(self) -> str:
        self.validate_replay()
        return json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_canonical_json(cls, payload: str) -> "ProspectKvHeuristicComparisonV1":
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ProspectKvHeuristicComparisonError("invalid JSON") from error
        if not isinstance(raw, dict):
            raise ProspectKvHeuristicComparisonError("comparison must be a JSON object")
        if json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False) != payload:
            raise ProspectKvHeuristicComparisonError("comparison JSON is not canonical")

        expected = {
            "schema",
            "effect",
            "budget_bytes",
            "random_seed",
            "results",
            "best_policy",
        }
        if set(raw) != expected:
            raise ProspectKvHeuristicComparisonError(
                "comparison fields do not match schema v1"
            )

        effect_raw = raw["effect"]
        if not isinstance(effect_raw, dict):
            raise ProspectKvHeuristicComparisonError("effect must be an object")
        effect_json = json.dumps(
            effect_raw,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        try:
            effect = ProspectKvEvictionEffectV1.from_canonical_json(effect_json)
        except ProspectKvEvictionEffectError as error:
            raise ProspectKvHeuristicComparisonError("invalid embedded effect") from error

        results_raw = raw["results"]
        if not isinstance(results_raw, list) or not results_raw:
            raise ProspectKvHeuristicComparisonError("results must be a non-empty array")
        results = tuple(_parse_result(item) for item in results_raw)

        comparison = cls(
            schema=_require_string(raw, "schema"),
            effect=effect,
            budget_bytes=_require_non_negative_int(raw, "budget_bytes"),
            random_seed=_require_u64(raw, "random_seed"),
            results=results,
            best_policy=_require_string(raw, "best_policy"),
        )
        comparison.validate_replay()
        return comparison

    def validate_replay(self) -> None:
        if self.schema != PROSPECT_KV_HEURISTIC_COMPARISON_SCHEMA_V1:
            raise ProspectKvHeuristicComparisonError("unsupported comparison schema")
        if self.budget_bytes != self.effect.eviction.logical_retained_bytes:
            raise ProspectKvHeuristicComparisonError(
                "comparison budget does not match eviction retained bytes"
            )
        if self.random_seed < 0 or self.random_seed > (2**64 - 1):
            raise ProspectKvHeuristicComparisonError("invalid random seed")

        replayed = _run_comparison(self.effect, self.random_seed)
        if len(self.results) != len(replayed):
            raise ProspectKvHeuristicComparisonError("result count does not match replay")
        for recorded, expected in zip(self.results, replayed, strict=True):
            if recorded.policy != expected.policy:
                raise ProspectKvHeuristicComparisonError("policy order does not match replay")
            if recorded.retained_region_ids != expected.retained_region_ids:
                raise ProspectKvHeuristicComparisonError(
                    f"retained regions do not match replay for {recorded.policy}"
                )
            if recorded.retained_bytes != expected.retained_bytes:
                raise ProspectKvHeuristicComparisonError(
                    f"retained bytes do not match replay for {recorded.policy}"
                )
            if recorded.unused_bytes != expected.unused_bytes:
                raise ProspectKvHeuristicComparisonError(
                    f"unused bytes do not match replay for {recorded.policy}"
                )
            if not _close(recorded.output_l2_delta, expected.output_l2_delta):
                raise ProspectKvHeuristicComparisonError(
                    f"L2 delta does not match replay for {recorded.policy}"
                )

        expected_best = _best_policy(replayed)
        if self.best_policy != expected_best:
            raise ProspectKvHeuristicComparisonError("best_policy does not match replay")


def _run_comparison(
    effect: ProspectKvEvictionEffectV1,
    random_seed: int,
) -> tuple[HeuristicResult, ...]:
    trace = _trace_from_effect(effect)
    budget = effect.eviction.logical_retained_bytes

    oldest = BudgetSelection(
        policy="oldest_first",
        budget_bytes=budget,
        retained_region_ids=effect.retained_region_ids,
        retained_bytes=budget,
    )
    selections = (
        oldest,
        select_lru_baseline(trace, budget),
        select_magnitude_baseline(trace, budget),
        select_sensitivity_per_byte_baseline(trace, budget),
        select_random_baseline(trace, budget, seed=random_seed),
    )
    evaluations = tuple(evaluate_selection(trace, selection) for selection in selections)
    return tuple(_result_from_evaluation(item) for item in evaluations)


def _trace_from_effect(effect: ProspectKvEvictionEffectV1) -> SyntheticKvTrace:
    try:
        return SyntheticKvTrace(
            effect.trace_id,
            tuple(
                KvRegion(
                    region.region_id,
                    region.storage_bytes,
                    region.contribution,
                )
                for region in effect.regions
            ),
        )
    except ValueError as error:
        raise ProspectKvHeuristicComparisonError("invalid synthetic trace") from error


def _result_from_evaluation(evaluation: SelectionEvaluation) -> HeuristicResult:
    return HeuristicResult(
        policy=evaluation.policy,
        retained_region_ids=evaluation.retained_region_ids,
        retained_bytes=evaluation.retained_bytes,
        unused_bytes=evaluation.unused_bytes,
        output_l2_delta=evaluation.output_l2_delta,
    )


def _best_policy(results: tuple[HeuristicResult, ...]) -> str:
    if not results:
        raise ProspectKvHeuristicComparisonError("comparison has no results")
    return min(results, key=lambda item: (item.output_l2_delta, item.policy)).policy


def _parse_result(raw: Any) -> HeuristicResult:
    if not isinstance(raw, dict):
        raise ProspectKvHeuristicComparisonError("result entries must be objects")
    expected = {
        "policy",
        "retained_region_ids",
        "retained_bytes",
        "unused_bytes",
        "output_l2_delta",
    }
    if set(raw) != expected:
        raise ProspectKvHeuristicComparisonError("result fields do not match schema v1")
    return HeuristicResult(
        policy=_require_string(raw, "policy"),
        retained_region_ids=_require_string_array(raw, "retained_region_ids"),
        retained_bytes=_require_non_negative_int(raw, "retained_bytes"),
        unused_bytes=_require_non_negative_int(raw, "unused_bytes"),
        output_l2_delta=_require_non_negative_float(raw, "output_l2_delta"),
    )


def _require_string(raw: dict[str, Any], field: str) -> str:
    value = raw[field]
    if not isinstance(value, str) or not value:
        raise ProspectKvHeuristicComparisonError(f"{field} must be a non-empty string")
    return value


def _require_string_array(raw: dict[str, Any], field: str) -> tuple[str, ...]:
    value = raw[field]
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ProspectKvHeuristicComparisonError(
            f"{field} must contain non-empty strings"
        )
    if len(value) != len(set(value)):
        raise ProspectKvHeuristicComparisonError(f"{field} values must be unique")
    return tuple(value)


def _require_non_negative_int(raw: dict[str, Any], field: str) -> int:
    value = raw[field]
    if type(value) is not int or value < 0:
        raise ProspectKvHeuristicComparisonError(
            f"{field} must be a non-negative integer"
        )
    return value


def _require_u64(raw: dict[str, Any], field: str) -> int:
    value = _require_non_negative_int(raw, field)
    if value > (2**64 - 1):
        raise ProspectKvHeuristicComparisonError(f"{field} exceeds u64")
    return value


def _require_non_negative_float(raw: dict[str, Any], field: str) -> float:
    value = raw[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProspectKvHeuristicComparisonError(f"{field} must be numeric")
    converted = float(value)
    if not isfinite(converted) or converted < 0.0:
        raise ProspectKvHeuristicComparisonError(
            f"{field} must be a finite non-negative number"
        )
    return converted


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= 1.0e-12 + 1.0e-12 * max(abs(left), abs(right))
