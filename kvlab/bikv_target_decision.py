"""Fail-closed decision evaluator for retained BIKV target campaigns.

This module closes a deliberate preregistration gap without changing any
existing v1 analysis plan.  A v2 decision plan wraps the already-content-
addressed v1 plan and freezes the previously implicit effect statistic,
interval semantics and candidate-side correctness guards before outcomes are
inspected.  The evaluator rebuilds the paired summary from retained runs; it
never trusts a caller-supplied aggregate.

A decision record is evidence about the declared campaign only.  It does not
open a protected holdout, authorize BKV-K9/NBKV, or turn logical byte counts
into physical traffic claims.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
import statistics
from typing import Any, Iterable, Mapping

from .bikv_target_analysis_plan import BikvTargetAnalysisPlanV1
from .bikv_target_campaign import BikvTargetCampaignV1
from .bikv_target_paired_summary import (
    BikvTargetPairedSummaryV1,
    PairedMetricSummaryV1,
    build_paired_summary,
)
from .bikv_target_protocol import REQUIRED_METRICS, BikvTargetProtocolV1
from .bikv_target_run import BikvTargetRunV1


BKV_TARGET_DECISION_PLAN_SCHEMA_V2 = "kvlab.bikv-target-decision-plan.v2"
BKV_TARGET_DECISION_SCHEMA_V1 = "kvlab.bikv-target-decision.v1"
EFFECT_STATISTICS = frozenset({"paired_mean", "paired_median"})
BOOTSTRAP_QUANTILE_METHOD = "linear_type7"
T_INTERVAL_METHOD = "student_t_equal_tail"
DECISION_ORIENTATION = "improvement_positive"
GUARD_OPERATORS = frozenset({"ge", "le", "eq_true"})
CORE_GUARDED_METRICS = frozenset(
    {
        "candidate_recall",
        "candidate_false_negative_rate",
        "o_error",
        "lse_error",
        "reset_reuse_correctness",
    }
)
_BOOLEAN_METRICS = frozenset({"reset_reuse_correctness"})
_PLAN_FIELDS = frozenset(
    {
        "schema",
        "base_plan",
        "primary_statistic",
        "quality_statistic",
        "bootstrap_quantile_method",
        "t_interval_method",
        "decision_orientation",
        "candidate_metric_guards",
    }
)
_GUARD_FIELDS = frozenset({"metric", "operator", "threshold", "unit"})


class BikvTargetDecisionError(ValueError):
    """Raised when a target decision plan or evidence binding is invalid."""


def _finite_number(name: str, value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BikvTargetDecisionError(f"{name} must be numeric")
    if isinstance(value, float) and not math.isfinite(value):
        raise BikvTargetDecisionError(f"{name} must be finite")
    return value


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise BikvTargetDecisionError(f"{name} must be non-empty trimmed text")
    return value


@dataclass(frozen=True, slots=True)
class BikvCandidateMetricGuardV1:
    """One preregistered candidate-side metric acceptance guard."""

    metric: str
    operator: str
    threshold: int | float | None
    unit: str

    def validate(self) -> None:
        if self.metric not in REQUIRED_METRICS:
            raise BikvTargetDecisionError(f"unknown guarded metric: {self.metric!r}")
        if self.operator not in GUARD_OPERATORS:
            raise BikvTargetDecisionError(
                f"guard operator must be one of {sorted(GUARD_OPERATORS)}"
            )
        _text("guard unit", self.unit)
        if self.operator == "eq_true":
            if self.metric not in _BOOLEAN_METRICS:
                raise BikvTargetDecisionError("eq_true is valid only for Boolean metrics")
            if self.threshold is not None:
                raise BikvTargetDecisionError("eq_true guard must not carry threshold")
            if self.unit != "boolean":
                raise BikvTargetDecisionError("Boolean guard unit must be 'boolean'")
            return
        if self.metric in _BOOLEAN_METRICS:
            raise BikvTargetDecisionError("Boolean metrics require eq_true guard")
        _finite_number("guard threshold", self.threshold)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BikvCandidateMetricGuardV1":
        if not isinstance(raw, Mapping) or frozenset(raw) != _GUARD_FIELDS:
            raise BikvTargetDecisionError("metric guard fields do not match v1 schema")
        guard = cls(
            metric=raw["metric"],
            operator=raw["operator"],
            threshold=raw["threshold"],
            unit=raw["unit"],
        )
        guard.validate()
        return guard


@dataclass(frozen=True, slots=True)
class BikvTargetDecisionPlanV2:
    """Exact statistical/guard semantics layered over a frozen v1 plan."""

    schema: str
    base_plan: BikvTargetAnalysisPlanV1
    primary_statistic: str
    quality_statistic: str
    bootstrap_quantile_method: str | None
    t_interval_method: str | None
    decision_orientation: str
    candidate_metric_guards: tuple[BikvCandidateMetricGuardV1, ...]

    def validate(self) -> None:
        if self.schema != BKV_TARGET_DECISION_PLAN_SCHEMA_V2:
            raise BikvTargetDecisionError("unsupported BIKV target decision-plan schema")
        self.base_plan.validate()
        if self.primary_statistic not in EFFECT_STATISTICS:
            raise BikvTargetDecisionError(
                f"primary_statistic must be one of {sorted(EFFECT_STATISTICS)}"
            )
        if self.quality_statistic not in EFFECT_STATISTICS:
            raise BikvTargetDecisionError(
                f"quality_statistic must be one of {sorted(EFFECT_STATISTICS)}"
            )
        if self.decision_orientation != DECISION_ORIENTATION:
            raise BikvTargetDecisionError(
                f"decision_orientation must be {DECISION_ORIENTATION!r}"
            )
        if self.base_plan.uncertainty_method == "paired_percentile_bootstrap":
            if (
                self.base_plan.bootstrap_seed is None
                or self.base_plan.bootstrap_seed > (1 << 64) - 1
            ):
                raise BikvTargetDecisionError(
                    "v2 SplitMix64 bootstrap requires bootstrap_seed to fit u64"
                )
            if self.bootstrap_quantile_method != BOOTSTRAP_QUANTILE_METHOD:
                raise BikvTargetDecisionError(
                    f"bootstrap_quantile_method must be {BOOTSTRAP_QUANTILE_METHOD!r}"
                )
            if self.t_interval_method is not None:
                raise BikvTargetDecisionError(
                    "bootstrap plan must not carry a t-interval method"
                )
        elif self.base_plan.uncertainty_method == "paired_t_interval":
            if self.t_interval_method != T_INTERVAL_METHOD:
                raise BikvTargetDecisionError(
                    f"t_interval_method must be {T_INTERVAL_METHOD!r}"
                )
            if self.bootstrap_quantile_method is not None:
                raise BikvTargetDecisionError(
                    "paired-t plan must not carry a bootstrap quantile method"
                )
            if self.primary_statistic != "paired_mean" or self.quality_statistic != "paired_mean":
                raise BikvTargetDecisionError(
                    "paired_t_interval requires paired_mean primary and quality statistics"
                )
        else:  # base-plan v1 already rejects this; retain fail-closed boundary.
            raise BikvTargetDecisionError("unsupported uncertainty method")

        if not isinstance(self.candidate_metric_guards, tuple):
            raise BikvTargetDecisionError("candidate_metric_guards must be a tuple")
        names: list[str] = []
        for guard in self.candidate_metric_guards:
            guard.validate()
            names.append(guard.metric)
        if len(set(names)) != len(names):
            raise BikvTargetDecisionError("candidate metric guards must not contain duplicates")
        missing = CORE_GUARDED_METRICS - set(names)
        if missing:
            raise BikvTargetDecisionError(
                f"candidate metric guards are missing core gates: {sorted(missing)}"
            )

    def validate_against(self, protocol: BikvTargetProtocolV1) -> None:
        self.validate()
        self.base_plan.validate_against(protocol)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "base_plan": self.base_plan.to_dict(),
            "primary_statistic": self.primary_statistic,
            "quality_statistic": self.quality_statistic,
            "bootstrap_quantile_method": self.bootstrap_quantile_method,
            "t_interval_method": self.t_interval_method,
            "decision_orientation": self.decision_orientation,
            "candidate_metric_guards": [guard.to_dict() for guard in self.candidate_metric_guards],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    def plan_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BikvTargetDecisionPlanV2":
        if not isinstance(raw, Mapping) or frozenset(raw) != _PLAN_FIELDS:
            raise BikvTargetDecisionError("decision-plan fields do not match v2 schema")
        base_raw = raw["base_plan"]
        if not isinstance(base_raw, Mapping):
            raise BikvTargetDecisionError("base_plan must be a JSON object")
        base_plan = BikvTargetAnalysisPlanV1.from_mapping(base_raw)
        guards_raw = raw["candidate_metric_guards"]
        if not isinstance(guards_raw, list):
            raise BikvTargetDecisionError("candidate_metric_guards must be a JSON array")
        plan = cls(
            schema=raw["schema"],
            base_plan=base_plan,
            primary_statistic=raw["primary_statistic"],
            quality_statistic=raw["quality_statistic"],
            bootstrap_quantile_method=raw["bootstrap_quantile_method"],
            t_interval_method=raw["t_interval_method"],
            decision_orientation=raw["decision_orientation"],
            candidate_metric_guards=tuple(
                BikvCandidateMetricGuardV1.from_mapping(item) for item in guards_raw
            ),
        )
        plan.validate()
        return plan

    @classmethod
    def from_canonical_json(cls, text: str) -> "BikvTargetDecisionPlanV2":
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BikvTargetDecisionError("decision plan must be valid JSON") from exc
        plan = cls.from_mapping(raw)
        if text != plan.canonical_json():
            raise BikvTargetDecisionError("decision plan JSON must use canonical encoding")
        return plan


@dataclass(frozen=True, slots=True)
class BikvIntervalDecisionV1:
    metric: str
    statistic: str
    direction: str
    measured_pairs: int
    required_pairs: int
    unit: str
    point_estimate_improvement: float | None
    interval_low_improvement: float | None
    interval_high_improvement: float | None
    required_floor_improvement: float
    passed: bool
    blocker: str | None


@dataclass(frozen=True, slots=True)
class BikvMetricGuardDecisionV1:
    metric: str
    operator: str
    threshold: int | float | None
    unit: str
    measured_candidate_attempts: int
    required_candidate_attempts: int
    passed: bool
    blocker: str | None


@dataclass(frozen=True, slots=True)
class BikvTargetDecisionV1:
    schema: str
    protocol_sha256: str
    campaign_sha256: str
    paired_summary_sha256: str
    decision_plan_sha256: str
    campaign_id: str
    primary: BikvIntervalDecisionV1
    quality: BikvIntervalDecisionV1
    metric_guards: tuple[BikvMetricGuardDecisionV1, ...]
    disposition: str
    interpretation: str

    def canonical_json_bytes(self) -> bytes:
        return json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    def decision_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()


class _SplitMix64:
    """Small fully specified PRNG for reproducible bootstrap resampling."""

    _MASK = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        self.state = seed & self._MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self._MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self._MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self._MASK
        return (z ^ (z >> 31)) & self._MASK

    def randbelow(self, bound: int) -> int:
        if bound < 1:
            raise BikvTargetDecisionError("bootstrap sample bound must be positive")
        limit = (1 << 64) - ((1 << 64) % bound)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % bound


def _statistic(values: list[float], name: str) -> float:
    if not values:
        raise BikvTargetDecisionError("cannot compute effect statistic from zero pairs")
    if name == "paired_mean":
        return float(statistics.fmean(values))
    if name == "paired_median":
        return float(statistics.median(values))
    raise BikvTargetDecisionError(f"unsupported effect statistic {name!r}")


def _type7_quantile(sorted_values: list[float], probability: Fraction) -> float:
    if not sorted_values:
        raise BikvTargetDecisionError("cannot compute quantile from empty sample")
    if probability < 0 or probability > 1:
        raise BikvTargetDecisionError("quantile probability must be in [0, 1]")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = probability * (len(sorted_values) - 1)
    low = rank.numerator // rank.denominator
    frac = rank - low
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + float(frac) * (sorted_values[high] - sorted_values[low])


def _bootstrap_interval(
    values: list[float], *, statistic: str, confidence_ppm: int, resamples: int, seed: int
) -> tuple[float, float, float]:
    rng = _SplitMix64(seed)
    observed = _statistic(values, statistic)
    draws: list[float] = []
    for _ in range(resamples):
        sample = [values[rng.randbelow(len(values))] for _ in values]
        draws.append(_statistic(sample, statistic))
    draws.sort()
    tail = Fraction(1_000_000 - confidence_ppm, 2_000_000)
    return observed, _type7_quantile(draws, tail), _type7_quantile(draws, 1 - tail)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta function."""
    max_iter = 300
    eps = 3.0e-14
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= eps:
            return h
    raise BikvTargetDecisionError("incomplete-beta continued fraction did not converge")


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _student_t_cdf(value: float, degrees_freedom: int) -> float:
    if degrees_freedom < 1:
        raise BikvTargetDecisionError("Student-t interval requires positive degrees of freedom")
    if value == 0.0:
        return 0.5
    x = degrees_freedom / (degrees_freedom + value * value)
    ibeta = _regularized_beta(x, degrees_freedom / 2.0, 0.5)
    return 1.0 - 0.5 * ibeta if value > 0 else 0.5 * ibeta


def _student_t_quantile(probability: float, degrees_freedom: int) -> float:
    if not 0.5 < probability < 1.0:
        raise BikvTargetDecisionError("upper Student-t probability must be in (0.5, 1)")
    low, high = 0.0, 1.0
    while _student_t_cdf(high, degrees_freedom) < probability:
        high *= 2.0
        if high > 1.0e8:
            raise BikvTargetDecisionError("failed to bracket Student-t quantile")
    for _ in range(100):
        mid = (low + high) / 2.0
        if _student_t_cdf(mid, degrees_freedom) < probability:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _paired_t_interval(values: list[float], confidence_ppm: int) -> tuple[float, float, float]:
    if len(values) < 2:
        raise BikvTargetDecisionError("paired_t_interval requires at least two complete pairs")
    mean = float(statistics.fmean(values))
    stddev = statistics.stdev(values)
    alpha_half = (1_000_000 - confidence_ppm) / 2_000_000.0
    critical = _student_t_quantile(1.0 - alpha_half, len(values) - 1)
    half_width = critical * stddev / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def _improvements(metric: PairedMetricSummaryV1, direction: str) -> list[float]:
    result: list[float] = []
    for point in metric.points:
        if (
            point.baseline_run_status != "completed"
            or point.candidate_run_status != "completed"
            or point.baseline_metric_status != "measured"
            or point.candidate_metric_status != "measured"
            or point.delta_candidate_minus_baseline is None
        ):
            continue
        delta = float(point.delta_candidate_minus_baseline)
        result.append(-delta if direction == "lower_is_better" else delta)
    return result


def _interval_decision(
    *,
    metric: PairedMetricSummaryV1,
    statistic: str,
    direction: str,
    required_pairs: int,
    expected_unit: str,
    required_floor: float,
    plan: BikvTargetDecisionPlanV2,
) -> BikvIntervalDecisionV1:
    if metric.measured_pairs != required_pairs or metric.incomplete_pairs != 0:
        return BikvIntervalDecisionV1(
            metric=metric.name,
            statistic=statistic,
            direction=direction,
            measured_pairs=metric.measured_pairs,
            required_pairs=required_pairs,
            unit=expected_unit,
            point_estimate_improvement=None,
            interval_low_improvement=None,
            interval_high_improvement=None,
            required_floor_improvement=required_floor,
            passed=False,
            blocker="incomplete paired evidence",
        )
    if metric.unit != expected_unit:
        raise BikvTargetDecisionError(
            f"metric {metric.name!r} unit {metric.unit!r} does not match preregistered {expected_unit!r}"
        )
    values = _improvements(metric, direction)
    if len(values) != required_pairs:
        raise BikvTargetDecisionError(
            f"metric {metric.name!r} measured-pair count disagrees with retained points"
        )
    base = plan.base_plan
    try:
        if base.uncertainty_method == "paired_percentile_bootstrap":
            assert base.bootstrap_resamples is not None and base.bootstrap_seed is not None
            point, low, high = _bootstrap_interval(
                values,
                statistic=statistic,
                confidence_ppm=base.confidence_level_ppm,
                resamples=base.bootstrap_resamples,
                seed=base.bootstrap_seed,
            )
        else:
            point, low, high = _paired_t_interval(values, base.confidence_level_ppm)
    except BikvTargetDecisionError as exc:
        return BikvIntervalDecisionV1(
            metric=metric.name,
            statistic=statistic,
            direction=direction,
            measured_pairs=metric.measured_pairs,
            required_pairs=required_pairs,
            unit=expected_unit,
            point_estimate_improvement=None,
            interval_low_improvement=None,
            interval_high_improvement=None,
            required_floor_improvement=required_floor,
            passed=False,
            blocker=str(exc),
        )
    return BikvIntervalDecisionV1(
        metric=metric.name,
        statistic=statistic,
        direction=direction,
        measured_pairs=metric.measured_pairs,
        required_pairs=required_pairs,
        unit=expected_unit,
        point_estimate_improvement=point,
        interval_low_improvement=low,
        interval_high_improvement=high,
        required_floor_improvement=required_floor,
        passed=low >= required_floor,
        blocker=None,
    )


def _guard_decisions(
    *,
    plan: BikvTargetDecisionPlanV2,
    protocol: BikvTargetProtocolV1,
    runs: tuple[BikvTargetRunV1, ...],
) -> tuple[BikvMetricGuardDecisionV1, ...]:
    candidate_runs = tuple(run for run in runs if run.variant == "candidate")
    expected = len(protocol.seeds) * protocol.repetitions
    if len(candidate_runs) != expected:
        raise BikvTargetDecisionError("candidate run count does not match frozen campaign slots")
    decisions: list[BikvMetricGuardDecisionV1] = []
    for guard in plan.candidate_metric_guards:
        measured = 0
        passed = True
        blocker: str | None = None
        for run in candidate_runs:
            metric = next(item for item in run.metrics if item.name == guard.metric)
            if run.status != "completed" or metric.status != "measured":
                passed = False
                blocker = "candidate guard evidence incomplete"
                continue
            measured += 1
            if metric.unit != guard.unit:
                raise BikvTargetDecisionError(
                    f"guard metric {guard.metric!r} unit does not match preregistered unit"
                )
            if guard.operator == "eq_true":
                current = metric.value is True
            else:
                value = _finite_number(f"guard metric {guard.metric}", metric.value)
                assert guard.threshold is not None
                current = value >= guard.threshold if guard.operator == "ge" else value <= guard.threshold
            if not current:
                passed = False
        if measured != expected:
            passed = False
            blocker = blocker or "candidate guard evidence incomplete"
        decisions.append(
            BikvMetricGuardDecisionV1(
                metric=guard.metric,
                operator=guard.operator,
                threshold=guard.threshold,
                unit=guard.unit,
                measured_candidate_attempts=measured,
                required_candidate_attempts=expected,
                passed=passed,
                blocker=blocker,
            )
        )
    return tuple(decisions)


def evaluate_target_campaign(
    *,
    protocol: BikvTargetProtocolV1,
    campaign: BikvTargetCampaignV1,
    runs: Iterable[BikvTargetRunV1],
    decision_plan: BikvTargetDecisionPlanV2,
) -> BikvTargetDecisionV1:
    """Apply an already-frozen v2 decision plan to retained campaign evidence."""

    decision_plan.validate_against(protocol)
    materialized = tuple(runs)
    campaign.validate_against(protocol)
    campaign.verify_runs(protocol=protocol, runs=materialized)
    summary = build_paired_summary(protocol=protocol, campaign=campaign, runs=materialized)
    summaries = {metric.name: metric for metric in summary.metrics}
    base = decision_plan.base_plan
    required = base.required_complete_pairs

    primary = _interval_decision(
        metric=summaries[base.primary_performance_metric],
        statistic=decision_plan.primary_statistic,
        direction=base.primary_direction,
        required_pairs=required,
        expected_unit=base.primary_unit,
        required_floor=float(base.primary_minimum_effect),
        plan=decision_plan,
    )
    quality = _interval_decision(
        metric=summaries[base.quality_metric],
        statistic=decision_plan.quality_statistic,
        direction=base.quality_direction,
        required_pairs=required,
        expected_unit=base.quality_unit,
        required_floor=-float(base.quality_noninferiority_margin),
        plan=decision_plan,
    )
    guards = _guard_decisions(
        plan=decision_plan, protocol=protocol, runs=materialized
    )

    complete = primary.blocker is None and quality.blocker is None and all(
        guard.blocker is None for guard in guards
    )
    if not complete:
        disposition = "blocked_incomplete_evidence"
    elif primary.passed and quality.passed and all(guard.passed for guard in guards):
        disposition = "candidate_meets_preregistered_gate"
    else:
        disposition = "candidate_does_not_meet_preregistered_gate"

    return BikvTargetDecisionV1(
        schema=BKV_TARGET_DECISION_SCHEMA_V1,
        protocol_sha256=protocol.protocol_sha256(),
        campaign_sha256=campaign.campaign_sha256(),
        paired_summary_sha256=summary.summary_sha256(),
        decision_plan_sha256=decision_plan.plan_sha256(),
        campaign_id=protocol.campaign_id,
        primary=primary,
        quality=quality,
        metric_guards=guards,
        disposition=disposition,
        interpretation=(
            "campaign-local preregistered decision evidence only; does not authorize "
            "BKV-K9/NBKV, protected-holdout access, adaptive routing, physical-traffic "
            "claims, energy claims, or general hardware/model performance"
        ),
    )
