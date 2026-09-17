"""Preregistered analysis-plan contract for BIKV target-host/model campaigns.

The contract freezes how paired target evidence will be interpreted before
outcome inspection. It validates against the already-frozen target protocol,
but it does not inspect measurements, compute uncertainty, choose a winner, or
authorize BKV-K9/NBKV promotion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping

from .bikv_target_protocol import REQUIRED_METRICS, BikvTargetProtocolV1


BKV_TARGET_ANALYSIS_PLAN_SCHEMA_V1 = "kvlab.bikv-target-analysis-plan.v1"
BKV_TARGET_PAIRED_SUMMARY_SCHEMA_V1 = "kvlab.bikv-target-paired-summary.v1"
DIRECTIONS = frozenset({"lower_is_better", "higher_is_better"})
UNCERTAINTY_METHODS = frozenset({"paired_percentile_bootstrap", "paired_t_interval"})
INCOMPLETE_PAIR_POLICY = "fail_closed"
PROMOTION_RULE = (
    "all_required_pairs_complete_and_primary_interval_meets_effect_"
    "and_quality_guard_passes"
)
_BOOLEAN_METRICS = frozenset({"reset_reuse_correctness"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FIELDS = frozenset(
    {
        "schema",
        "protocol_sha256",
        "campaign_id",
        "paired_summary_schema",
        "hypothesis_h0",
        "hypothesis_h1",
        "primary_performance_metric",
        "primary_direction",
        "primary_minimum_effect",
        "primary_unit",
        "quality_metric",
        "quality_direction",
        "quality_noninferiority_margin",
        "quality_unit",
        "uncertainty_method",
        "confidence_level_ppm",
        "bootstrap_resamples",
        "bootstrap_seed",
        "required_complete_pairs",
        "incomplete_pair_policy",
        "multiplicity_policy",
        "holdout_policy",
        "created_before_outcome_inspection",
        "tuning_permitted",
        "promotion_rule",
    }
)


class BikvTargetAnalysisPlanError(ValueError):
    """Raised when a BIKV target analysis plan is unsafe or incomplete."""


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise BikvTargetAnalysisPlanError(f"{name} must be non-empty text")
    if value != value.strip():
        raise BikvTargetAnalysisPlanError(f"{name} must not contain leading/trailing whitespace")
    return value


def _sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise BikvTargetAnalysisPlanError(f"{name} must be a lowercase 64-hex SHA-256")
    return value


def _finite_nonnegative(name: str, value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BikvTargetAnalysisPlanError(f"{name} must be numeric")
    if isinstance(value, float) and not math.isfinite(value):
        raise BikvTargetAnalysisPlanError(f"{name} must be finite")
    if value < 0:
        raise BikvTargetAnalysisPlanError(f"{name} must be >= 0")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BikvTargetAnalysisPlanError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class BikvTargetAnalysisPlanV1:
    schema: str
    protocol_sha256: str
    campaign_id: str
    paired_summary_schema: str
    hypothesis_h0: str
    hypothesis_h1: str
    primary_performance_metric: str
    primary_direction: str
    primary_minimum_effect: int | float
    primary_unit: str
    quality_metric: str
    quality_direction: str
    quality_noninferiority_margin: int | float
    quality_unit: str
    uncertainty_method: str
    confidence_level_ppm: int
    bootstrap_resamples: int | None
    bootstrap_seed: int | None
    required_complete_pairs: int
    incomplete_pair_policy: str
    multiplicity_policy: str
    holdout_policy: str
    created_before_outcome_inspection: bool
    tuning_permitted: bool
    promotion_rule: str = PROMOTION_RULE

    def validate(self) -> None:
        if self.schema != BKV_TARGET_ANALYSIS_PLAN_SCHEMA_V1:
            raise BikvTargetAnalysisPlanError("unsupported BIKV target analysis-plan schema")
        _sha256("protocol_sha256", self.protocol_sha256)
        _text("campaign_id", self.campaign_id)
        if self.paired_summary_schema != BKV_TARGET_PAIRED_SUMMARY_SCHEMA_V1:
            raise BikvTargetAnalysisPlanError("unsupported paired summary schema")
        _text("hypothesis_h0", self.hypothesis_h0)
        _text("hypothesis_h1", self.hypothesis_h1)
        if self.primary_performance_metric not in REQUIRED_METRICS:
            raise BikvTargetAnalysisPlanError("primary_performance_metric must be a required metric")
        if self.primary_performance_metric in _BOOLEAN_METRICS:
            raise BikvTargetAnalysisPlanError("primary_performance_metric must be numeric")
        if self.primary_direction not in DIRECTIONS:
            raise BikvTargetAnalysisPlanError(f"primary_direction must be one of {sorted(DIRECTIONS)}")
        _finite_nonnegative("primary_minimum_effect", self.primary_minimum_effect)
        _text("primary_unit", self.primary_unit)
        if self.quality_metric not in REQUIRED_METRICS:
            raise BikvTargetAnalysisPlanError("quality_metric must be a required metric")
        if self.quality_metric in _BOOLEAN_METRICS:
            raise BikvTargetAnalysisPlanError("quality_metric must be numeric in analysis-plan v1")
        if self.quality_direction not in DIRECTIONS:
            raise BikvTargetAnalysisPlanError(f"quality_direction must be one of {sorted(DIRECTIONS)}")
        _finite_nonnegative("quality_noninferiority_margin", self.quality_noninferiority_margin)
        _text("quality_unit", self.quality_unit)
        if self.uncertainty_method not in UNCERTAINTY_METHODS:
            raise BikvTargetAnalysisPlanError(
                f"uncertainty_method must be one of {sorted(UNCERTAINTY_METHODS)}"
            )
        if (
            isinstance(self.confidence_level_ppm, bool)
            or not isinstance(self.confidence_level_ppm, int)
            or not 500_001 <= self.confidence_level_ppm <= 999_999
        ):
            raise BikvTargetAnalysisPlanError(
                "confidence_level_ppm must be an integer in [500001, 999999]"
            )
        if self.uncertainty_method == "paired_percentile_bootstrap":
            if self.bootstrap_resamples is None or self.bootstrap_seed is None:
                raise BikvTargetAnalysisPlanError(
                    "paired_percentile_bootstrap requires bootstrap_resamples and bootstrap_seed"
                )
            if self.bootstrap_resamples < 1:
                raise BikvTargetAnalysisPlanError("bootstrap_resamples must be >= 1")
            _nonnegative_int("bootstrap_seed", self.bootstrap_seed)
        else:
            if self.bootstrap_resamples is not None or self.bootstrap_seed is not None:
                raise BikvTargetAnalysisPlanError(
                    "paired_t_interval must not carry bootstrap parameters"
                )
        if (
            isinstance(self.required_complete_pairs, bool)
            or not isinstance(self.required_complete_pairs, int)
            or self.required_complete_pairs < 1
        ):
            raise BikvTargetAnalysisPlanError("required_complete_pairs must be an integer >= 1")
        if self.incomplete_pair_policy != INCOMPLETE_PAIR_POLICY:
            raise BikvTargetAnalysisPlanError(
                f"incomplete_pair_policy must be {INCOMPLETE_PAIR_POLICY!r}"
            )
        _text("multiplicity_policy", self.multiplicity_policy)
        _text("holdout_policy", self.holdout_policy)
        if self.created_before_outcome_inspection is not True:
            raise BikvTargetAnalysisPlanError(
                "created_before_outcome_inspection must be true for a preregistered plan"
            )
        if self.tuning_permitted is not False:
            raise BikvTargetAnalysisPlanError("analysis-plan v1 cannot permit tuning")
        if self.promotion_rule != PROMOTION_RULE:
            raise BikvTargetAnalysisPlanError("promotion_rule must match the fail-closed v1 rule")

    def validate_against(self, protocol: BikvTargetProtocolV1) -> None:
        self.validate()
        protocol.validate()
        if self.protocol_sha256 != protocol.protocol_sha256():
            raise BikvTargetAnalysisPlanError("analysis plan does not bind the supplied protocol")
        if self.campaign_id != protocol.campaign_id:
            raise BikvTargetAnalysisPlanError("campaign_id does not match protocol")
        if self.hypothesis_h0 != protocol.hypothesis_h0 or self.hypothesis_h1 != protocol.hypothesis_h1:
            raise BikvTargetAnalysisPlanError("analysis plan must preserve the protocol H0/H1 verbatim")
        if self.quality_metric != protocol.quality_metric:
            raise BikvTargetAnalysisPlanError("quality_metric must match the frozen protocol")
        if self.holdout_policy != protocol.holdout_policy:
            raise BikvTargetAnalysisPlanError("holdout_policy must match the frozen protocol")
        planned_pairs = len(protocol.seeds) * protocol.repetitions
        if self.required_complete_pairs != planned_pairs:
            raise BikvTargetAnalysisPlanError(
                "required_complete_pairs must equal every preregistered seed/repetition pair"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    def plan_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BikvTargetAnalysisPlanV1":
        if not isinstance(raw, Mapping):
            raise BikvTargetAnalysisPlanError("analysis plan must be a JSON object")
        unknown = set(raw) - _FIELDS
        missing = _FIELDS - set(raw)
        if unknown:
            raise BikvTargetAnalysisPlanError(f"unknown analysis-plan fields: {sorted(unknown)}")
        if missing:
            raise BikvTargetAnalysisPlanError(f"missing analysis-plan fields: {sorted(missing)}")
        plan = cls(**{key: raw[key] for key in _FIELDS})
        plan.validate()
        return plan

    @classmethod
    def from_canonical_json(cls, text: str) -> "BikvTargetAnalysisPlanV1":
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BikvTargetAnalysisPlanError("analysis plan must be valid JSON") from exc
        plan = cls.from_mapping(raw)
        if text != plan.canonical_json():
            raise BikvTargetAnalysisPlanError("analysis plan JSON must use canonical encoding")
        return plan
