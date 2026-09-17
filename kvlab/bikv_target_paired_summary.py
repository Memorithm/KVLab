"""Paired descriptive summary for a complete BIKV target campaign.

The summary is intentionally non-inferential.  It verifies the retained campaign
against its frozen protocol, pairs baseline/candidate attempts by seed and
repetition, preserves unavailable/failed metric states, and emits only
reproducible descriptive deltas.  It does not choose a winner or authorize
BKV-K9/NBKV promotion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import statistics
from typing import Any, Iterable

from .bikv_target_campaign import BikvTargetCampaignV1
from .bikv_target_protocol import REQUIRED_METRICS, BikvTargetProtocolV1
from .bikv_target_run import BikvMetricObservationV1, BikvTargetRunV1


BKV_TARGET_PAIRED_SUMMARY_SCHEMA_V1 = "kvlab.bikv-target-paired-summary.v1"
_BOOLEAN_METRICS = frozenset({"reset_reuse_correctness"})


class BikvTargetPairedSummaryError(ValueError):
    """Raised when paired target evidence cannot be summarized safely."""


def _metric_map(run: BikvTargetRunV1) -> dict[str, BikvMetricObservationV1]:
    run.validate()
    return {metric.name: metric for metric in run.metrics}


def _numeric(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BikvTargetPairedSummaryError("paired numeric metric has non-numeric value")
    if isinstance(value, float) and not math.isfinite(value):
        raise BikvTargetPairedSummaryError("paired numeric metric must be finite")
    return value


@dataclass(frozen=True, slots=True)
class PairedMetricPointV1:
    seed: int
    repetition_index: int
    baseline_run_status: str
    candidate_run_status: str
    baseline_metric_status: str
    candidate_metric_status: str
    baseline_value: int | float | bool | None
    candidate_value: int | float | bool | None
    unit: str | None
    baseline_reason: str | None
    candidate_reason: str | None
    delta_candidate_minus_baseline: int | float | None
    boolean_equal: bool | None


@dataclass(frozen=True, slots=True)
class PairedMetricSummaryV1:
    name: str
    total_pairs: int
    measured_pairs: int
    incomplete_pairs: int
    unit: str | None
    median_delta_candidate_minus_baseline: int | float | None
    min_delta_candidate_minus_baseline: int | float | None
    max_delta_candidate_minus_baseline: int | float | None
    boolean_equal_pairs: int | None
    points: tuple[PairedMetricPointV1, ...]


@dataclass(frozen=True, slots=True)
class BikvTargetPairedSummaryV1:
    schema: str
    protocol_sha256: str
    campaign_sha256: str
    campaign_id: str
    interpretation: str
    metrics: tuple[PairedMetricSummaryV1, ...]

    def canonical_json_bytes(self) -> bytes:
        payload = asdict(self)
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    def summary_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()


def _pair_point(
    *,
    seed: int,
    repetition_index: int,
    name: str,
    baseline: BikvTargetRunV1,
    candidate: BikvTargetRunV1,
) -> PairedMetricPointV1:
    base_metric = _metric_map(baseline)[name]
    cand_metric = _metric_map(candidate)[name]

    unit: str | None = None
    delta: int | float | None = None
    boolean_equal: bool | None = None
    if base_metric.status == "measured" and cand_metric.status == "measured":
        if base_metric.unit != cand_metric.unit:
            raise BikvTargetPairedSummaryError(
                f"metric {name!r} unit mismatch for seed={seed} repetition={repetition_index}"
            )
        unit = base_metric.unit
        if name in _BOOLEAN_METRICS:
            if not isinstance(base_metric.value, bool) or not isinstance(cand_metric.value, bool):
                raise BikvTargetPairedSummaryError(
                    f"metric {name!r} must retain Boolean measured values"
                )
            boolean_equal = base_metric.value == cand_metric.value
        else:
            base_value = _numeric(base_metric.value)
            cand_value = _numeric(cand_metric.value)
            delta = cand_value - base_value

    return PairedMetricPointV1(
        seed=seed,
        repetition_index=repetition_index,
        baseline_run_status=baseline.status,
        candidate_run_status=candidate.status,
        baseline_metric_status=base_metric.status,
        candidate_metric_status=cand_metric.status,
        baseline_value=base_metric.value,
        candidate_value=cand_metric.value,
        unit=unit,
        baseline_reason=base_metric.reason,
        candidate_reason=cand_metric.reason,
        delta_candidate_minus_baseline=delta,
        boolean_equal=boolean_equal,
    )


def build_paired_summary(
    *,
    protocol: BikvTargetProtocolV1,
    campaign: BikvTargetCampaignV1,
    runs: Iterable[BikvTargetRunV1],
) -> BikvTargetPairedSummaryV1:
    """Verify and summarize a frozen baseline/candidate campaign.

    Missing, failed, or not-exposed measurements stay visible as incomplete
    points.  Numeric deltas are descriptive only; no p-value, confidence
    interval, superiority decision, or promotion status is synthesized.
    """

    protocol.validate()
    materialized = tuple(runs)
    campaign.validate_against(protocol)
    campaign.verify_runs(protocol=protocol, runs=materialized)

    indexed: dict[tuple[str, int, int], BikvTargetRunV1] = {}
    for run in materialized:
        run.validate_against(protocol)
        key = (run.variant, run.seed, run.repetition_index)
        if key in indexed:
            raise BikvTargetPairedSummaryError("duplicate retained run slot")
        indexed[key] = run

    summaries: list[PairedMetricSummaryV1] = []
    for name in REQUIRED_METRICS:
        points: list[PairedMetricPointV1] = []
        observed_units: set[str] = set()
        for seed in protocol.seeds:
            for repetition_index in range(protocol.repetitions):
                baseline = indexed[("baseline", seed, repetition_index)]
                candidate = indexed[("candidate", seed, repetition_index)]
                point = _pair_point(
                    seed=seed,
                    repetition_index=repetition_index,
                    name=name,
                    baseline=baseline,
                    candidate=candidate,
                )
                points.append(point)
                if point.unit is not None:
                    observed_units.add(point.unit)

        if len(observed_units) > 1:
            raise BikvTargetPairedSummaryError(
                f"metric {name!r} changes unit across measured pairs"
            )
        unit = next(iter(observed_units), None)
        complete = [
            point
            for point in points
            if point.baseline_run_status == "completed"
            and point.candidate_run_status == "completed"
            and point.baseline_metric_status == "measured"
            and point.candidate_metric_status == "measured"
        ]
        if name in _BOOLEAN_METRICS:
            equal_pairs = sum(point.boolean_equal is True for point in complete)
            median_delta = min_delta = max_delta = None
        else:
            deltas = [point.delta_candidate_minus_baseline for point in complete]
            if any(delta is None for delta in deltas):
                raise BikvTargetPairedSummaryError(
                    f"metric {name!r} complete pair is missing numeric delta"
                )
            numeric_deltas = [delta for delta in deltas if delta is not None]
            median_delta = statistics.median(numeric_deltas) if numeric_deltas else None
            min_delta = min(numeric_deltas) if numeric_deltas else None
            max_delta = max(numeric_deltas) if numeric_deltas else None
            equal_pairs = None

        summaries.append(
            PairedMetricSummaryV1(
                name=name,
                total_pairs=len(points),
                measured_pairs=len(complete),
                incomplete_pairs=len(points) - len(complete),
                unit=unit,
                median_delta_candidate_minus_baseline=median_delta,
                min_delta_candidate_minus_baseline=min_delta,
                max_delta_candidate_minus_baseline=max_delta,
                boolean_equal_pairs=equal_pairs,
                points=tuple(points),
            )
        )

    return BikvTargetPairedSummaryV1(
        schema=BKV_TARGET_PAIRED_SUMMARY_SCHEMA_V1,
        protocol_sha256=protocol.protocol_sha256(),
        campaign_sha256=campaign.campaign_sha256(),
        campaign_id=protocol.campaign_id,
        interpretation=(
            "descriptive paired evidence only; no inferential interval, winner, "
            "BKV-K9/NBKV promotion, hardware claim, or protected-holdout authorization"
        ),
        metrics=tuple(summaries),
    )
