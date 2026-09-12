"""Deterministic K9 protocol gates for cross-model KV-cache transfer.

This module encodes protocol prerequisites only. It does not perform a transfer,
fit a mapper, or claim latency/quality improvements. The initial protocol follows
the matched-KV setting described by Heo et al. (2026): source and target must
share KV-head count and per-head dimension; key mappings must declare whether
RoPE is removed before fitting; ridge regression is the mandatory linear
baseline; calibration and final holdout identities must remain disjoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class TransferApplicability(str, Enum):
    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not-applicable"


class MapperBaseline(str, Enum):
    RIDGE = "ridge"
    LINEAR = "linear"


@dataclass(frozen=True, slots=True)
class KVGeometry:
    kv_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        if self.kv_heads <= 0:
            raise ValueError("kv_heads must be positive")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")


@dataclass(frozen=True, slots=True)
class ModelRevision:
    model_id: str
    revision: str
    tokenizer_revision: str
    kv: KVGeometry

    def __post_init__(self) -> None:
        for name, value in (
            ("model_id", self.model_id),
            ("revision", self.revision),
            ("tokenizer_revision", self.tokenizer_revision),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")


@dataclass(frozen=True, slots=True)
class CrossModelTransferProtocol:
    source: ModelRevision
    target: ModelRevision
    mapper: MapperBaseline
    ridge_alpha: float
    source_layers_per_target: int
    remove_rope_from_keys: bool
    calibration_ids: tuple[str, ...]
    final_holdout_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.ridge_alpha < 0:
            raise ValueError("ridge_alpha must be non-negative")
        if self.source_layers_per_target <= 0:
            raise ValueError("source_layers_per_target must be positive")
        _require_unique_non_empty("calibration_ids", self.calibration_ids)
        _require_unique_non_empty("final_holdout_ids", self.final_holdout_ids)
        overlap = set(self.calibration_ids).intersection(self.final_holdout_ids)
        if overlap:
            raise ValueError("calibration_ids and final_holdout_ids must be disjoint")


@dataclass(frozen=True, slots=True)
class TransferDecision:
    applicability: TransferApplicability
    reasons: tuple[str, ...]

    @property
    def is_applicable(self) -> bool:
        return self.applicability is TransferApplicability.APPLICABLE


def assess_cross_model_transfer(protocol: CrossModelTransferProtocol) -> TransferDecision:
    """Validate prerequisites before any K9 calibration or holdout execution."""

    reasons: list[str] = []
    if protocol.source.kv != protocol.target.kv:
        reasons.append(
            "source and target do not have matched KV-head count and per-head dimension"
        )
    if protocol.mapper is MapperBaseline.RIDGE and not protocol.remove_rope_from_keys:
        reasons.append("ridge key mapping requires declared RoPE removal before fitting")
    if protocol.source == protocol.target:
        reasons.append("source and target revisions are identical; this is not cross-model transfer")

    if reasons:
        return TransferDecision(TransferApplicability.NOT_APPLICABLE, tuple(reasons))
    return TransferDecision(
        TransferApplicability.APPLICABLE,
        ("matched KV geometry and preregistered calibration/holdout separation",),
    )


def _require_unique_non_empty(name: str, values: Iterable[str]) -> None:
    values = tuple(values)
    if not values:
        raise ValueError(f"{name} must be non-empty")
    if any(not value.strip() for value in values):
        raise ValueError(f"{name} entries must be non-empty")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")
