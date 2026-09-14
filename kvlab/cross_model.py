"""Deterministic K9 protocol gates for cross-model KV-cache transfer.

This module encodes protocol prerequisites only. It does not perform a transfer,
fit a mapper, or claim latency/quality improvements. The initial protocol follows
the matched-KV setting described by Heo et al. (2026): source and target must
share KV-head count and per-head dimension; key mappings must declare whether
RoPE is removed before fitting; ridge regression is the mandatory linear
baseline; calibration and final holdout identities must remain disjoint.

Protocol inputs are validated at runtime and split identities are captured as
immutable tuples. Valid identities, their order and numeric values are preserved;
malformed types are rejected rather than silently coerced.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


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
        _require_positive_int("kv_heads", self.kv_heads)
        _require_positive_int("head_dim", self.head_dim)


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
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.kv, KVGeometry):
            raise ValueError("kv must be a validated KVGeometry")


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
        for name, value in (("source", self.source), ("target", self.target)):
            if not isinstance(value, ModelRevision):
                raise ValueError(f"{name} must be a validated ModelRevision")
        if not isinstance(self.mapper, MapperBaseline):
            raise ValueError("mapper must be a MapperBaseline member")
        if type(self.remove_rope_from_keys) is not bool:
            raise ValueError("remove_rope_from_keys must be a bool")
        if type(self.ridge_alpha) not in (int, float):
            raise ValueError("ridge_alpha must be a finite non-negative number")
        try:
            valid_alpha = isfinite(self.ridge_alpha) and self.ridge_alpha >= 0
        except OverflowError:
            valid_alpha = False
        if not valid_alpha:
            raise ValueError("ridge_alpha must be a finite non-negative number")
        _require_positive_int("source_layers_per_target", self.source_layers_per_target)

        # Freeze exactly the sequences that are checked. A frozen dataclass alone
        # does not protect lists supplied by the caller from subsequent mutation.
        calibration = _freeze_ids("calibration_ids", self.calibration_ids)
        holdout = _freeze_ids("final_holdout_ids", self.final_holdout_ids)
        if set(calibration).intersection(holdout):
            raise ValueError("calibration_ids and final_holdout_ids must be disjoint")
        object.__setattr__(self, "calibration_ids", calibration)
        object.__setattr__(self, "final_holdout_ids", holdout)


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


def _require_positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _freeze_ids(name: str, values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple of sample IDs")
    snapshot = tuple(values)
    if not snapshot:
        raise ValueError(f"{name} must be non-empty")
    if any(not isinstance(value, str) or not value.strip() for value in snapshot):
        raise ValueError(f"{name} entries must be non-empty strings")
    if len(snapshot) != len(set(snapshot)):
        raise ValueError(f"{name} must not contain duplicates")
    return snapshot
