"""Explicit K3 KV representation and quantization accounting contracts.

Theoretical payload size is deliberately separated from measured residency,
metadata, alignment and backend support. Naming a representation does not mean
that a runtime can execute it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .instrumentation import KVResourceAccounting, MeasurementKind


class KvRepresentation(str, Enum):
    BF16 = "bf16"
    FP16 = "fp16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"

    @property
    def payload_bits_per_value(self) -> int:
        return {
            KvRepresentation.BF16: 16,
            KvRepresentation.FP16: 16,
            KvRepresentation.FP8: 8,
            KvRepresentation.INT8: 8,
            KvRepresentation.INT4: 4,
        }[self]


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    source: KvRepresentation
    target: KvRepresentation
    values: int
    group_size: int | None = None
    metadata_bits: int = 0

    def __post_init__(self) -> None:
        if self.values <= 0:
            raise ValueError("values must be positive")
        if self.group_size is not None and self.group_size <= 0:
            raise ValueError("group_size must be positive when specified")
        if self.metadata_bits < 0:
            raise ValueError("metadata_bits must be non-negative")
        if self.source is self.target:
            raise ValueError("source and target representations must differ")

    @property
    def theoretical_source_payload_bits(self) -> int:
        return self.values * self.source.payload_bits_per_value

    @property
    def theoretical_target_payload_bits(self) -> int:
        return self.values * self.target.payload_bits_per_value

    @property
    def theoretical_target_total_bits(self) -> int:
        """Payload plus explicitly supplied metadata, excluding unknown padding."""

        return self.theoretical_target_payload_bits + self.metadata_bits

    @property
    def theoretical_payload_reduction_bits(self) -> int:
        """Ideal payload-only delta; this is not measured GPU/RAM residency."""

        return self.theoretical_source_payload_bits - self.theoretical_target_payload_bits


@dataclass(frozen=True, slots=True)
class BackendQuantizationCapability:
    backend: str
    supported_targets: frozenset[KvRepresentation]

    def __post_init__(self) -> None:
        if not self.backend.strip():
            raise ValueError("backend must be non-empty")

    def require_supported(self, target: KvRepresentation) -> None:
        if target not in self.supported_targets:
            raise ValueError(f"backend {self.backend!r} does not support {target.value}")


@dataclass(frozen=True, slots=True)
class QuantizationAccountingComparison:
    """Bind theoretical K3 accounting to separately observed source/target resources.

    This object never infers physical savings from representation bit width. GPU
    residency deltas are exposed only when both arms were actually measured;
    estimated or unavailable telemetry remains explicit in the input records.
    """

    config: QuantizationConfig
    source: KVResourceAccounting
    target: KVResourceAccounting

    def __post_init__(self) -> None:
        self.source.validate()
        self.target.validate()

    @property
    def theoretical_target_total_bytes_ceiling(self) -> int:
        return (self.config.theoretical_target_total_bits + 7) // 8

    @property
    def logical_cache_delta_bytes(self) -> float | None:
        source = self.source.logical_cache_bytes
        target = self.target.logical_cache_bytes
        if source.value is None or target.value is None:
            return None
        return source.value - target.value

    @property
    def measured_gpu_residency_delta_bytes(self) -> float | None:
        source = self.source.gpu_resident_bytes
        target = self.target.gpu_resident_bytes
        if (
            source.kind is not MeasurementKind.MEASURED
            or target.kind is not MeasurementKind.MEASURED
        ):
            return None
        assert source.value is not None and target.value is not None
        return source.value - target.value


class ReconstructionMetric(str, Enum):
    RMSE = "rmse"
    MAX_ABS_ERROR = "max_abs_error"
    RELATIVE_L2_ERROR = "relative_l2_error"


@dataclass(frozen=True, slots=True)
class QuantizationQualityObservation:
    """One reconstruction-quality observation with explicit provenance.

    This record stores an observed or estimated error. It deliberately contains
    no pass/fail threshold and therefore cannot itself satisfy a preregistered
    scientific decision rule or justify a claim about downstream model quality.
    """

    metric: ReconstructionMetric
    value: float | None
    kind: MeasurementKind

    def __post_init__(self) -> None:
        if self.kind is MeasurementKind.NOT_EXPOSED:
            if self.value is not None:
                raise ValueError("not_exposed quality observations must not carry a value")
            return
        if self.value is None or not math.isfinite(self.value) or self.value < 0.0:
            raise ValueError(
                "measured/estimated quality observations require a finite non-negative value"
            )

    @property
    def measured_value(self) -> float | None:
        if self.kind is MeasurementKind.MEASURED:
            return self.value
        return None
