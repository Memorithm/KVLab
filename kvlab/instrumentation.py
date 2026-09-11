"""KV-cache resource accounting with measured/estimated/not-exposed provenance.

The accounting layer deliberately separates logical cache size from residency,
transfers, fragmentation, and recomputation. Callers must state how each value
was obtained; absent hardware telemetry remains explicit rather than fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class MeasurementKind(str, Enum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    NOT_EXPOSED = "not_exposed"


class ResourceAccountingError(ValueError):
    pass


@dataclass(frozen=True)
class Quantity:
    value: float | None
    unit: str
    kind: MeasurementKind

    def validate(self) -> None:
        if not self.unit:
            raise ResourceAccountingError("unit is required")
        if self.kind is MeasurementKind.NOT_EXPOSED:
            if self.value is not None:
                raise ResourceAccountingError("not_exposed quantities must not carry a value")
            return
        if self.value is None or not math.isfinite(self.value) or self.value < 0.0:
            raise ResourceAccountingError("measured/estimated quantities require a finite non-negative value")


@dataclass(frozen=True)
class KVResourceAccounting:
    logical_cache_bytes: Quantity
    gpu_resident_bytes: Quantity
    host_resident_bytes: Quantity
    secondary_storage_bytes: Quantity
    fragmentation_bytes: Quantity
    host_to_gpu_bytes: Quantity
    gpu_to_host_bytes: Quantity
    bytes_read: Quantity
    bytes_written: Quantity
    recomputed_token_count: Quantity

    def validate(self) -> None:
        quantities = (
            self.logical_cache_bytes,
            self.gpu_resident_bytes,
            self.host_resident_bytes,
            self.secondary_storage_bytes,
            self.fragmentation_bytes,
            self.host_to_gpu_bytes,
            self.gpu_to_host_bytes,
            self.bytes_read,
            self.bytes_written,
            self.recomputed_token_count,
        )
        for quantity in quantities:
            quantity.validate()
        for quantity in quantities[:-1]:
            if quantity.unit != "bytes":
                raise ResourceAccountingError("memory/traffic quantities must use bytes")
        if self.recomputed_token_count.unit != "tokens":
            raise ResourceAccountingError("recomputed_token_count must use tokens")

    @property
    def measured_gpu_residency(self) -> float | None:
        if self.gpu_resident_bytes.kind is MeasurementKind.MEASURED:
            return self.gpu_resident_bytes.value
        return None

    @property
    def estimated_gpu_residency(self) -> float | None:
        if self.gpu_resident_bytes.kind is MeasurementKind.ESTIMATED:
            return self.gpu_resident_bytes.value
        return None


def bytes_quantity(value: int | float, kind: MeasurementKind) -> Quantity:
    quantity = Quantity(float(value), "bytes", kind)
    quantity.validate()
    return quantity


def tokens_quantity(value: int | float, kind: MeasurementKind) -> Quantity:
    quantity = Quantity(float(value), "tokens", kind)
    quantity.validate()
    return quantity


def not_exposed(unit: str) -> Quantity:
    quantity = Quantity(None, unit, MeasurementKind.NOT_EXPOSED)
    quantity.validate()
    return quantity
