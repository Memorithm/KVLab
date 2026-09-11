"""Latency accounting for KV-cache experiments.

Prefill, TTFT, decode/TPOT, and transformation costs remain separate. Values
carry measured/estimated/not-exposed provenance so unavailable GPU/backend
telemetry is never silently invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class LatencyKind(str, Enum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    NOT_EXPOSED = "not_exposed"


class LatencyError(ValueError):
    pass


@dataclass(frozen=True)
class LatencyQuantity:
    milliseconds: float | None
    kind: LatencyKind

    def validate(self) -> None:
        if self.kind is LatencyKind.NOT_EXPOSED:
            if self.milliseconds is not None:
                raise LatencyError("not_exposed latency must not carry a value")
            return
        if (
            self.milliseconds is None
            or not math.isfinite(self.milliseconds)
            or self.milliseconds < 0.0
        ):
            raise LatencyError("measured/estimated latency must be finite and non-negative")


@dataclass(frozen=True)
class KVLatencyAccounting:
    prefill: LatencyQuantity
    ttft: LatencyQuantity
    decode_total: LatencyQuantity
    tpot: LatencyQuantity
    cache_transform: LatencyQuantity
    cache_reconstruction: LatencyQuantity

    def validate(self) -> None:
        for quantity in (
            self.prefill,
            self.ttft,
            self.decode_total,
            self.tpot,
            self.cache_transform,
            self.cache_reconstruction,
        ):
            quantity.validate()

    def measured_values(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, quantity in (
            ("prefill_ms", self.prefill),
            ("ttft_ms", self.ttft),
            ("decode_total_ms", self.decode_total),
            ("tpot_ms", self.tpot),
            ("cache_transform_ms", self.cache_transform),
            ("cache_reconstruction_ms", self.cache_reconstruction),
        ):
            if quantity.kind is LatencyKind.MEASURED:
                assert quantity.milliseconds is not None
                result[name] = quantity.milliseconds
        return result


def measured(milliseconds: float) -> LatencyQuantity:
    quantity = LatencyQuantity(milliseconds, LatencyKind.MEASURED)
    quantity.validate()
    return quantity


def estimated(milliseconds: float) -> LatencyQuantity:
    quantity = LatencyQuantity(milliseconds, LatencyKind.ESTIMATED)
    quantity.validate()
    return quantity


def not_exposed() -> LatencyQuantity:
    return LatencyQuantity(None, LatencyKind.NOT_EXPOSED)
