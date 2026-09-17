"""Typed BIKV byte-accounting evidence.

BKV studies need to distinguish logical packed-payload accounting from observed
host/device transfers and physical DRAM counters.  This module deliberately
refuses to form the headline avoided/read ratio from mixed evidence kinds.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


LOGICAL_PACKED_PAYLOAD = "logical_packed_payload"
HOST_OBSERVED_TRANSFER = "host_observed_transfer"
DEVICE_OBSERVED_TRANSFER = "device_observed_transfer"
PHYSICAL_DRAM_COUNTER = "physical_dram_counter"

_TRAFFIC_EVIDENCE_KINDS = frozenset(
    {
        LOGICAL_PACKED_PAYLOAD,
        HOST_OBSERVED_TRANSFER,
        DEVICE_OBSERVED_TRANSFER,
        PHYSICAL_DRAM_COUNTER,
    }
)


class BikvTrafficEvidenceError(ValueError):
    """Raised when BIKV byte evidence is malformed or semantically mixed."""


def _require_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BikvTrafficEvidenceError(f"{name} must be a non-negative integer")
    return value


def _require_kind(name: str, value: object) -> str:
    if not isinstance(value, str) or value not in _TRAFFIC_EVIDENCE_KINDS:
        allowed = ", ".join(sorted(_TRAFFIC_EVIDENCE_KINDS))
        raise BikvTrafficEvidenceError(f"{name} must be one of: {allowed}")
    return value


@dataclass(frozen=True, slots=True)
class BikvTrafficEvidence:
    """Evidence-kind-bound byte counters for one BIKV comparison.

    `logical_packed_payload` means exact logical/packed payload accounting.  It
    is not a measurement of allocator residency, cache-line traffic, PCIe/NVLink
    traffic, memory-controller traffic, or DRAM bandwidth.  The three observed
    kinds must come from the corresponding measurement surface in the retained
    experiment evidence.

    A ratio is defined only when numerator and denominator use the same evidence
    kind.  This prevents a logical avoided-byte estimate from being divided by a
    physical counter (or vice versa) and silently presented as one traffic ratio.
    """

    numerical_kv_bytes_avoided: int
    numerical_evidence_kind: str
    boolean_kv_bytes_read: int
    boolean_evidence_kind: str

    def validate(self) -> None:
        _require_nonnegative_int(
            "numerical_kv_bytes_avoided", self.numerical_kv_bytes_avoided
        )
        _require_nonnegative_int("boolean_kv_bytes_read", self.boolean_kv_bytes_read)
        numerical_kind = _require_kind("numerical_evidence_kind", self.numerical_evidence_kind)
        boolean_kind = _require_kind("boolean_evidence_kind", self.boolean_evidence_kind)
        if self.numerical_kv_bytes_avoided > 0 and self.boolean_kv_bytes_read == 0:
            raise BikvTrafficEvidenceError(
                "non-zero numerical KV bytes avoided requires non-zero Boolean KV bytes read"
            )
        if numerical_kind != boolean_kind:
            raise BikvTrafficEvidenceError(
                "numerical and Boolean byte counters must use the same evidence kind"
            )

    @property
    def evidence_kind(self) -> str:
        """Return the common validated evidence kind."""

        self.validate()
        return self.numerical_evidence_kind

    @property
    def numerical_bytes_avoided_per_boolean_byte(self) -> Fraction | None:
        """Return an exact like-for-like ratio, or ``None`` at zero denominator."""

        self.validate()
        if self.boolean_kv_bytes_read == 0:
            return None
        return Fraction(self.numerical_kv_bytes_avoided, self.boolean_kv_bytes_read)
