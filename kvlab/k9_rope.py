"""Fail-closed standard-RoPE key normalization for K9 cross-model transfer.

The K9 ridge baseline is fitted on keys with positional rotation removed.  This
module implements only the two explicitly named coordinate layouts used by
common decoder implementations.  It does not infer a model's layout, scaling
rule, base, rotary dimension, or token position from tensor contents.

All inputs are promoted to Python ``float`` (binary64) for this reference
implementation.  A production/backend adapter must independently qualify its
native-dtype implementation against this oracle before its outputs are used as
K9 calibration evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from math import cos, isfinite, sin
import struct
from typing import Iterable


SCHEMA_ID = "kvlab.k9-rope-normalization.v1"


class RopeLayout(str, Enum):
    """Declared coordinate pairing for the rotary subspace."""

    INTERLEAVED_PAIRS = "interleaved-pairs-v1"
    HALF_SPLIT = "half-split-v1"


@dataclass(frozen=True, slots=True)
class RopeNormalizationSpec:
    """Frozen standard-RoPE parameters used to remove key rotation."""

    theta_base: float
    rotary_dim: int
    layout: RopeLayout

    def __post_init__(self) -> None:
        if type(self.theta_base) not in (int, float):
            raise ValueError("theta_base must be a finite number greater than 1")
        theta = float(self.theta_base)
        if not isfinite(theta) or theta <= 1.0:
            raise ValueError("theta_base must be a finite number greater than 1")
        if type(self.rotary_dim) is not int or self.rotary_dim <= 0 or self.rotary_dim % 2:
            raise ValueError("rotary_dim must be a positive even integer")
        if type(self.layout) is not RopeLayout:
            raise ValueError("layout must be an explicit RopeLayout")

    def fingerprint(self) -> str:
        """Return a stable identity for the declared normalization semantics."""

        payload = (
            f"{SCHEMA_ID}\n"
            f"theta_base={float(self.theta_base).hex()}\n"
            f"rotary_dim={self.rotary_dim}\n"
            f"layout={self.layout.value}\n"
        )
        return sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class RopeNormalizationRecord:
    """Content-addressed reference evidence for one normalized key vector."""

    sample_id: str
    layer: int
    head: int
    position: int
    input_f64_sha256: str
    output_f64_sha256: str
    spec_fingerprint: str

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        for name, value in (("layer", self.layer), ("head", self.head), ("position", self.position)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name, value in (
            ("input_f64_sha256", self.input_f64_sha256),
            ("output_f64_sha256", self.output_f64_sha256),
            ("spec_fingerprint", self.spec_fingerprint),
        ):
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def remove_rope(
    key: Iterable[float],
    *,
    position: int,
    spec: RopeNormalizationSpec,
) -> tuple[float, ...]:
    """Remove declared standard RoPE from one key vector.

    The dimensions after ``spec.rotary_dim`` are copied unchanged.  No model or
    tokenizer metadata is inferred.  Non-finite numeric inputs fail closed.
    """

    return _rotate(key, position=position, spec=spec, direction=-1.0)


def apply_rope_reference(
    key: Iterable[float],
    *,
    position: int,
    spec: RopeNormalizationSpec,
) -> tuple[float, ...]:
    """Reference forward rotation used only for oracle/round-trip qualification."""

    return _rotate(key, position=position, spec=spec, direction=1.0)


def normalization_record(
    key: Iterable[float],
    *,
    sample_id: str,
    layer: int,
    head: int,
    position: int,
    spec: RopeNormalizationSpec,
) -> tuple[tuple[float, ...], RopeNormalizationRecord]:
    """Normalize one key and bind the binary64 reference input/output identities."""

    values = _validated_vector(key, spec=spec)
    normalized = remove_rope(values, position=position, spec=spec)
    record = RopeNormalizationRecord(
        sample_id=sample_id,
        layer=layer,
        head=head,
        position=position,
        input_f64_sha256=_f64_digest(values),
        output_f64_sha256=_f64_digest(normalized),
        spec_fingerprint=spec.fingerprint(),
    )
    return normalized, record


def _rotate(
    key: Iterable[float],
    *,
    position: int,
    spec: RopeNormalizationSpec,
    direction: float,
) -> tuple[float, ...]:
    if type(position) is not int or position < 0:
        raise ValueError("position must be a non-negative integer")
    values = list(_validated_vector(key, spec=spec))
    half = spec.rotary_dim // 2

    for pair_index in range(half):
        angle = direction * position * float(spec.theta_base) ** (-2.0 * pair_index / spec.rotary_dim)
        c = cos(angle)
        s = sin(angle)
        if spec.layout is RopeLayout.INTERLEAVED_PAIRS:
            left = 2 * pair_index
            right = left + 1
        else:
            left = pair_index
            right = pair_index + half
        x = values[left]
        y = values[right]
        values[left] = x * c - y * s
        values[right] = x * s + y * c

    if any(not isfinite(value) for value in values):
        raise ValueError("RoPE normalization produced a non-finite value")
    return tuple(values)


def _validated_vector(
    key: Iterable[float],
    *,
    spec: RopeNormalizationSpec,
) -> tuple[float, ...]:
    try:
        raw = tuple(key)
    except TypeError as exc:
        raise ValueError("key must be an iterable of finite numbers") from exc
    if len(raw) < spec.rotary_dim:
        raise ValueError("key width must be at least rotary_dim")
    values: list[float] = []
    for item in raw:
        if type(item) not in (int, float):
            raise ValueError("key values must be finite numbers")
        value = float(item)
        if not isfinite(value):
            raise ValueError("key values must be finite numbers")
        values.append(value)
    return tuple(values)


def _f64_digest(values: Iterable[float]) -> str:
    digest = sha256()
    for value in values:
        digest.update(struct.pack(">d", float(value)))
    return digest.hexdigest()
