"""Deterministic, backend-neutral full-cache oracle capture/replay primitives.

This module does not infer tensor semantics. Callers provide already-materialized K/V
payload bytes plus explicit dtype/shape metadata. The oracle records exactly those
bytes and a canonical digest so later candidates can be compared against an
unchanged native/full-cache baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Iterator


class OracleError(ValueError):
    """Raised when an oracle snapshot would be ambiguous or malformed."""


@dataclass(frozen=True)
class KVLayerRecord:
    layer: int
    dtype: str
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    key_bytes: bytes
    value_bytes: bytes

    def validate(self) -> None:
        if self.layer < 0:
            raise OracleError("layer must be non-negative")
        if not self.dtype or any(ch.isspace() for ch in self.dtype):
            raise OracleError("dtype must be a non-empty canonical token")
        for name, shape in (("key_shape", self.key_shape), ("value_shape", self.value_shape)):
            if not shape or any((not isinstance(dim, int)) or dim <= 0 for dim in shape):
                raise OracleError(f"{name} must contain positive integer dimensions")
        if not isinstance(self.key_bytes, bytes) or not isinstance(self.value_bytes, bytes):
            raise OracleError("K/V payloads must be immutable bytes")

    @property
    def logical_bytes(self) -> int:
        return len(self.key_bytes) + len(self.value_bytes)


@dataclass(frozen=True)
class FullCacheSnapshot:
    schema_version: int
    model_revision: str
    tokenizer_revision: str
    sequence_length: int
    records: tuple[KVLayerRecord, ...]
    digest_sha256: str

    @property
    def logical_bytes(self) -> int:
        return sum(record.logical_bytes for record in self.records)

    def replay(self) -> Iterator[KVLayerRecord]:
        """Yield the exact recorded K/V payloads after verifying the snapshot digest."""
        expected = _digest(
            self.model_revision,
            self.tokenizer_revision,
            self.sequence_length,
            self.records,
        )
        if expected != self.digest_sha256:
            raise OracleError("snapshot digest mismatch")
        yield from self.records


def capture_full_cache(
    *,
    model_revision: str,
    tokenizer_revision: str,
    sequence_length: int,
    records: Iterable[KVLayerRecord],
) -> FullCacheSnapshot:
    """Freeze an unchanged native/full-cache reference for later comparison."""
    if not model_revision or not tokenizer_revision:
        raise OracleError("model and tokenizer revisions are required")
    if sequence_length <= 0:
        raise OracleError("sequence_length must be positive")

    frozen = tuple(records)
    if not frozen:
        raise OracleError("at least one K/V layer record is required")
    for record in frozen:
        record.validate()

    layers = [record.layer for record in frozen]
    if layers != sorted(layers) or len(set(layers)) != len(layers):
        raise OracleError("layer records must be unique and strictly ordered")

    return FullCacheSnapshot(
        schema_version=1,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        sequence_length=sequence_length,
        records=frozen,
        digest_sha256=_digest(model_revision, tokenizer_revision, sequence_length, frozen),
    )


def _digest(
    model_revision: str,
    tokenizer_revision: str,
    sequence_length: int,
    records: tuple[KVLayerRecord, ...],
) -> str:
    hasher = hashlib.sha256()
    header = {
        "schema_version": 1,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "sequence_length": sequence_length,
    }
    hasher.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for record in records:
        metadata = {
            "layer": record.layer,
            "dtype": record.dtype,
            "key_shape": record.key_shape,
            "value_shape": record.value_shape,
            "key_len": len(record.key_bytes),
            "value_len": len(record.value_bytes),
        }
        hasher.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        hasher.update(record.key_bytes)
        hasher.update(record.value_bytes)
    return hasher.hexdigest()
