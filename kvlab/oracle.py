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
        if type(self.layer) is not int or self.layer < 0:
            raise OracleError("layer must be a non-negative integer")
        if not isinstance(self.dtype, str) or not self.dtype or any(ch.isspace() for ch in self.dtype):
            raise OracleError("dtype must be a non-empty canonical token")
        for name, shape in (("key_shape", self.key_shape), ("value_shape", self.value_shape)):
            if (
                not isinstance(shape, (list, tuple))
                or not shape
                or any(type(dim) is not int or dim <= 0 for dim in shape)
            ):
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
        """Validate v1 metadata and yield only the immutable records actually hashed.

        Validation occurs at the first iteration, before any record is yielded.
        Snapshotting again supports directly constructed dataclasses without
        trusting mutable record lists or shape lists retained by their callers.
        This is byte replay, not device restoration or tensor-semantic validation.
        """
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise OracleError("unsupported full-cache snapshot schema version")
        _validate_header(self.model_revision, self.tokenizer_revision, self.sequence_length)
        frozen = _freeze_records(self.records)
        expected = _digest(
            self.model_revision,
            self.tokenizer_revision,
            self.sequence_length,
            frozen,
        )
        if expected != self.digest_sha256:
            raise OracleError("snapshot digest mismatch")
        yield from frozen


def capture_full_cache(
    *,
    model_revision: str,
    tokenizer_revision: str,
    sequence_length: int,
    records: Iterable[KVLayerRecord],
) -> FullCacheSnapshot:
    """Freeze byte payloads and independent immutable metadata for comparison.

    Lists supplied as shapes are copied to tuples before validation and hashing.
    No dtype is inferred and no byte-count/shape equivalence is assumed.
    """
    _validate_header(model_revision, tokenizer_revision, sequence_length)
    frozen = _freeze_records(records)

    return FullCacheSnapshot(
        schema_version=1,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        sequence_length=sequence_length,
        records=frozen,
        digest_sha256=_digest(model_revision, tokenizer_revision, sequence_length, frozen),
    )


def _validate_header(model_revision: str, tokenizer_revision: str, sequence_length: int) -> None:
    """Check v1 header types without normalizing caller identity strings."""
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (model_revision, tokenizer_revision)
    ):
        raise OracleError("model and tokenizer revisions must be non-empty strings")
    if type(sequence_length) is not int or sequence_length <= 0:
        raise OracleError("sequence_length must be a positive integer")


def _freeze_records(records: Iterable[KVLayerRecord]) -> tuple[KVLayerRecord, ...]:
    """Freeze nested shape metadata, then validate exactly the retained records."""
    try:
        supplied = tuple(records)
    except TypeError as error:
        raise OracleError("records must be an iterable of KVLayerRecord values") from error
    if not supplied:
        raise OracleError("at least one K/V layer record is required")
    frozen = []
    for record in supplied:
        if not isinstance(record, KVLayerRecord):
            raise OracleError("records must contain only KVLayerRecord values")
        if not isinstance(record.key_shape, (list, tuple)) or not isinstance(
            record.value_shape, (list, tuple)
        ):
            raise OracleError("key_shape and value_shape must be lists or tuples")
        captured = KVLayerRecord(
            record.layer,
            record.dtype,
            tuple(record.key_shape),
            tuple(record.value_shape),
            record.key_bytes,
            record.value_bytes,
        )
        captured.validate()
        frozen.append(captured)
    layers = [record.layer for record in frozen]
    if layers != sorted(layers) or len(set(layers)) != len(layers):
        raise OracleError("layer records must be unique and strictly ordered")
    return tuple(frozen)


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
