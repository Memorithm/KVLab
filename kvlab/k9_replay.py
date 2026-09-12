"""Fail-closed replay validation for K9 cross-model KV captures.

This module validates that replayed tensor payload bytes correspond exactly to an
immutable :class:`CaptureManifest`. It does not interpret tensor values, fit a
mapper, execute a model, or make quality/performance claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

from kvlab.k9_capture import CaptureManifest, TensorCapture


@dataclass(frozen=True, slots=True, order=True)
class CaptureIdentity:
    sample_id: str
    layer: int
    kind: str

    @classmethod
    def from_capture(cls, capture: TensorCapture) -> "CaptureIdentity":
        return cls(capture.sample_id, capture.layer, capture.kind)


@dataclass(frozen=True, slots=True)
class ReplayPayload:
    identity: CaptureIdentity
    chunks: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if not self.chunks:
            raise ValueError("replay payload must contain at least one byte chunk")
        if any(not isinstance(chunk, bytes) for chunk in self.chunks):
            raise TypeError("replay payload chunks must be bytes")

    @property
    def content_sha256(self) -> str:
        digest = sha256()
        for chunk in self.chunks:
            digest.update(chunk)
        return digest.hexdigest()

    @property
    def byte_length(self) -> int:
        return sum(len(chunk) for chunk in self.chunks)


def validate_replay_payloads(
    manifest: CaptureManifest,
    payloads: Iterable[ReplayPayload],
) -> dict[CaptureIdentity, int]:
    """Validate exact manifest coverage and content digests.

    Returns a mapping from capture identity to observed payload byte length only
    after every manifest record has exactly one matching payload and every digest
    matches. Extra, missing, duplicate, or corrupted payloads fail closed.
    """

    expected = {
        CaptureIdentity.from_capture(capture): capture
        for capture in manifest.captures
    }
    observed: dict[CaptureIdentity, ReplayPayload] = {}
    for payload in payloads:
        if payload.identity in observed:
            raise ValueError(f"duplicate replay payload: {_format_identity(payload.identity)}")
        observed[payload.identity] = payload

    missing = sorted(set(expected).difference(observed))
    if missing:
        joined = ", ".join(_format_identity(identity) for identity in missing)
        raise ValueError(f"missing replay payloads: {joined}")

    extra = sorted(set(observed).difference(expected))
    if extra:
        joined = ", ".join(_format_identity(identity) for identity in extra)
        raise ValueError(f"unexpected replay payloads: {joined}")

    byte_lengths: dict[CaptureIdentity, int] = {}
    for identity, capture in expected.items():
        payload = observed[identity]
        if payload.content_sha256 != capture.content_sha256.lower():
            raise ValueError(f"replay digest mismatch: {_format_identity(identity)}")
        byte_lengths[identity] = payload.byte_length

    return byte_lengths


def _format_identity(identity: CaptureIdentity) -> str:
    return f"{identity.sample_id}/layer-{identity.layer}/{identity.kind}"
