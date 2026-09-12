"""Deterministic capture metadata for K9 cross-model KV experiments.

The capture manifest records immutable provenance and tensor geometry before any
mapper fitting or final-holdout execution. It intentionally stores metadata and
content digests only; it does not claim transfer quality, latency, or memory
improvements.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable


@dataclass(frozen=True, slots=True)
class TensorCapture:
    sample_id: str
    layer: int
    kind: str
    shape: tuple[int, ...]
    dtype: str
    content_sha256: str
    rope_removed: bool

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        if self.kind not in {"key", "value"}:
            raise ValueError("kind must be 'key' or 'value'")
        if not self.shape or any(dim <= 0 for dim in self.shape):
            raise ValueError("shape dimensions must be positive")
        if not self.dtype.strip():
            raise ValueError("dtype must be non-empty")
        digest = self.content_sha256.lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("content_sha256 must be a 64-character hexadecimal digest")
        if self.kind == "value" and self.rope_removed:
            raise ValueError("rope_removed is only meaningful for key captures")


@dataclass(frozen=True, slots=True)
class CaptureManifest:
    experiment_id: str
    repository_commit: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    runtime: str
    split: str
    captures: tuple[TensorCapture, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("experiment_id", self.experiment_id),
            ("repository_commit", self.repository_commit),
            ("model_id", self.model_id),
            ("model_revision", self.model_revision),
            ("tokenizer_revision", self.tokenizer_revision),
            ("runtime", self.runtime),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.split not in {"calibration", "final-holdout"}:
            raise ValueError("split must be calibration or final-holdout")
        if not self.captures:
            raise ValueError("captures must be non-empty")
        identities = [(c.sample_id, c.layer, c.kind) for c in self.captures]
        if len(identities) != len(set(identities)):
            raise ValueError("captures must not duplicate sample/layer/kind identities")

    @property
    def sample_ids(self) -> frozenset[str]:
        return frozenset(capture.sample_id for capture in self.captures)

    def fingerprint(self) -> str:
        """Return a deterministic metadata fingerprint for replay/provenance checks."""

        rows = [
            self.experiment_id,
            self.repository_commit,
            self.model_id,
            self.model_revision,
            self.tokenizer_revision,
            self.runtime,
            self.split,
        ]
        for capture in sorted(
            self.captures,
            key=lambda item: (item.sample_id, item.layer, item.kind),
        ):
            rows.append(
                "|".join(
                    (
                        capture.sample_id,
                        str(capture.layer),
                        capture.kind,
                        ",".join(str(dim) for dim in capture.shape),
                        capture.dtype,
                        capture.content_sha256.lower(),
                        "rope-removed" if capture.rope_removed else "rope-present",
                    )
                )
            )
        return sha256("\n".join(rows).encode("utf-8")).hexdigest()


def assert_disjoint_capture_splits(
    calibration: CaptureManifest,
    final_holdout: CaptureManifest,
) -> None:
    """Fail closed if calibration and final holdout share any sample identity."""

    if calibration.split != "calibration":
        raise ValueError("calibration manifest must declare split='calibration'")
    if final_holdout.split != "final-holdout":
        raise ValueError("final holdout manifest must declare split='final-holdout'")
    overlap = calibration.sample_ids.intersection(final_holdout.sample_ids)
    if overlap:
        joined = ", ".join(sorted(overlap))
        raise ValueError(f"calibration/final-holdout sample overlap: {joined}")


def digest_bytes(chunks: Iterable[bytes]) -> str:
    """Hash captured bytes without interpreting them or changing tensor semantics."""

    digest = sha256()
    seen = False
    for chunk in chunks:
        if not isinstance(chunk, bytes):
            raise TypeError("capture chunks must be bytes")
        digest.update(chunk)
        seen = True
    if not seen:
        raise ValueError("at least one capture chunk is required")
    return digest.hexdigest()
