"""Bind K9 capture manifests to the preregistered cross-model protocol.

The binding is metadata-only and fail-closed. It ensures source/target captures,
model revisions, tokenizer revisions, split identities, and key RoPE state match
the frozen protocol before mapper fitting or final-holdout evaluation.
"""

from __future__ import annotations

from enum import Enum

from kvlab.cross_model import CrossModelTransferProtocol, ModelRevision
from kvlab.k9_capture import CaptureManifest


class ModelRole(str, Enum):
    SOURCE = "source"
    TARGET = "target"


def bind_capture_manifest(
    manifest: CaptureManifest,
    protocol: CrossModelTransferProtocol,
    role: ModelRole,
) -> None:
    """Fail closed unless a capture manifest exactly matches its protocol role."""

    expected_model = protocol.source if role is ModelRole.SOURCE else protocol.target
    _require_model_revision(manifest, expected_model)

    if manifest.split == "calibration":
        expected_ids = frozenset(protocol.calibration_ids)
    elif manifest.split == "final-holdout":
        expected_ids = frozenset(protocol.final_holdout_ids)
    else:  # CaptureManifest already rejects this, retained for defensive callers.
        raise ValueError(f"unsupported capture split: {manifest.split}")

    if manifest.sample_ids != expected_ids:
        missing = sorted(expected_ids.difference(manifest.sample_ids))
        extra = sorted(manifest.sample_ids.difference(expected_ids))
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise ValueError("capture sample ids do not match frozen protocol: " + "; ".join(details))

    key_captures = tuple(capture for capture in manifest.captures if capture.kind == "key")
    if not key_captures:
        raise ValueError("capture manifest must contain key captures")
    if protocol.remove_rope_from_keys and any(not capture.rope_removed for capture in key_captures):
        raise ValueError("protocol requires RoPE-removed key captures before fitting")
    if not protocol.remove_rope_from_keys and any(capture.rope_removed for capture in key_captures):
        raise ValueError("capture key RoPE state disagrees with frozen protocol")


def _require_model_revision(manifest: CaptureManifest, expected: ModelRevision) -> None:
    mismatches: list[str] = []
    if manifest.model_id != expected.model_id:
        mismatches.append("model_id")
    if manifest.model_revision != expected.revision:
        mismatches.append("model_revision")
    if manifest.tokenizer_revision != expected.tokenizer_revision:
        mismatches.append("tokenizer_revision")
    if mismatches:
        raise ValueError("capture revision mismatch: " + ", ".join(mismatches))
