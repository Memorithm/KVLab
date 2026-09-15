"""Opaque provenance receipts for upstream BIKV evidence envelopes.

This module deliberately does not interpret scientific or performance fields from
an upstream evidence envelope. It binds an exact JSON byte sequence to producer
identity so KVLab can retain/cross-link evidence without silently changing its
meaning.

A valid receipt attests only byte identity and provenance. It does not validate
latency, bandwidth, KV traffic, model quality, hardware effects, or any scientific
claim contained in the source artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import re
from typing import Any


_SCHEMA = "kvlab.bikv-evidence-receipt.v1"
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REPO_RE = re.compile(r"^[^/\s]+/[^/\s]+$")
_RECEIPT_FIELDS = {
    "schema",
    "producer_repo",
    "producer_commit",
    "source_sha256",
    "source_bytes",
}


@dataclass(frozen=True, slots=True)
class BikvEvidenceReceiptV1:
    """Exact-byte identity receipt for an upstream JSON evidence artifact."""

    schema: str
    producer_repo: str
    producer_commit: str
    source_sha256: str
    source_bytes: int

    def validate(self) -> None:
        """Reject malformed provenance without interpreting source semantics."""

        if not isinstance(self.schema, str) or self.schema != _SCHEMA:
            raise ValueError(f"unsupported receipt schema: {self.schema!r}")
        if not isinstance(self.producer_repo, str) or not _REPO_RE.fullmatch(self.producer_repo):
            raise ValueError("producer_repo must be in owner/name form")
        if not isinstance(self.producer_commit, str) or not _SHA1_RE.fullmatch(self.producer_commit):
            raise ValueError("producer_commit must be a lowercase 40-hex commit SHA")
        if not isinstance(self.source_sha256, str) or not _SHA256_RE.fullmatch(self.source_sha256):
            raise ValueError("source_sha256 must be a lowercase 64-hex digest")
        if isinstance(self.source_bytes, bool) or not isinstance(self.source_bytes, int):
            raise ValueError("source_bytes must be an integer")
        if self.source_bytes <= 0:
            raise ValueError("source_bytes must be positive")

    def canonical_json_bytes(self) -> bytes:
        """Return deterministic receipt bytes suitable for hashing/storage."""

        self.validate()
        return json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    def receipt_sha256(self) -> str:
        """Hash the canonical receipt, not the upstream source artifact."""

        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()

    def verify_source_bytes(self, payload: bytes) -> None:
        """Fail closed unless ``payload`` is the exact byte sequence in the receipt.

        Verification checks only byte length and SHA-256 identity. It deliberately
        does not parse or reinterpret the upstream JSON after receipt creation.
        """

        self.validate()
        if not isinstance(payload, bytes):
            raise ValueError("payload must be bytes")
        if len(payload) != self.source_bytes:
            raise ValueError("source byte length does not match receipt")
        digest = hashlib.sha256(payload).hexdigest()
        if not hmac.compare_digest(digest, self.source_sha256):
            raise ValueError("source SHA-256 does not match receipt")

    @classmethod
    def from_canonical_json_bytes(cls, payload: bytes) -> "BikvEvidenceReceiptV1":
        """Reconstruct a receipt only from its exact canonical JSON encoding.

        Rejecting alternate whitespace/key order prevents two byte encodings from
        representing the same receipt identity in a persistent evidence store.
        """

        if not isinstance(payload, bytes) or not payload:
            raise ValueError("receipt payload must be non-empty bytes")
        try:
            decoded = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("receipt payload must be UTF-8 JSON") from exc
        try:
            parsed: Any = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise ValueError("receipt payload must be valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("receipt payload must contain a top-level JSON object")
        if set(parsed) != _RECEIPT_FIELDS:
            raise ValueError("receipt payload fields do not match schema")

        receipt = cls(
            schema=parsed["schema"],
            producer_repo=parsed["producer_repo"],
            producer_commit=parsed["producer_commit"],
            source_sha256=parsed["source_sha256"],
            source_bytes=parsed["source_bytes"],
        )
        receipt.validate()
        if receipt.canonical_json_bytes() != payload:
            raise ValueError("receipt payload is not in canonical JSON form")
        return receipt

    @classmethod
    def from_json_bytes(
        cls,
        *,
        producer_repo: str,
        producer_commit: str,
        payload: bytes,
    ) -> "BikvEvidenceReceiptV1":
        """Bind exact upstream bytes after only structural JSON validation.

        The payload must be UTF-8 JSON with an object at the top level. The digest
        is computed over the original bytes exactly as supplied; the parsed object
        is intentionally discarded so KVLab cannot accidentally normalize or
        reinterpret upstream evidence while producing this provenance receipt.
        """

        if not isinstance(payload, bytes) or not payload:
            raise ValueError("payload must be non-empty bytes")
        try:
            decoded = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("payload must be UTF-8 JSON") from exc
        try:
            parsed: Any = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise ValueError("payload must be valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("payload must contain a top-level JSON object")

        receipt = cls(
            schema=_SCHEMA,
            producer_repo=producer_repo,
            producer_commit=producer_commit,
            source_sha256=hashlib.sha256(payload).hexdigest(),
            source_bytes=len(payload),
        )
        receipt.validate()
        return receipt
