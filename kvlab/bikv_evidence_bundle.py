"""Canonical cross-links for multiple opaque BIKV evidence receipts.

The bundle advances the KVLab BKV-K6 provenance gate without interpreting any
scientific field from the upstream artifacts.  Each member is an already
validated :class:`BikvEvidenceReceiptV1`; the bundle binds a declared role to the
canonical receipt SHA-256 and repeats the producer/source identities needed to
detect accidental substitution during replay.

A valid bundle proves only that a set of exact provenance receipts was grouped
together under distinct declared roles.  It does not establish latency, traffic,
quality, recall, hardware effects, first-token readiness or speedup.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Iterable

from .bikv_evidence_receipt import BikvEvidenceReceiptV1

_SCHEMA = "kvlab.bikv-evidence-bundle.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROLE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_ENTRY_FIELDS = {
    "role",
    "receipt_sha256",
    "producer_repo",
    "producer_commit",
    "source_sha256",
    "source_bytes",
}
_BUNDLE_FIELDS = {"schema", "entries"}


@dataclass(frozen=True, slots=True)
class BikvEvidenceBundleEntryV1:
    """One role-bound opaque evidence receipt inside a bundle."""

    role: str
    receipt_sha256: str
    producer_repo: str
    producer_commit: str
    source_sha256: str
    source_bytes: int

    @classmethod
    def from_receipt(
        cls,
        *,
        role: str,
        receipt: BikvEvidenceReceiptV1,
    ) -> "BikvEvidenceBundleEntryV1":
        """Create an entry from one fully validated canonical receipt."""

        receipt.validate()
        entry = cls(
            role=role,
            receipt_sha256=receipt.receipt_sha256(),
            producer_repo=receipt.producer_repo,
            producer_commit=receipt.producer_commit,
            source_sha256=receipt.source_sha256,
            source_bytes=receipt.source_bytes,
        )
        entry.validate()
        return entry

    def validate(self) -> None:
        """Reject malformed role or copied provenance fields."""

        if not isinstance(self.role, str) or not _ROLE_RE.fullmatch(self.role):
            raise ValueError("role must match [a-z0-9][a-z0-9._-]{0,63}")
        if not isinstance(self.receipt_sha256, str) or not _SHA256_RE.fullmatch(
            self.receipt_sha256
        ):
            raise ValueError("receipt_sha256 must be a lowercase 64-hex digest")

        # Reuse the receipt validator for producer/source field contracts.  The
        # stored receipt hash is deliberately not inferred from these repeated
        # fields; verify_receipt performs that exact identity check.
        proxy = BikvEvidenceReceiptV1(
            schema="kvlab.bikv-evidence-receipt.v1",
            producer_repo=self.producer_repo,
            producer_commit=self.producer_commit,
            source_sha256=self.source_sha256,
            source_bytes=self.source_bytes,
        )
        proxy.validate()

    def verify_receipt(self, receipt: BikvEvidenceReceiptV1) -> None:
        """Fail closed unless ``receipt`` is exactly the receipt bound here."""

        self.validate()
        receipt.validate()
        if receipt.receipt_sha256() != self.receipt_sha256:
            raise ValueError("receipt SHA-256 does not match bundle entry")
        if receipt.producer_repo != self.producer_repo:
            raise ValueError("producer repository does not match bundle entry")
        if receipt.producer_commit != self.producer_commit:
            raise ValueError("producer commit does not match bundle entry")
        if receipt.source_sha256 != self.source_sha256:
            raise ValueError("source SHA-256 does not match bundle entry")
        if receipt.source_bytes != self.source_bytes:
            raise ValueError("source byte length does not match bundle entry")


@dataclass(frozen=True, slots=True)
class BikvEvidenceBundleV1:
    """Canonical ordered set of distinct-role BIKV evidence receipt links."""

    schema: str
    entries: tuple[BikvEvidenceBundleEntryV1, ...]

    @classmethod
    def from_receipts(
        cls,
        entries: Iterable[tuple[str, BikvEvidenceReceiptV1]],
    ) -> "BikvEvidenceBundleV1":
        """Build a canonical bundle sorted by role from receipt objects."""

        materialized = tuple(
            BikvEvidenceBundleEntryV1.from_receipt(role=role, receipt=receipt)
            for role, receipt in entries
        )
        bundle = cls(schema=_SCHEMA, entries=tuple(sorted(materialized, key=lambda item: item.role)))
        bundle.validate()
        return bundle

    def validate(self) -> None:
        """Require a non-empty canonical role ordering with no duplicate receipts."""

        if self.schema != _SCHEMA:
            raise ValueError(f"unsupported bundle schema: {self.schema!r}")
        if not isinstance(self.entries, tuple) or not self.entries:
            raise ValueError("entries must be a non-empty tuple")

        previous_role: str | None = None
        receipt_hashes: set[str] = set()
        for entry in self.entries:
            if not isinstance(entry, BikvEvidenceBundleEntryV1):
                raise ValueError("entries must contain BikvEvidenceBundleEntryV1 values")
            entry.validate()
            if previous_role is not None and entry.role <= previous_role:
                raise ValueError("entries must have unique roles in ascending canonical order")
            if entry.receipt_sha256 in receipt_hashes:
                raise ValueError("one receipt cannot occupy multiple bundle roles")
            previous_role = entry.role
            receipt_hashes.add(entry.receipt_sha256)

    def canonical_json_bytes(self) -> bytes:
        """Return deterministic bundle JSON bytes suitable for persistence."""

        self.validate()
        payload = {
            "schema": self.schema,
            "entries": [asdict(entry) for entry in self.entries],
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    def bundle_sha256(self) -> str:
        """Hash the canonical bundle bytes."""

        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()

    @classmethod
    def from_canonical_json_bytes(cls, payload: bytes) -> "BikvEvidenceBundleV1":
        """Reconstruct only an exact canonical bundle encoding."""

        if not isinstance(payload, bytes) or not payload:
            raise ValueError("bundle payload must be non-empty bytes")
        try:
            parsed: Any = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("bundle payload must be UTF-8 JSON") from exc
        if not isinstance(parsed, dict) or set(parsed) != _BUNDLE_FIELDS:
            raise ValueError("bundle payload fields do not match schema")
        raw_entries = parsed["entries"]
        if not isinstance(raw_entries, list):
            raise ValueError("entries must be a JSON array")

        decoded_entries: list[BikvEvidenceBundleEntryV1] = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict) or set(raw_entry) != _ENTRY_FIELDS:
                raise ValueError("bundle entry fields do not match schema")
            decoded_entries.append(BikvEvidenceBundleEntryV1(**raw_entry))

        bundle = cls(schema=parsed["schema"], entries=tuple(decoded_entries))
        bundle.validate()
        if bundle.canonical_json_bytes() != payload:
            raise ValueError("bundle payload is not in canonical JSON form")
        return bundle
