"""Canonical downstream-consumer provenance receipts for BIKV handoffs.

A receipt links one exact KVLab structural handoff payload to the exact producer
and consumer revisions that exchanged it.  It is provenance only: it carries no
latency, traffic, quality, correctness, or speedup result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .prospect_handoff import ProspectBkvHandoffV1


BIKV_CONSUMER_RECEIPT_SCHEMA_V1 = "kvlab.bikv-consumer-receipt/v1"
_EXPECTED_FIELDS = frozenset(
    {
        "schema",
        "handoff_schema",
        "handoff_sha256",
        "producer_repository",
        "producer_revision",
        "consumer_repository",
        "consumer_revision",
        "consumer_evidence_schema",
    }
)


class BikvConsumerReceiptError(ValueError):
    """Raised when a BIKV consumer provenance receipt is malformed."""


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise BikvConsumerReceiptError(f"{name} must be non-empty text")
    return value


def _require_repository(name: str, value: object) -> str:
    repository = _require_text(name, value)
    owner_repo = repository.split("/")
    if len(owner_repo) != 2 or any(not part for part in owner_repo):
        raise BikvConsumerReceiptError(f"{name} must use owner/repository form")
    return repository


def _require_git_sha(name: str, value: object) -> str:
    revision = _require_text(name, value)
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise BikvConsumerReceiptError(f"{name} must be a 40-digit lowercase hexadecimal Git SHA")
    return revision


def _require_sha256(name: str, value: object) -> str:
    digest = _require_text(name, value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise BikvConsumerReceiptError(f"{name} must be a 64-digit lowercase hexadecimal SHA-256")
    return digest


def _handoff_digest(handoff: ProspectBkvHandoffV1) -> str:
    if not isinstance(handoff, ProspectBkvHandoffV1):
        raise BikvConsumerReceiptError("handoff must be a ProspectBkvHandoffV1")
    return hashlib.sha256(handoff.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BikvConsumerReceiptV1:
    """Canonical receipt proving which revisions exchanged one exact handoff."""

    schema: str
    handoff_schema: str
    handoff_sha256: str
    producer_repository: str
    producer_revision: str
    consumer_repository: str
    consumer_revision: str
    consumer_evidence_schema: str

    def __post_init__(self) -> None:
        if self.schema != BIKV_CONSUMER_RECEIPT_SCHEMA_V1:
            raise BikvConsumerReceiptError("unsupported BIKV consumer receipt schema")
        _require_text("handoff_schema", self.handoff_schema)
        _require_sha256("handoff_sha256", self.handoff_sha256)
        _require_repository("producer_repository", self.producer_repository)
        _require_git_sha("producer_revision", self.producer_revision)
        _require_repository("consumer_repository", self.consumer_repository)
        _require_git_sha("consumer_revision", self.consumer_revision)
        _require_text("consumer_evidence_schema", self.consumer_evidence_schema)

    @classmethod
    def capture(
        cls,
        handoff: ProspectBkvHandoffV1,
        *,
        producer_repository: str,
        producer_revision: str,
        consumer_repository: str,
        consumer_revision: str,
        consumer_evidence_schema: str,
    ) -> "BikvConsumerReceiptV1":
        return cls(
            schema=BIKV_CONSUMER_RECEIPT_SCHEMA_V1,
            handoff_schema=handoff.schema,
            handoff_sha256=_handoff_digest(handoff),
            producer_repository=producer_repository,
            producer_revision=producer_revision,
            consumer_repository=consumer_repository,
            consumer_revision=consumer_revision,
            consumer_evidence_schema=consumer_evidence_schema,
        )

    def verifies_handoff(self, handoff: ProspectBkvHandoffV1) -> bool:
        """Return whether ``handoff`` is byte-for-byte the payload this receipt binds."""

        return handoff.schema == self.handoff_schema and _handoff_digest(handoff) == self.handoff_sha256

    def canonical_json(self) -> str:
        return json.dumps(
            {
                "schema": self.schema,
                "handoff_schema": self.handoff_schema,
                "handoff_sha256": self.handoff_sha256,
                "producer_repository": self.producer_repository,
                "producer_revision": self.producer_revision,
                "consumer_repository": self.consumer_repository,
                "consumer_revision": self.consumer_revision,
                "consumer_evidence_schema": self.consumer_evidence_schema,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_canonical_json(cls, payload: str) -> "BikvConsumerReceiptV1":
        if not isinstance(payload, str):
            raise BikvConsumerReceiptError("receipt JSON must be text")
        try:
            decoded: Any = json.loads(payload)
        except json.JSONDecodeError as error:
            raise BikvConsumerReceiptError("invalid BIKV consumer receipt JSON") from error
        if not isinstance(decoded, dict):
            raise BikvConsumerReceiptError("receipt JSON root must be an object")
        if frozenset(decoded) != _EXPECTED_FIELDS:
            raise BikvConsumerReceiptError("receipt JSON fields do not match the v1 schema")
        receipt = cls(
            schema=decoded["schema"],
            handoff_schema=decoded["handoff_schema"],
            handoff_sha256=decoded["handoff_sha256"],
            producer_repository=decoded["producer_repository"],
            producer_revision=decoded["producer_revision"],
            consumer_repository=decoded["consumer_repository"],
            consumer_revision=decoded["consumer_revision"],
            consumer_evidence_schema=decoded["consumer_evidence_schema"],
        )
        if receipt.canonical_json() != payload:
            raise BikvConsumerReceiptError("receipt JSON is valid but not canonical")
        return receipt
