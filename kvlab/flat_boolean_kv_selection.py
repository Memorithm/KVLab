"""Fail-closed consumer for FLAT M13B.5 Boolean-KV page-selection evidence.

FLAT owns page-routing execution and the canonical producer encoding. KVLab owns
retention and experimental interpretation. This module accepts only the exact
``flat.boolean-kv-selection.v1`` encoding from the pinned FLAT revision, verifies
its producer checksum and structural/accounting invariants, and retains a SHA-256
content identity. It does not infer physical DRAM traffic, latency improvement,
model quality, residency, or speedup from logical byte accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

FLAT_BOOLEAN_KV_SELECTION_SCHEMA = "flat.boolean-kv-selection.v1"
FLAT_BOOLEAN_KV_SELECTION_REFERENCE_REVISION = (
    "c1249bad4fd3f41c93a35f8a6204c98de9a3b687"
)
_U64_MAX = (1 << 64) - 1
_TOP_FIELDS = (
    "schema",
    "generation",
    "signature_bits",
    "live_tokens",
    "mapped_pages",
    "boolean_pages_scanned",
    "boolean_key_bytes_read",
    "full_numerical_kv_bytes",
    "selected_numerical_kv_bytes",
    "avoided_numerical_kv_bytes",
    "selected_pages",
    "evidence_checksum",
)
_PAGE_FIELDS = (
    "logical_page",
    "physical_page",
    "live_tokens",
    "hamming_distance",
    "xnor_matches",
)
_CHECKSUM_FIELDS = ("algorithm", "value")


class FlatBooleanKvSelectionError(ValueError):
    """Raised when FLAT page-selection evidence is malformed or non-canonical."""


def _u64(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FlatBooleanKvSelectionError(f"{name} must be an integer u64")
    if not 0 <= value <= _U64_MAX:
        raise FlatBooleanKvSelectionError(f"{name} must be within the u64 range")
    return value


def _fnv1a64(payload: bytes) -> str:
    value = 0xCBF29CE484222325
    for byte in payload:
        value ^= byte
        value = (value * 0x100000001B3) & _U64_MAX
    return f"{value:016x}"


@dataclass(frozen=True, slots=True)
class FlatBooleanKvSelectedPageV1:
    logical_page: int
    physical_page: int
    live_tokens: int
    hamming_distance: int
    xnor_matches: int

    def validate(self, *, mapped_pages: int, signature_bits: int) -> None:
        logical_page = _u64("logical_page", self.logical_page)
        _u64("physical_page", self.physical_page)
        live_tokens = _u64("page.live_tokens", self.live_tokens)
        hamming = _u64("hamming_distance", self.hamming_distance)
        xnor = _u64("xnor_matches", self.xnor_matches)
        if logical_page >= mapped_pages:
            raise FlatBooleanKvSelectionError(
                f"logical_page {logical_page} is outside {mapped_pages} mapped pages"
            )
        if live_tokens == 0:
            raise FlatBooleanKvSelectionError("selected pages must retain live tokens")
        if hamming + xnor != signature_bits:
            raise FlatBooleanKvSelectionError(
                "hamming_distance + xnor_matches must equal signature_bits"
            )


@dataclass(frozen=True, slots=True)
class FlatBooleanKvSelectionV1:
    schema: str
    generation: int
    signature_bits: int
    live_tokens: int
    mapped_pages: int
    boolean_pages_scanned: int
    boolean_key_bytes_read: int
    full_numerical_kv_bytes: int
    selected_numerical_kv_bytes: int
    avoided_numerical_kv_bytes: int
    selected_pages: tuple[FlatBooleanKvSelectedPageV1, ...]
    evidence_checksum: str

    def validate(self) -> None:
        if self.schema != FLAT_BOOLEAN_KV_SELECTION_SCHEMA:
            raise FlatBooleanKvSelectionError(f"unsupported selection schema: {self.schema!r}")
        _u64("generation", self.generation)
        signature_bits = _u64("signature_bits", self.signature_bits)
        live_tokens = _u64("live_tokens", self.live_tokens)
        mapped_pages = _u64("mapped_pages", self.mapped_pages)
        scanned = _u64("boolean_pages_scanned", self.boolean_pages_scanned)
        boolean_bytes = _u64("boolean_key_bytes_read", self.boolean_key_bytes_read)
        full_bytes = _u64("full_numerical_kv_bytes", self.full_numerical_kv_bytes)
        selected_bytes = _u64(
            "selected_numerical_kv_bytes", self.selected_numerical_kv_bytes
        )
        avoided_bytes = _u64(
            "avoided_numerical_kv_bytes", self.avoided_numerical_kv_bytes
        )
        if signature_bits == 0:
            raise FlatBooleanKvSelectionError("signature_bits must be non-zero")
        if scanned != mapped_pages:
            raise FlatBooleanKvSelectionError(
                "boolean_pages_scanned must equal mapped_pages for the v1 full-scan router"
            )
        if len(self.selected_pages) > mapped_pages:
            raise FlatBooleanKvSelectionError("selected page count exceeds mapped_pages")
        if selected_bytes + avoided_bytes != full_bytes:
            raise FlatBooleanKvSelectionError("numerical byte accounting is inconsistent")
        if mapped_pages and boolean_bytes == 0:
            raise FlatBooleanKvSelectionError(
                "mapped pages require non-zero Boolean key bytes read"
            )

        previous_logical: int | None = None
        physical_pages: set[int] = set()
        selected_live_tokens = 0
        for page in self.selected_pages:
            if not isinstance(page, FlatBooleanKvSelectedPageV1):
                raise FlatBooleanKvSelectionError(
                    "selected_pages must contain FlatBooleanKvSelectedPageV1 values"
                )
            page.validate(mapped_pages=mapped_pages, signature_bits=signature_bits)
            if previous_logical is not None and page.logical_page <= previous_logical:
                raise FlatBooleanKvSelectionError(
                    "selected logical pages must be strictly increasing"
                )
            if page.physical_page in physical_pages:
                raise FlatBooleanKvSelectionError("selected physical pages must be unique")
            physical_pages.add(page.physical_page)
            selected_live_tokens += page.live_tokens
            previous_logical = page.logical_page
        if selected_live_tokens > live_tokens:
            raise FlatBooleanKvSelectionError(
                "selected page live-token total exceeds live_tokens"
            )
        if not isinstance(self.evidence_checksum, str) or len(self.evidence_checksum) != 16:
            raise FlatBooleanKvSelectionError("evidence checksum must be 16 lowercase hex digits")
        if self.evidence_checksum != self.evidence_checksum.lower() or any(
            character not in "0123456789abcdef" for character in self.evidence_checksum
        ):
            raise FlatBooleanKvSelectionError("evidence checksum must be 16 lowercase hex digits")
        expected = _fnv1a64(self._producer_prefix_bytes())
        if self.evidence_checksum != expected:
            raise FlatBooleanKvSelectionError(
                "evidence checksum does not match the canonical producer prefix"
            )

    def _producer_prefix_bytes(self) -> bytes:
        payload = {
            "schema": self.schema,
            "generation": self.generation,
            "signature_bits": self.signature_bits,
            "live_tokens": self.live_tokens,
            "mapped_pages": self.mapped_pages,
            "boolean_pages_scanned": self.boolean_pages_scanned,
            "boolean_key_bytes_read": self.boolean_key_bytes_read,
            "full_numerical_kv_bytes": self.full_numerical_kv_bytes,
            "selected_numerical_kv_bytes": self.selected_numerical_kv_bytes,
            "avoided_numerical_kv_bytes": self.avoided_numerical_kv_bytes,
            "selected_pages": [
                {
                    "logical_page": page.logical_page,
                    "physical_page": page.physical_page,
                    "live_tokens": page.live_tokens,
                    "hamming_distance": page.hamming_distance,
                    "xnor_matches": page.xnor_matches,
                }
                for page in self.selected_pages
            ],
        }
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return encoded[:-1]

    def canonical_json_bytes(self) -> bytes:
        self.validate()
        prefix = self._producer_prefix_bytes()
        # FLAT computes FNV-1a over the object prefix before adding this final field.
        return (
            prefix
            + b',"evidence_checksum":{"algorithm":"fnv1a64","value":"'
            + self.evidence_checksum.encode("ascii")
            + b'"}}'
        )

    def selection_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()

    @classmethod
    def from_canonical_json_bytes(cls, payload: bytes) -> "FlatBooleanKvSelectionV1":
        if not isinstance(payload, bytes) or not payload:
            raise FlatBooleanKvSelectionError("selection payload must be non-empty bytes")
        try:
            raw: Any = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FlatBooleanKvSelectionError("selection payload must be UTF-8 JSON") from exc
        if not isinstance(raw, dict) or tuple(raw) != _TOP_FIELDS:
            raise FlatBooleanKvSelectionError(
                "selection payload fields/order do not match the FLAT schema"
            )
        raw_pages = raw["selected_pages"]
        if not isinstance(raw_pages, list):
            raise FlatBooleanKvSelectionError("selected_pages must be a JSON array")
        pages: list[FlatBooleanKvSelectedPageV1] = []
        for raw_page in raw_pages:
            if not isinstance(raw_page, dict) or tuple(raw_page) != _PAGE_FIELDS:
                raise FlatBooleanKvSelectionError(
                    "selected page fields/order do not match the FLAT schema"
                )
            pages.append(FlatBooleanKvSelectedPageV1(**raw_page))
        checksum = raw["evidence_checksum"]
        if not isinstance(checksum, dict) or tuple(checksum) != _CHECKSUM_FIELDS:
            raise FlatBooleanKvSelectionError("evidence_checksum fields do not match schema")
        if checksum["algorithm"] != "fnv1a64":
            raise FlatBooleanKvSelectionError("unsupported selection checksum algorithm")
        selection = cls(
            schema=raw["schema"],
            generation=raw["generation"],
            signature_bits=raw["signature_bits"],
            live_tokens=raw["live_tokens"],
            mapped_pages=raw["mapped_pages"],
            boolean_pages_scanned=raw["boolean_pages_scanned"],
            boolean_key_bytes_read=raw["boolean_key_bytes_read"],
            full_numerical_kv_bytes=raw["full_numerical_kv_bytes"],
            selected_numerical_kv_bytes=raw["selected_numerical_kv_bytes"],
            avoided_numerical_kv_bytes=raw["avoided_numerical_kv_bytes"],
            selected_pages=tuple(pages),
            evidence_checksum=checksum["value"],
        )
        selection.validate()
        if selection.canonical_json_bytes() != payload:
            raise FlatBooleanKvSelectionError(
                "selection payload is not in canonical FLAT JSON form"
            )
        return selection
