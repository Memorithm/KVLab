"""Canonical Boolean-KV handoff for ProspectEngine and FLAT routing.

This module exports structural routing evidence only.  It binds the exact
bit-packed query/page signatures, generation and Hamming threshold to the set
of admitted logical pages.  Numerical K/V remains authoritative in BIKV mode;
no latency, traffic, quality or end-to-end speedup claim follows from this
handoff.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .boolean_kv import BooleanKvCache, BooleanKvError, PackedBits


PROSPECT_BKV_HANDOFF_SCHEMA_V1 = "kvlab.prospect-bkv-handoff/v1"
_HEX_WORD_DIGITS = 16
_EXPECTED_FIELDS = frozenset(
    {
        "schema",
        "signature_bits",
        "generation",
        "query_words",
        "page_words",
        "max_distance",
        "admitted_pages",
    }
)


class ProspectBkvHandoffError(ValueError):
    """Raised when a ProspectEngine BKV handoff is malformed or inconsistent."""


def _require_plain_non_negative_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProspectBkvHandoffError(f"{name} must be a non-negative integer")
    return value


def _word_to_hex(word: int) -> str:
    if not isinstance(word, int) or isinstance(word, bool) or not 0 <= word < 1 << 64:
        raise ProspectBkvHandoffError("packed words must be unsigned 64-bit integers")
    return f"{word:0{_HEX_WORD_DIGITS}x}"


def _hex_to_word(value: object) -> int:
    if not isinstance(value, str) or len(value) != _HEX_WORD_DIGITS:
        raise ProspectBkvHandoffError("packed words must be 16-digit lowercase hexadecimal strings")
    if value != value.lower():
        raise ProspectBkvHandoffError("packed hexadecimal words must be lowercase")
    try:
        word = int(value, 16)
    except ValueError as error:
        raise ProspectBkvHandoffError("packed words contain invalid hexadecimal digits") from error
    if _word_to_hex(word) != value:
        raise ProspectBkvHandoffError("packed words must use canonical 16-digit hexadecimal encoding")
    return word


def _packed_from_hex(signature_bits: int, words: tuple[str, ...]) -> PackedBits:
    try:
        return PackedBits(
            bit_length=signature_bits,
            words=tuple(_hex_to_word(word) for word in words),
        )
    except BooleanKvError as error:
        raise ProspectBkvHandoffError(f"invalid packed Boolean signature: {error}") from error


def _require_word_tuple(name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ProspectBkvHandoffError(f"{name} must be a non-empty JSON array")
    words = tuple(value)
    for word in words:
        _hex_to_word(word)
    return words


def _snapshot_sequence(name: str, value: object) -> tuple[Any, ...]:
    """Detach direct-constructor inputs from caller-owned mutable containers."""
    if not isinstance(value, (tuple, list)):
        raise ProspectBkvHandoffError(f"{name} must be a tuple or list")
    return tuple(value)


@dataclass(frozen=True)
class ProspectBkvHandoffV1:
    """Self-validating, canonical structural handoff for one BKV query.

    The direct constructor snapshots list/tuple inputs before validation,
    including nested page-word lists. Frozen dataclass fields alone would not
    prevent caller-owned lists from changing a previously validated handoff.
    """

    schema: str
    signature_bits: int
    generation: int
    query_words: tuple[str, ...]
    page_words: tuple[tuple[str, ...], ...]
    max_distance: int
    admitted_pages: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.schema != PROSPECT_BKV_HANDOFF_SCHEMA_V1:
            raise ProspectBkvHandoffError("unsupported ProspectEngine BKV handoff schema")

        # Validate exactly the snapshot that will be retained, not the original
        # lists followed by a later copy. Never retain nested mutable aliases.
        query_words = _snapshot_sequence("query_words", self.query_words)
        raw_pages = _snapshot_sequence("page_words", self.page_words)
        page_words = tuple(
            _snapshot_sequence(f"page_words[{index}]", words)
            for index, words in enumerate(raw_pages)
        )
        admitted = _snapshot_sequence("admitted_pages", self.admitted_pages)

        signature_bits = _require_plain_non_negative_int("signature_bits", self.signature_bits)
        if signature_bits == 0:
            raise ProspectBkvHandoffError("signature_bits must be greater than zero")
        _require_plain_non_negative_int("generation", self.generation)
        max_distance = _require_plain_non_negative_int("max_distance", self.max_distance)
        if max_distance > signature_bits:
            raise ProspectBkvHandoffError("max_distance cannot exceed signature_bits")
        if not page_words:
            raise ProspectBkvHandoffError("at least one Boolean KV page is required")

        query = _packed_from_hex(signature_bits, query_words)
        pages = tuple(_packed_from_hex(signature_bits, words) for words in page_words)

        for logical_page in admitted:
            _require_plain_non_negative_int("admitted page", logical_page)
            if logical_page >= len(pages):
                raise ProspectBkvHandoffError("admitted page is outside the page set")
        if tuple(sorted(set(admitted))) != admitted:
            raise ProspectBkvHandoffError(
                "admitted_pages must be unique and sorted in logical page order"
            )

        expected = tuple(
            logical_page
            for logical_page, page in enumerate(pages)
            if query.hamming_distance(page) <= max_distance
        )
        if admitted != expected:
            raise ProspectBkvHandoffError(
                "admitted_pages does not match the exact Hamming rule over recorded signatures"
            )

        object.__setattr__(self, "query_words", query_words)
        object.__setattr__(self, "page_words", page_words)
        object.__setattr__(self, "admitted_pages", admitted)

    @classmethod
    def capture(
        cls,
        cache: BooleanKvCache,
        query: PackedBits,
        *,
        max_distance: int,
    ) -> "ProspectBkvHandoffV1":
        if not isinstance(cache, BooleanKvCache):
            raise ProspectBkvHandoffError("cache must be a BooleanKvCache")
        if len(cache) == 0:
            raise ProspectBkvHandoffError("cannot export an empty Boolean KV cache")

        try:
            matches = cache.search_hamming(query, max_distance=max_distance)
        except BooleanKvError as error:
            raise ProspectBkvHandoffError(f"invalid Boolean KV handoff inputs: {error}") from error

        pages = []
        for logical_page in range(len(cache)):
            page = cache.page(logical_page)
            if page is None:
                raise ProspectBkvHandoffError(
                    "Boolean KV cache contains a page outside the active generation"
                )
            pages.append(tuple(_word_to_hex(word) for word in page.key_signature.words))

        # BooleanKvCache.search_hamming ranks by distance then page.  FLAT masks
        # address logical blocks, so the cross-project handoff intentionally
        # normalizes the same candidate set into logical page order.
        admitted_pages = tuple(sorted(match.logical_page for match in matches))
        return cls(
            schema=PROSPECT_BKV_HANDOFF_SCHEMA_V1,
            signature_bits=cache.signature_bits,
            generation=cache.generation,
            query_words=tuple(_word_to_hex(word) for word in query.words),
            page_words=tuple(pages),
            max_distance=max_distance,
            admitted_pages=admitted_pages,
        )

    def canonical_json(self) -> str:
        """Return stable, whitespace-free JSON with lexicographically sorted keys."""

        return json.dumps(
            {
                "schema": self.schema,
                "signature_bits": self.signature_bits,
                "generation": self.generation,
                "query_words": list(self.query_words),
                "page_words": [list(words) for words in self.page_words],
                "max_distance": self.max_distance,
                "admitted_pages": list(self.admitted_pages),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_canonical_json(cls, payload: str) -> "ProspectBkvHandoffV1":
        if not isinstance(payload, str):
            raise ProspectBkvHandoffError("handoff JSON must be text")
        try:
            decoded: Any = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ProspectBkvHandoffError("invalid ProspectEngine BKV handoff JSON") from error
        if not isinstance(decoded, dict):
            raise ProspectBkvHandoffError("handoff JSON root must be an object")
        if frozenset(decoded) != _EXPECTED_FIELDS:
            raise ProspectBkvHandoffError("handoff JSON fields do not match the v1 schema")

        query_words = _require_word_tuple("query_words", decoded["query_words"])
        raw_pages = decoded["page_words"]
        if not isinstance(raw_pages, list) or not raw_pages:
            raise ProspectBkvHandoffError("page_words must be a non-empty JSON array")
        page_words = tuple(
            _require_word_tuple(f"page_words[{index}]", words)
            for index, words in enumerate(raw_pages)
        )

        raw_admitted = decoded["admitted_pages"]
        if not isinstance(raw_admitted, list):
            raise ProspectBkvHandoffError("admitted_pages must be a JSON array")
        admitted_pages = tuple(
            _require_plain_non_negative_int("admitted page", logical_page)
            for logical_page in raw_admitted
        )

        evidence = cls(
            schema=decoded["schema"],
            signature_bits=_require_plain_non_negative_int(
                "signature_bits", decoded["signature_bits"]
            ),
            generation=_require_plain_non_negative_int("generation", decoded["generation"]),
            query_words=query_words,
            page_words=page_words,
            max_distance=_require_plain_non_negative_int(
                "max_distance", decoded["max_distance"]
            ),
            admitted_pages=admitted_pages,
        )
        if evidence.canonical_json() != payload:
            raise ProspectBkvHandoffError("handoff JSON is valid but not canonical")
        return evidence
