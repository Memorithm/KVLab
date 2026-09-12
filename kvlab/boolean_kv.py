"""First-class Boolean KV cache contracts for KVLab.

BKV-K0 is deliberately an exact, deterministic CPU oracle.  It defines the
bit-level representation and lifecycle used by later SIMD, NUMA, WGPU and
CPU/GPU-cooperative experiments.  It makes no performance claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


WORD_BITS = 64
WORD_MASK = (1 << WORD_BITS) - 1
BOOLEAN_KV_SCHEMA_VERSION = 1


class BooleanKvError(ValueError):
    """Raised when a Boolean KV representation is malformed or incompatible."""


def _word_count(bit_length: int) -> int:
    if not isinstance(bit_length, int) or isinstance(bit_length, bool) or bit_length <= 0:
        raise BooleanKvError("bit_length must be a positive integer")
    return (bit_length + WORD_BITS - 1) // WORD_BITS


@dataclass(frozen=True)
class PackedBits:
    """Canonical little-bit-order packed Boolean vector.

    Bit ``i`` lives in ``words[i // 64]`` at bit position ``i % 64``.  Bits
    above ``bit_length`` in the final word are required to be zero.  This makes
    byte-for-byte equality a safe exact-representation equality test.
    """

    bit_length: int
    words: tuple[int, ...]

    def __post_init__(self) -> None:
        expected = _word_count(self.bit_length)
        if len(self.words) != expected:
            raise BooleanKvError(
                f"expected {expected} packed words for {self.bit_length} bits, got {len(self.words)}"
            )
        for word in self.words:
            if not isinstance(word, int) or isinstance(word, bool) or not 0 <= word <= WORD_MASK:
                raise BooleanKvError("packed words must be unsigned 64-bit integers")
        tail_bits = self.bit_length % WORD_BITS
        if tail_bits:
            valid_mask = (1 << tail_bits) - 1
            if self.words[-1] & ~valid_mask:
                raise BooleanKvError("unused high bits in the final packed word must be zero")

    @classmethod
    def from_bools(cls, bits: Iterable[bool]) -> "PackedBits":
        values = tuple(bits)
        if not values:
            raise BooleanKvError("Boolean signatures must contain at least one bit")
        if any(type(bit) is not bool for bit in values):
            raise BooleanKvError("Boolean signatures must contain only bool values")
        words = [0] * _word_count(len(values))
        for index, bit in enumerate(values):
            if bit:
                words[index // WORD_BITS] |= 1 << (index % WORD_BITS)
        return cls(bit_length=len(values), words=tuple(words))

    @property
    def logical_bits(self) -> int:
        return self.bit_length

    @property
    def physical_bits(self) -> int:
        return len(self.words) * WORD_BITS

    @property
    def physical_bytes(self) -> int:
        return len(self.words) * (WORD_BITS // 8)

    def hamming_distance(self, other: "PackedBits") -> int:
        self._require_same_width(other)
        return sum((left ^ right).bit_count() for left, right in zip(self.words, other.words))

    def xnor_matches(self, other: "PackedBits") -> int:
        self._require_same_width(other)
        return self.bit_length - self.hamming_distance(other)

    def _require_same_width(self, other: "PackedBits") -> None:
        if not isinstance(other, PackedBits):
            raise BooleanKvError("comparison requires another PackedBits value")
        if self.bit_length != other.bit_length:
            raise BooleanKvError(
                f"signature width mismatch: {self.bit_length} != {other.bit_length}"
            )


@dataclass(frozen=True)
class BooleanKvPage:
    """One logical Boolean-KV page.

    ``key_signature`` is mandatory.  ``value_signature`` is optional because
    BIKV may use Boolean K only to select authoritative numerical K/V pages.
    """

    logical_page: int
    generation: int
    key_signature: PackedBits
    value_signature: PackedBits | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.logical_page, int) or isinstance(self.logical_page, bool) or self.logical_page < 0:
            raise BooleanKvError("logical_page must be a non-negative integer")
        if not isinstance(self.generation, int) or isinstance(self.generation, bool) or self.generation < 0:
            raise BooleanKvError("generation must be a non-negative integer")
        if self.value_signature is not None and self.value_signature.bit_length != self.key_signature.bit_length:
            raise BooleanKvError("K and V Boolean signatures must use the same width in BKV-K0")


@dataclass(frozen=True)
class BooleanKvAccounting:
    pages: int
    signature_bits: int
    key_logical_bits: int
    value_logical_bits: int
    key_physical_bytes: int
    value_physical_bytes: int
    total_physical_bytes: int


@dataclass(frozen=True)
class BooleanKvMatch:
    logical_page: int
    hamming_distance: int
    xnor_matches: int


class BooleanKvCache:
    """Deterministic first-class Boolean KV page cache.

    This object owns Boolean metadata only.  In BIKV mode, numerical K/V remains
    authoritative and this cache is an index/router.  NBKV semantics are a
    later research gate and must not be inferred from this class.
    """

    def __init__(self, signature_bits: int) -> None:
        _word_count(signature_bits)
        self._signature_bits = signature_bits
        self._generation = 0
        self._pages: list[BooleanKvPage] = []

    @property
    def signature_bits(self) -> int:
        return self._signature_bits

    @property
    def generation(self) -> int:
        return self._generation

    def __len__(self) -> int:
        return len(self._pages)

    def append(
        self,
        key_signature: PackedBits,
        value_signature: PackedBits | None = None,
    ) -> BooleanKvPage:
        self._require_width(key_signature)
        if value_signature is not None:
            self._require_width(value_signature)
        page = BooleanKvPage(
            logical_page=len(self._pages),
            generation=self._generation,
            key_signature=key_signature,
            value_signature=value_signature,
        )
        self._pages.append(page)
        return page

    def page(self, logical_page: int) -> BooleanKvPage | None:
        if not isinstance(logical_page, int) or isinstance(logical_page, bool) or logical_page < 0:
            raise BooleanKvError("logical_page must be a non-negative integer")
        if logical_page >= len(self._pages):
            return None
        page = self._pages[logical_page]
        return page if page.generation == self._generation else None

    def reset(self) -> None:
        self._generation += 1
        self._pages.clear()

    def accounting(self) -> BooleanKvAccounting:
        key_logical_bits = sum(page.key_signature.logical_bits for page in self._pages)
        value_pages = [page for page in self._pages if page.value_signature is not None]
        value_logical_bits = sum(page.value_signature.logical_bits for page in value_pages if page.value_signature is not None)
        key_physical_bytes = sum(page.key_signature.physical_bytes for page in self._pages)
        value_physical_bytes = sum(
            page.value_signature.physical_bytes
            for page in value_pages
            if page.value_signature is not None
        )
        return BooleanKvAccounting(
            pages=len(self._pages),
            signature_bits=self._signature_bits,
            key_logical_bits=key_logical_bits,
            value_logical_bits=value_logical_bits,
            key_physical_bytes=key_physical_bytes,
            value_physical_bytes=value_physical_bytes,
            total_physical_bytes=key_physical_bytes + value_physical_bytes,
        )

    def search_hamming(
        self,
        query: PackedBits,
        *,
        max_distance: int,
        limit: int | None = None,
    ) -> tuple[BooleanKvMatch, ...]:
        """Return a deterministic distance-then-page ordered candidate set."""

        self._require_width(query)
        if not isinstance(max_distance, int) or isinstance(max_distance, bool) or max_distance < 0:
            raise BooleanKvError("max_distance must be a non-negative integer")
        if max_distance > self._signature_bits:
            raise BooleanKvError("max_distance cannot exceed signature width")
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
        ):
            raise BooleanKvError("limit must be a positive integer when provided")

        matches = []
        for page in self._pages:
            distance = query.hamming_distance(page.key_signature)
            if distance <= max_distance:
                matches.append(
                    BooleanKvMatch(
                        logical_page=page.logical_page,
                        hamming_distance=distance,
                        xnor_matches=self._signature_bits - distance,
                    )
                )
        matches.sort(key=lambda item: (item.hamming_distance, item.logical_page))
        if limit is not None:
            matches = matches[:limit]
        return tuple(matches)

    def _require_width(self, signature: PackedBits) -> None:
        if not isinstance(signature, PackedBits):
            raise BooleanKvError("Boolean KV signatures must be PackedBits")
        if signature.bit_length != self._signature_bits:
            raise BooleanKvError(
                f"signature width mismatch: expected {self._signature_bits}, got {signature.bit_length}"
            )


def pack_bits(bits: Sequence[bool]) -> PackedBits:
    """Convenience wrapper kept explicit for experiment manifests/runners."""

    return PackedBits.from_bools(bits)
