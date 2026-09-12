"""Deterministic BKV-K1 signature and page-aggregation controls.

These are experimental controls, not optimized kernels and not claims that any
signature captures semantic relevance.  Candidate quality must be established
against preregistered dense-attention targets and matched-density baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Iterable, Sequence

from .boolean_kv import BooleanKvError, PackedBits


class SignatureControlError(BooleanKvError):
    """Raised when a BKV-K1 control is malformed or ambiguous."""


def _require_bit_length(bit_length: int) -> int:
    if not isinstance(bit_length, int) or isinstance(bit_length, bool) or bit_length <= 0:
        raise SignatureControlError("bit_length must be a positive integer")
    return bit_length


def _digest_bits(payload: bytes, bit_length: int) -> PackedBits:
    """Expand a tagged payload deterministically into exactly ``bit_length`` bits."""

    _require_bit_length(bit_length)
    needed_bytes = (bit_length + 7) // 8
    output = bytearray()
    counter = 0
    while len(output) < needed_bytes:
        output.extend(
            hashlib.blake2b(
                payload + counter.to_bytes(8, "little"),
                digest_size=64,
                person=b"KVLab-BKV-K1",
            ).digest()
        )
        counter += 1
    bits = []
    for byte in output[:needed_bytes]:
        bits.extend(bool((byte >> bit) & 1) for bit in range(8))
    return PackedBits.from_bools(bits[:bit_length])


def random_control_signature(*, bit_length: int, seed: int, identity: int) -> PackedBits:
    """Stable matched-control signature with no semantic claim.

    The function is deterministic across processes and Python hash seeds.  The
    ``identity`` field lets experiments freeze one random control per page.
    """

    _require_bit_length(bit_length)
    for name, value in (("seed", seed), ("identity", identity)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise SignatureControlError(f"{name} must be a non-negative integer")
    payload = b"random-control\0" + seed.to_bytes(16, "little") + identity.to_bytes(16, "little")
    return _digest_bits(payload, bit_length)


def sign_projection(values: Sequence[float], *, threshold: float = 0.0) -> PackedBits:
    """Map finite numeric values to bits via the preregisterable predicate ``x >= threshold``."""

    if not values:
        raise SignatureControlError("sign projection requires at least one value")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not math.isfinite(float(threshold)):
        raise SignatureControlError("threshold must be finite")
    normalized = []
    for value in values:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise SignatureControlError("sign projection values must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise SignatureControlError("sign projection values must be finite")
        normalized.append(numeric >= float(threshold))
    return PackedBits.from_bools(normalized)


def positional_control_signature(
    *,
    bit_length: int,
    logical_page: int,
    token_start: int,
    token_count: int,
) -> PackedBits:
    """Stable structural/positional control independent of K/V values."""

    _require_bit_length(bit_length)
    for name, value in (
        ("logical_page", logical_page),
        ("token_start", token_start),
        ("token_count", token_count),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise SignatureControlError(f"{name} must be a non-negative integer")
    if token_count == 0:
        raise SignatureControlError("token_count must be non-zero")
    payload = b"positional-control\0" + struct.pack(
        "<QQQ", logical_page, token_start, token_count
    )
    return _digest_bits(payload, bit_length)


def aggregate_page_signatures(
    signatures: Sequence[PackedBits],
    *,
    policy: str,
) -> PackedBits:
    """Aggregate token signatures into a page signature using a frozen Boolean rule.

    Supported controls are ``or``, ``and`` and strict ``majority``.  Majority
    uses ``ones * 2 > n`` so an even tie maps to false; that tie rule is part of
    the experimental contract.
    """

    if not signatures:
        raise SignatureControlError("page aggregation requires at least one signature")
    width = signatures[0].bit_length
    if any(signature.bit_length != width for signature in signatures):
        raise SignatureControlError("all page signatures must have the same bit width")
    if policy not in {"or", "and", "majority"}:
        raise SignatureControlError(f"unsupported page aggregation policy: {policy}")

    result: list[bool] = []
    for bit_index in range(width):
        word_index = bit_index // 64
        bit_offset = bit_index % 64
        ones = sum((signature.words[word_index] >> bit_offset) & 1 for signature in signatures)
        if policy == "or":
            value = ones > 0
        elif policy == "and":
            value = ones == len(signatures)
        else:
            value = ones * 2 > len(signatures)
        result.append(value)
    return PackedBits.from_bools(result)


@dataclass(frozen=True)
class CandidateSelectionMetrics:
    total_pages: int
    target_pages: int
    selected_pages: int
    true_positives: int
    false_negatives: int
    recall: float
    false_negative_rate: float
    candidate_density: float


def evaluate_candidate_selection(
    *,
    selected_pages: Iterable[int],
    dense_target_pages: Iterable[int],
    total_pages: int,
) -> CandidateSelectionMetrics:
    """Evaluate a frozen candidate set against a non-empty declared dense target."""

    if not isinstance(total_pages, int) or isinstance(total_pages, bool) or total_pages <= 0:
        raise SignatureControlError("total_pages must be a positive integer")

    selected = set(selected_pages)
    target = set(dense_target_pages)
    if not target:
        raise SignatureControlError("dense_target_pages must be non-empty")
    for name, pages in (("selected_pages", selected), ("dense_target_pages", target)):
        if any(not isinstance(page, int) or isinstance(page, bool) for page in pages):
            raise SignatureControlError(f"{name} must contain integer page ids")
        if any(page < 0 or page >= total_pages for page in pages):
            raise SignatureControlError(f"{name} contains a page outside [0, total_pages)")

    true_positives = len(selected & target)
    false_negatives = len(target - selected)
    recall = true_positives / len(target)
    false_negative_rate = false_negatives / len(target)
    candidate_density = len(selected) / total_pages
    return CandidateSelectionMetrics(
        total_pages=total_pages,
        target_pages=len(target),
        selected_pages=len(selected),
        true_positives=true_positives,
        false_negatives=false_negatives,
        recall=recall,
        false_negative_rate=false_negative_rate,
        candidate_density=candidate_density,
    )
