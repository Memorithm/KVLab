"""Fail-closed retention for FLAT portable WGPU Boolean-router parity evidence.

FLAT owns WGPU routing execution and canonical producer encoding. KVLab owns the
BKV-K7 experimental protocol and interpretation. This consumer accepts the
``flat.boolean-kv-wgpu-parity.v1`` record, verifies exact compact bytes,
structural geometry, checksum, CPU/WGPU candidate-set consistency and retains a
SHA-256 content identity.

A well-formed candidate-set mismatch is retained as negative evidence. It is
never silently discarded, while ``require_exact_match`` fail-closes any later
performance-comparison gate. This module contains no timing, physical traffic,
energy, model-quality or speedup inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

FLAT_BOOLEAN_KV_WGPU_PARITY_SCHEMA = "flat.boolean-kv-wgpu-parity.v1"
# Candidate producer head in FLAT-ATTENTION PR #255. This is deliberately not a
# qualified reference revision. Replace it with the final merge SHA only after
# that PR's exact-head CI and material review are green.
FLAT_BOOLEAN_KV_WGPU_PARITY_CANDIDATE_REVISION = (
    "9f2e6507780ca2c3d6f6f89d3bdf297b11b805ce"
)

_U32_MAX = (1 << 32) - 1
_U64_MAX = (1 << 64) - 1
_TOP_FIELDS = (
    "schema",
    "signature_bits",
    "key_count",
    "words_per_signature",
    "max_distance",
    "query_words_u32",
    "key_words_u32",
    "cpu_admitted_blocks",
    "wgpu_admitted_blocks",
    "exact_candidate_set_match",
    "parity_checksum",
)
_CHECKSUM_FIELDS = ("algorithm", "value")


class FlatBooleanKvWgpuParityError(ValueError):
    """Raised when FLAT WGPU parity evidence is malformed or non-canonical."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FlatBooleanKvWgpuParityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _bounded_int(name: str, value: Any, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FlatBooleanKvWgpuParityError(f"{name} must be an integer")
    if not 0 <= value <= maximum:
        raise FlatBooleanKvWgpuParityError(f"{name} is outside its declared unsigned range")
    return value


def _u32(name: str, value: Any) -> int:
    return _bounded_int(name, value, _U32_MAX)


def _u64(name: str, value: Any) -> int:
    return _bounded_int(name, value, _U64_MAX)


def _fnv1a64(payload: bytes) -> str:
    value = 0xCBF29CE484222325
    for byte in payload:
        value ^= byte
        value = (value * 0x100000001B3) & _U64_MAX
    return f"{value:016x}"


def _compact_json(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _checksum_value(value: Any) -> str:
    if not isinstance(value, dict) or tuple(value) != _CHECKSUM_FIELDS:
        raise FlatBooleanKvWgpuParityError(
            "parity_checksum fields/order do not match algorithm/value schema"
        )
    if value["algorithm"] != "fnv1a64":
        raise FlatBooleanKvWgpuParityError("parity_checksum algorithm must be fnv1a64")
    checksum = value["value"]
    if (
        not isinstance(checksum, str)
        or len(checksum) != 16
        or checksum != checksum.lower()
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        raise FlatBooleanKvWgpuParityError(
            "parity_checksum value must be 16 lowercase hexadecimal digits"
        )
    return checksum


def _u32_array(name: str, value: Any, expected_len: int) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != expected_len:
        raise FlatBooleanKvWgpuParityError(
            f"{name} must contain exactly {expected_len} u32 values"
        )
    return tuple(_u32(f"{name}[{index}]", item) for index, item in enumerate(value))


def _candidate_array(name: str, value: Any, key_count: int) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise FlatBooleanKvWgpuParityError(f"{name} must be an array")
    result: list[int] = []
    previous: int | None = None
    for index, item in enumerate(value):
        candidate = _u32(f"{name}[{index}]", item)
        if candidate >= key_count:
            raise FlatBooleanKvWgpuParityError(
                f"{name}[{index}]={candidate} is outside key_count={key_count}"
            )
        if previous is not None and candidate <= previous:
            raise FlatBooleanKvWgpuParityError(
                f"{name} must be strictly increasing and duplicate-free"
            )
        result.append(candidate)
        previous = candidate
    return tuple(result)


def _validate_tail_bits(signature_bits: int, words_per_signature: int, words: tuple[int, ...]) -> None:
    u64_words = words_per_signature // 2
    tail_bits = signature_bits % 64
    if tail_bits == 0:
        return
    valid_mask = (1 << tail_bits) - 1
    for signature_start in range(0, len(words), words_per_signature):
        last = signature_start + (u64_words - 1) * 2
        final_u64 = words[last] | (words[last + 1] << 32)
        if final_u64 & ~valid_mask:
            raise FlatBooleanKvWgpuParityError(
                "unused high signature tail bits must be zero"
            )


@dataclass(frozen=True, slots=True)
class FlatBooleanKvWgpuParityV1:
    """Validated FLAT WGPU/CPU candidate parity record."""

    canonical_bytes: bytes
    parity_sha256: str
    signature_bits: int
    key_count: int
    words_per_signature: int
    max_distance: int
    cpu_admitted_blocks: tuple[int, ...]
    wgpu_admitted_blocks: tuple[int, ...]
    exact_candidate_set_match: bool

    @classmethod
    def from_canonical_json_bytes(cls, payload: bytes) -> "FlatBooleanKvWgpuParityV1":
        if not isinstance(payload, bytes) or not payload:
            raise FlatBooleanKvWgpuParityError("parity payload must be non-empty bytes")
        try:
            raw = json.loads(
                payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FlatBooleanKvWgpuParityError("parity payload must be UTF-8 JSON") from exc
        if not isinstance(raw, dict) or tuple(raw) != _TOP_FIELDS:
            raise FlatBooleanKvWgpuParityError(
                "parity fields/order do not match the FLAT schema"
            )
        if raw["schema"] != FLAT_BOOLEAN_KV_WGPU_PARITY_SCHEMA:
            raise FlatBooleanKvWgpuParityError("unsupported WGPU parity schema")

        checksum = _checksum_value(raw["parity_checksum"])
        without_checksum = {
            key: value for key, value in raw.items() if key != "parity_checksum"
        }
        prefix = _compact_json(without_checksum)[:-1]
        if checksum != _fnv1a64(prefix):
            raise FlatBooleanKvWgpuParityError(
                "parity_checksum does not match the canonical producer prefix"
            )
        canonical = (
            prefix
            + b',"parity_checksum":{"algorithm":"fnv1a64","value":"'
            + checksum.encode("ascii")
            + b'"}}'
        )
        if canonical != payload:
            raise FlatBooleanKvWgpuParityError(
                "parity payload is not the canonical compact producer encoding"
            )

        signature_bits = _u64("signature_bits", raw["signature_bits"])
        if signature_bits == 0:
            raise FlatBooleanKvWgpuParityError("signature_bits must be non-zero")
        key_count = _u32("key_count", raw["key_count"])
        if key_count == 0:
            raise FlatBooleanKvWgpuParityError("key_count must be non-zero")
        words_per_signature = _u32(
            "words_per_signature", raw["words_per_signature"]
        )
        if words_per_signature == 0 or words_per_signature % 2:
            raise FlatBooleanKvWgpuParityError(
                "words_per_signature must be a non-zero even u32 count"
            )
        expected_words = 2 * ((signature_bits + 63) // 64)
        if words_per_signature != expected_words:
            raise FlatBooleanKvWgpuParityError(
                "words_per_signature does not match signature_bits geometry"
            )
        max_distance = _u32("max_distance", raw["max_distance"])
        if max_distance > signature_bits:
            raise FlatBooleanKvWgpuParityError(
                "max_distance cannot exceed signature_bits"
            )
        query_words = _u32_array(
            "query_words_u32", raw["query_words_u32"], words_per_signature
        )
        key_words = _u32_array(
            "key_words_u32",
            raw["key_words_u32"],
            key_count * words_per_signature,
        )
        _validate_tail_bits(signature_bits, words_per_signature, query_words)
        _validate_tail_bits(signature_bits, words_per_signature, key_words)

        cpu = _candidate_array(
            "cpu_admitted_blocks", raw["cpu_admitted_blocks"], key_count
        )
        wgpu = _candidate_array(
            "wgpu_admitted_blocks", raw["wgpu_admitted_blocks"], key_count
        )
        declared_match = raw["exact_candidate_set_match"]
        if not isinstance(declared_match, bool):
            raise FlatBooleanKvWgpuParityError(
                "exact_candidate_set_match must be a JSON boolean"
            )
        actual_match = cpu == wgpu
        if declared_match != actual_match:
            raise FlatBooleanKvWgpuParityError(
                "exact_candidate_set_match disagrees with recomputed candidate sets"
            )

        return cls(
            canonical_bytes=payload,
            parity_sha256=hashlib.sha256(payload).hexdigest(),
            signature_bits=signature_bits,
            key_count=key_count,
            words_per_signature=words_per_signature,
            max_distance=max_distance,
            cpu_admitted_blocks=cpu,
            wgpu_admitted_blocks=wgpu,
            exact_candidate_set_match=actual_match,
        )

    def require_exact_match(self) -> None:
        """Reject negative parity before any CPU-vs-WGPU performance comparison."""
        if not self.exact_candidate_set_match:
            raise FlatBooleanKvWgpuParityError(
                "CPU/WGPU candidate-set mismatch blocks performance comparison"
            )
