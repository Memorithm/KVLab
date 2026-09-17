"""Fail-closed retention for FLAT BKV-K6 target-page quality evidence.

FLAT owns Boolean-KV page routing and the canonical producer encoding. KVLab
owns experimental protocol, target construction, retention and interpretation.
This module accepts the candidate ``flat.bikv-selection-quality.v1`` envelope,
revalidates the embedded FLAT selection, recomputes every set-quality metric,
requires canonical producer bytes and derives SHA-256 identities for retention.

The target-page recall/FNR record is not downstream model quality and carries no
performance, physical-traffic, energy or promotion claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .flat_bikv_selection_binding import (
    FlatBikvSelectionBindingError,
    _validate_selection,
)

FLAT_BIKV_SELECTION_QUALITY_SCHEMA = "flat.bikv-selection-quality.v1"
# Candidate producer head for FLAT-ATTENTION #252. This is deliberately not
# called a final reference revision until the producer PR is qualified/merged.
FLAT_BIKV_SELECTION_QUALITY_CANDIDATE_REVISION = (
    "c31284780e21a67b5a47f469d4493a17e396259d"
)

_U64_MAX = (1 << 64) - 1
_TOP_FIELDS = (
    "schema",
    "selection",
    "declared_dense_target_pages",
    "metrics",
)
_METRIC_FIELDS = (
    "mapped_pages",
    "selected_pages",
    "target_pages",
    "true_positive_pages",
    "false_negative_pages",
    "false_positive_pages",
    "recall",
    "false_negative_rate",
    "candidate_density",
)
_FRACTION_FIELDS = ("numerator", "denominator")


class FlatBikvSelectionQualityError(ValueError):
    """Raised when FLAT target-page quality evidence is invalid/non-canonical."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FlatBikvSelectionQualityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _u64(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FlatBikvSelectionQualityError(f"{name} must be an integer u64")
    if not 0 <= value <= _U64_MAX:
        raise FlatBikvSelectionQualityError(f"{name} must be within the u64 range")
    return value


def _compact_json(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _fraction(name: str, value: Any) -> tuple[int, int]:
    if not isinstance(value, dict) or tuple(value) != _FRACTION_FIELDS:
        raise FlatBikvSelectionQualityError(
            f"{name} fields/order do not match numerator/denominator schema"
        )
    numerator = _u64(f"{name}.numerator", value["numerator"])
    denominator = _u64(f"{name}.denominator", value["denominator"])
    if denominator == 0:
        raise FlatBikvSelectionQualityError(f"{name}.denominator must be non-zero")
    return numerator, denominator


def _validate_target(mapped_pages: int, target: Any) -> tuple[int, ...]:
    if not isinstance(target, list) or not target:
        raise FlatBikvSelectionQualityError(
            "declared_dense_target_pages must be a non-empty JSON array"
        )
    pages: list[int] = []
    previous: int | None = None
    for index, value in enumerate(target):
        logical_page = _u64(f"declared_dense_target_pages[{index}]", value)
        if logical_page >= mapped_pages:
            raise FlatBikvSelectionQualityError(
                f"dense-target logical page {logical_page} is outside {mapped_pages} mapped pages"
            )
        if previous is not None and logical_page <= previous:
            raise FlatBikvSelectionQualityError(
                "declared dense-target pages must be strictly increasing and duplicate-free"
            )
        pages.append(logical_page)
        previous = logical_page
    return tuple(pages)


def _intersection_count(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    left_index = 0
    right_index = 0
    count = 0
    while left_index < len(left) and right_index < len(right):
        if left[left_index] < right[right_index]:
            left_index += 1
        elif left[left_index] > right[right_index]:
            right_index += 1
        else:
            count += 1
            left_index += 1
            right_index += 1
    return count


@dataclass(frozen=True, slots=True)
class FlatBikvSelectionQualityV1:
    """Validated target-page quality record with exact-byte content identity."""

    canonical_bytes: bytes
    quality_sha256: str
    selection_sha256: str
    target_pages: tuple[int, ...]
    true_positive_pages: int
    false_negative_pages: int
    false_positive_pages: int

    @classmethod
    def from_canonical_json_bytes(
        cls, payload: bytes
    ) -> "FlatBikvSelectionQualityV1":
        if not isinstance(payload, bytes) or not payload:
            raise FlatBikvSelectionQualityError(
                "quality payload must be non-empty bytes"
            )
        try:
            raw = json.loads(
                payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FlatBikvSelectionQualityError(
                "quality payload must be UTF-8 JSON"
            ) from exc
        if not isinstance(raw, dict) or tuple(raw) != _TOP_FIELDS:
            raise FlatBikvSelectionQualityError(
                "quality fields/order do not match FLAT schema"
            )
        if raw["schema"] != FLAT_BIKV_SELECTION_QUALITY_SCHEMA:
            raise FlatBikvSelectionQualityError("unsupported quality schema")

        selection = raw["selection"]
        try:
            _validate_selection(selection)
        except FlatBikvSelectionBindingError as exc:
            raise FlatBikvSelectionQualityError(
                f"embedded selection is invalid: {exc}"
            ) from exc

        mapped_pages = _u64("selection.mapped_pages", selection["mapped_pages"])
        if mapped_pages == 0:
            raise FlatBikvSelectionQualityError(
                "selection quality requires at least one mapped page"
            )
        target_pages = _validate_target(mapped_pages, raw["declared_dense_target_pages"])
        selected_pages = tuple(
            _u64("selection.page.logical_page", page["logical_page"])
            for page in selection["selected_pages"]
        )

        true_positive = _intersection_count(selected_pages, target_pages)
        false_negative = len(target_pages) - true_positive
        false_positive = len(selected_pages) - true_positive

        metrics = raw["metrics"]
        if not isinstance(metrics, dict) or tuple(metrics) != _METRIC_FIELDS:
            raise FlatBikvSelectionQualityError(
                "quality metrics fields/order do not match FLAT schema"
            )
        expected_scalars = {
            "mapped_pages": mapped_pages,
            "selected_pages": len(selected_pages),
            "target_pages": len(target_pages),
            "true_positive_pages": true_positive,
            "false_negative_pages": false_negative,
            "false_positive_pages": false_positive,
        }
        for name, expected in expected_scalars.items():
            observed = _u64(f"metrics.{name}", metrics[name])
            if observed != expected:
                raise FlatBikvSelectionQualityError(
                    f"metrics.{name} does not match recomputed selection/target value"
                )

        expected_fractions = {
            "recall": (true_positive, len(target_pages)),
            "false_negative_rate": (false_negative, len(target_pages)),
            "candidate_density": (len(selected_pages), mapped_pages),
        }
        for name, expected in expected_fractions.items():
            if _fraction(f"metrics.{name}", metrics[name]) != expected:
                raise FlatBikvSelectionQualityError(
                    f"metrics.{name} does not match recomputed exact fraction"
                )

        canonical = _compact_json(raw)
        if canonical != payload:
            raise FlatBikvSelectionQualityError(
                "quality payload is not the canonical compact producer encoding"
            )
        selection_bytes = _compact_json(selection)
        return cls(
            canonical_bytes=payload,
            quality_sha256=hashlib.sha256(payload).hexdigest(),
            selection_sha256=hashlib.sha256(selection_bytes).hexdigest(),
            target_pages=target_pages,
            true_positive_pages=true_positive,
            false_negative_pages=false_negative,
            false_positive_pages=false_positive,
        )
