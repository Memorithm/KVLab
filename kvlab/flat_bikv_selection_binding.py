"""Fail-closed retention for FLAT BKV-K6 selection-binding evidence.

FLAT owns Boolean-KV routing and qualification production. KVLab owns retained
experimental evidence. This module accepts the canonical
``flat.bikv-selection-binding.v1`` envelope introduced by FLAT-ATTENTION #251,
revalidates the selection/accounting relation exposed by that envelope, verifies
all producer FNV-1a corruption checks that are visible at this layer, and derives
SHA-256 identities over the exact retained bytes.

FNV-1a is not cryptographic attestation. Logical avoided numerical K/V bytes are
not physical DRAM/HBM traffic and this consumer makes no performance claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

FLAT_BIKV_SELECTION_BINDING_SCHEMA = "flat.bikv-selection-binding.v1"
FLAT_BOOLEAN_KV_SELECTION_SCHEMA_V2 = "flat.boolean-kv-selection.v2"
FLAT_BIKV_SELECTION_BINDING_REFERENCE_REVISION = (
    "5f474c51bff4b782b3a124375cb0f4b651b4b708"
)

_U64_MAX = (1 << 64) - 1
_BINDING_FIELDS = ("schema", "selection", "qualification", "binding_checksum")
_SELECTION_FIELDS = (
    "schema",
    "generation",
    "signature_bits",
    "max_distance",
    "live_tokens",
    "mapped_pages",
    "boolean_pages_scanned",
    "boolean_key_bytes_read",
    "numerical_kv_bytes_per_token",
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
_QUALIFICATION_FIELDS = (
    "schema_version",
    "candidate",
    "dense_baseline",
    "selection",
    "scope",
    "accounting",
    "phase_medians_ns",
    "gates",
    "promotion_decision",
    "evidence_checksum",
)
_QUALIFICATION_SELECTION_FIELDS = ("signature_bits", "max_distance", "policy")
_ACCOUNTING_FIELDS = (
    "live_tokens",
    "selected_live_tokens",
    "mapped_pages",
    "selected_pages",
    "page_size",
    "kv_heads",
    "head_dim",
    "scalar_bytes",
    "boolean_index_bytes_read",
    "kv_bytes_per_token",
    "dense_numerical_kv_bytes",
    "selected_numerical_kv_bytes",
    "avoided_numerical_kv_bytes",
)
_GATE_FIELDS = (
    "all_accept_k6_vs_m16",
    "sparse_k6_vs_restricted_oracle",
    "correctness_gate_passed",
    "quality_gate_passed",
)
_CHECKSUM_FIELDS = ("algorithm", "value")


class FlatBikvSelectionBindingError(ValueError):
    """Raised when retained FLAT selection-binding evidence is invalid."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FlatBikvSelectionBindingError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _u64(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FlatBikvSelectionBindingError(f"{name} must be an integer u64")
    if not 0 <= value <= _U64_MAX:
        raise FlatBikvSelectionBindingError(f"{name} must be within the u64 range")
    return value


def _boolean(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise FlatBikvSelectionBindingError(f"{name} must be a JSON boolean")
    return value


def _fnv1a64(payload: bytes) -> str:
    value = 0xCBF29CE484222325
    for byte in payload:
        value ^= byte
        value = (value * 0x100000001B3) & _U64_MAX
    return f"{value:016x}"


def _compact_json(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _checksum_value(name: str, value: Any) -> str:
    if not isinstance(value, dict) or tuple(value) != _CHECKSUM_FIELDS:
        raise FlatBikvSelectionBindingError(
            f"{name} checksum fields/order do not match schema"
        )
    if value["algorithm"] != "fnv1a64":
        raise FlatBikvSelectionBindingError(f"{name} checksum algorithm must be fnv1a64")
    checksum = value["value"]
    if (
        not isinstance(checksum, str)
        or len(checksum) != 16
        or checksum != checksum.lower()
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        raise FlatBikvSelectionBindingError(
            f"{name} checksum must be 16 lowercase hexadecimal digits"
        )
    return checksum


def _verify_final_object_checksum(name: str, value: dict[str, Any]) -> None:
    checksum = _checksum_value(name, value["evidence_checksum"])
    without_checksum = {
        key: item for key, item in value.items() if key != "evidence_checksum"
    }
    prefix = _compact_json(without_checksum)[:-1]
    if checksum != _fnv1a64(prefix):
        raise FlatBikvSelectionBindingError(
            f"{name} checksum does not match the canonical producer prefix"
        )


def _validate_selection(selection: Any) -> None:
    if not isinstance(selection, dict) or tuple(selection) != _SELECTION_FIELDS:
        raise FlatBikvSelectionBindingError(
            "selection fields/order do not match the FLAT v2 schema"
        )
    if selection["schema"] != FLAT_BOOLEAN_KV_SELECTION_SCHEMA_V2:
        raise FlatBikvSelectionBindingError(
            "selection schema must be flat.boolean-kv-selection.v2"
        )

    _u64("selection.generation", selection["generation"])
    signature_bits = _u64("selection.signature_bits", selection["signature_bits"])
    max_distance = _u64("selection.max_distance", selection["max_distance"])
    live_tokens = _u64("selection.live_tokens", selection["live_tokens"])
    mapped_pages = _u64("selection.mapped_pages", selection["mapped_pages"])
    scanned_pages = _u64(
        "selection.boolean_pages_scanned", selection["boolean_pages_scanned"]
    )
    boolean_bytes = _u64(
        "selection.boolean_key_bytes_read", selection["boolean_key_bytes_read"]
    )
    bytes_per_token = _u64(
        "selection.numerical_kv_bytes_per_token",
        selection["numerical_kv_bytes_per_token"],
    )
    full_bytes = _u64(
        "selection.full_numerical_kv_bytes", selection["full_numerical_kv_bytes"]
    )
    selected_bytes = _u64(
        "selection.selected_numerical_kv_bytes",
        selection["selected_numerical_kv_bytes"],
    )
    avoided_bytes = _u64(
        "selection.avoided_numerical_kv_bytes",
        selection["avoided_numerical_kv_bytes"],
    )

    if signature_bits == 0 or max_distance > signature_bits:
        raise FlatBikvSelectionBindingError(
            "selection signature/max-distance geometry is invalid"
        )
    if scanned_pages != mapped_pages:
        raise FlatBikvSelectionBindingError(
            "selection boolean_pages_scanned must equal mapped_pages"
        )
    expected_boolean_bytes = mapped_pages * ((signature_bits + 63) // 64) * 8
    if boolean_bytes != expected_boolean_bytes:
        raise FlatBikvSelectionBindingError("selection Boolean byte accounting mismatch")
    if (live_tokens == 0) != (mapped_pages == 0):
        raise FlatBikvSelectionBindingError(
            "selection live-token/page cardinality mismatch"
        )

    pages = selection["selected_pages"]
    if not isinstance(pages, list) or len(pages) > mapped_pages:
        raise FlatBikvSelectionBindingError("selection selected_pages count is invalid")
    previous_logical: int | None = None
    physical_pages: set[int] = set()
    selected_live_tokens = 0
    for page in pages:
        if not isinstance(page, dict) or tuple(page) != _PAGE_FIELDS:
            raise FlatBikvSelectionBindingError(
                "selection page fields/order do not match schema"
            )
        logical = _u64("selection.page.logical_page", page["logical_page"])
        physical = _u64("selection.page.physical_page", page["physical_page"])
        page_live = _u64("selection.page.live_tokens", page["live_tokens"])
        hamming = _u64(
            "selection.page.hamming_distance", page["hamming_distance"]
        )
        xnor = _u64("selection.page.xnor_matches", page["xnor_matches"])
        if logical >= mapped_pages or page_live == 0:
            raise FlatBikvSelectionBindingError(
                "selection page range/live-token invariant failed"
            )
        if previous_logical is not None and logical <= previous_logical:
            raise FlatBikvSelectionBindingError(
                "selection logical pages must be strictly increasing"
            )
        if physical in physical_pages:
            raise FlatBikvSelectionBindingError(
                "selection physical pages must be unique"
            )
        if hamming + xnor != signature_bits or hamming > max_distance:
            raise FlatBikvSelectionBindingError(
                "selection Hamming/XNOR threshold invariant failed"
            )
        previous_logical = logical
        physical_pages.add(physical)
        selected_live_tokens += page_live
        if selected_live_tokens > _U64_MAX:
            raise FlatBikvSelectionBindingError(
                "selection selected live-token sum exceeds u64"
            )

    if selected_live_tokens > live_tokens:
        raise FlatBikvSelectionBindingError(
            "selection selected live tokens exceed total"
        )
    if live_tokens == 0:
        if (
            bytes_per_token == 0
            or full_bytes
            or selected_bytes
            or avoided_bytes
            or pages
        ):
            raise FlatBikvSelectionBindingError("empty selection accounting mismatch")
    else:
        if bytes_per_token == 0:
            raise FlatBikvSelectionBindingError(
                "selection numerical bytes/token must be non-zero"
            )
        if full_bytes != live_tokens * bytes_per_token:
            raise FlatBikvSelectionBindingError(
                "selection full numerical byte accounting mismatch"
            )
        if selected_bytes != selected_live_tokens * bytes_per_token:
            raise FlatBikvSelectionBindingError(
                "selection selected numerical byte accounting mismatch"
            )
        if avoided_bytes != full_bytes - selected_bytes:
            raise FlatBikvSelectionBindingError(
                "selection avoided numerical byte accounting mismatch"
            )
    _verify_final_object_checksum("selection", selection)


def _validate_qualification(qualification: Any) -> None:
    if (
        not isinstance(qualification, dict)
        or tuple(qualification) != _QUALIFICATION_FIELDS
    ):
        raise FlatBikvSelectionBindingError(
            "qualification fields/order do not match FLAT BKV-K6 schema"
        )
    if qualification["schema_version"] != 1:
        raise FlatBikvSelectionBindingError("qualification schema_version must be 1")
    for manifest_name in ("candidate", "dense_baseline"):
        manifest = qualification[manifest_name]
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise FlatBikvSelectionBindingError(
                f"qualification.{manifest_name} must be a benchmark manifest v1"
            )
        commit_sha = manifest.get("commit_sha")
        if (
            not isinstance(commit_sha, str)
            or len(commit_sha) != 40
            or any(character not in "0123456789abcdefABCDEF" for character in commit_sha)
        ):
            raise FlatBikvSelectionBindingError(
                f"qualification.{manifest_name}.commit_sha must be 40 hexadecimal digits"
            )
    if (
        qualification["candidate"]["commit_sha"]
        != qualification["dense_baseline"]["commit_sha"]
    ):
        raise FlatBikvSelectionBindingError(
            "qualification candidate/dense commit SHA mismatch"
        )

    q_selection = qualification["selection"]
    if (
        not isinstance(q_selection, dict)
        or tuple(q_selection) != _QUALIFICATION_SELECTION_FIELDS
    ):
        raise FlatBikvSelectionBindingError(
            "qualification selection fields/order do not match schema"
        )
    signature_bits = _u64(
        "qualification.selection.signature_bits", q_selection["signature_bits"]
    )
    max_distance = _u64(
        "qualification.selection.max_distance", q_selection["max_distance"]
    )
    if signature_bits == 0 or max_distance > signature_bits:
        raise FlatBikvSelectionBindingError(
            "qualification selection geometry is invalid"
        )
    if not isinstance(q_selection["policy"], str) or not q_selection["policy"].strip():
        raise FlatBikvSelectionBindingError(
            "qualification selection policy must be non-empty"
        )

    accounting = qualification["accounting"]
    if not isinstance(accounting, dict) or tuple(accounting) != _ACCOUNTING_FIELDS:
        raise FlatBikvSelectionBindingError(
            "qualification accounting fields/order do not match schema"
        )
    values = {
        field: _u64(f"qualification.accounting.{field}", accounting[field])
        for field in _ACCOUNTING_FIELDS
    }
    for field in ("page_size", "kv_heads", "head_dim", "scalar_bytes"):
        if values[field] == 0:
            raise FlatBikvSelectionBindingError(
                f"qualification.accounting.{field} must be non-zero"
            )
    expected_bytes_per_token = (
        2 * values["kv_heads"] * values["head_dim"] * values["scalar_bytes"]
    )
    if values["kv_bytes_per_token"] != expected_bytes_per_token:
        raise FlatBikvSelectionBindingError(
            "qualification K+V bytes/token accounting mismatch"
        )
    if values["selected_live_tokens"] > values["live_tokens"]:
        raise FlatBikvSelectionBindingError(
            "qualification selected live tokens exceed total"
        )
    if values["selected_pages"] > values["mapped_pages"]:
        raise FlatBikvSelectionBindingError(
            "qualification selected pages exceed mapped pages"
        )
    if (
        values["dense_numerical_kv_bytes"]
        != values["live_tokens"] * expected_bytes_per_token
    ):
        raise FlatBikvSelectionBindingError(
            "qualification dense numerical byte accounting mismatch"
        )
    if (
        values["selected_numerical_kv_bytes"]
        != values["selected_live_tokens"] * expected_bytes_per_token
    ):
        raise FlatBikvSelectionBindingError(
            "qualification selected numerical byte accounting mismatch"
        )
    if values["avoided_numerical_kv_bytes"] != (
        values["dense_numerical_kv_bytes"] - values["selected_numerical_kv_bytes"]
    ):
        raise FlatBikvSelectionBindingError(
            "qualification avoided numerical byte accounting mismatch"
        )

    gates = qualification["gates"]
    if not isinstance(gates, dict) or tuple(gates) != _GATE_FIELDS:
        raise FlatBikvSelectionBindingError(
            "qualification gates fields/order do not match schema"
        )
    all_accept = _boolean(
        "qualification.gates.all_accept_k6_vs_m16", gates["all_accept_k6_vs_m16"]
    )
    sparse = _boolean(
        "qualification.gates.sparse_k6_vs_restricted_oracle",
        gates["sparse_k6_vs_restricted_oracle"],
    )
    correctness = _boolean(
        "qualification.gates.correctness_gate_passed",
        gates["correctness_gate_passed"],
    )
    _boolean(
        "qualification.gates.quality_gate_passed", gates["quality_gate_passed"]
    )
    if correctness != (all_accept and sparse):
        raise FlatBikvSelectionBindingError(
            "qualification correctness gate is inconsistent"
        )
    _verify_final_object_checksum("qualification", qualification)


def _validate_cross_accounting(
    selection: dict[str, Any], qualification: dict[str, Any]
) -> None:
    q_selection = qualification["selection"]
    accounting = qualification["accounting"]
    selected_live_tokens = sum(
        page["live_tokens"] for page in selection["selected_pages"]
    )
    pairs = (
        ("signature_bits", selection["signature_bits"], q_selection["signature_bits"]),
        ("max_distance", selection["max_distance"], q_selection["max_distance"]),
        ("live_tokens", selection["live_tokens"], accounting["live_tokens"]),
        ("mapped_pages", selection["mapped_pages"], accounting["mapped_pages"]),
        ("selected_pages", len(selection["selected_pages"]), accounting["selected_pages"]),
        ("selected_live_tokens", selected_live_tokens, accounting["selected_live_tokens"]),
        (
            "boolean_index_bytes_read",
            selection["boolean_key_bytes_read"],
            accounting["boolean_index_bytes_read"],
        ),
        (
            "kv_bytes_per_token",
            selection["numerical_kv_bytes_per_token"],
            accounting["kv_bytes_per_token"],
        ),
        (
            "dense_numerical_kv_bytes",
            selection["full_numerical_kv_bytes"],
            accounting["dense_numerical_kv_bytes"],
        ),
        (
            "selected_numerical_kv_bytes",
            selection["selected_numerical_kv_bytes"],
            accounting["selected_numerical_kv_bytes"],
        ),
        (
            "avoided_numerical_kv_bytes",
            selection["avoided_numerical_kv_bytes"],
            accounting["avoided_numerical_kv_bytes"],
        ),
    )
    for field, selected, qualified in pairs:
        if selected != qualified:
            raise FlatBikvSelectionBindingError(
                f"selection/qualification accounting mismatch: {field}"
            )


@dataclass(frozen=True, slots=True)
class FlatBikvSelectionBindingV1:
    """Validated retained BKV-K6 binding with exact-byte identities."""

    canonical_bytes: bytes
    selection_sha256: str
    qualification_sha256: str
    binding_sha256: str

    @classmethod
    def from_canonical_json_bytes(
        cls, payload: bytes
    ) -> "FlatBikvSelectionBindingV1":
        if not isinstance(payload, bytes) or not payload:
            raise FlatBikvSelectionBindingError(
                "binding payload must be non-empty bytes"
            )
        try:
            raw = json.loads(
                payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FlatBikvSelectionBindingError(
                "binding payload must be UTF-8 JSON"
            ) from exc
        if not isinstance(raw, dict) or tuple(raw) != _BINDING_FIELDS:
            raise FlatBikvSelectionBindingError(
                "binding fields/order do not match FLAT schema"
            )
        if raw["schema"] != FLAT_BIKV_SELECTION_BINDING_SCHEMA:
            raise FlatBikvSelectionBindingError("unsupported binding schema")

        _validate_selection(raw["selection"])
        _validate_qualification(raw["qualification"])
        _validate_cross_accounting(raw["selection"], raw["qualification"])
        checksum = _checksum_value("binding", raw["binding_checksum"])

        selection_bytes = _compact_json(raw["selection"])
        qualification_bytes = _compact_json(raw["qualification"])
        prefix = (
            b'{"schema":"'
            + FLAT_BIKV_SELECTION_BINDING_SCHEMA.encode("ascii")
            + b'","selection":'
            + selection_bytes
            + b',"qualification":'
            + qualification_bytes
        )
        if checksum != _fnv1a64(prefix):
            raise FlatBikvSelectionBindingError(
                "binding checksum does not match canonical producer prefix"
            )
        suffix = (
            b',"binding_checksum":{"algorithm":"fnv1a64","value":"'
            + checksum.encode("ascii")
            + b'"}}'
        )
        canonical = prefix + suffix
        if canonical != payload:
            raise FlatBikvSelectionBindingError(
                "binding payload is not the canonical compact producer encoding"
            )

        return cls(
            canonical_bytes=payload,
            selection_sha256=hashlib.sha256(selection_bytes).hexdigest(),
            qualification_sha256=hashlib.sha256(qualification_bytes).hexdigest(),
            binding_sha256=hashlib.sha256(payload).hexdigest(),
        )
