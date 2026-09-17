import hashlib
import json
import unittest

from kvlab.flat_bikv_selection_binding import (
    FLAT_BIKV_SELECTION_BINDING_REFERENCE_REVISION,
    FlatBikvSelectionBindingError,
    FlatBikvSelectionBindingV1,
)


def _compact(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _fnv(payload):
    value = 0xCBF29CE484222325
    for byte in payload:
        value ^= byte
        value = (value * 0x100000001B3) & ((1 << 64) - 1)
    return f"{value:016x}"


def _with_checksum(value):
    prefix = _compact(value)[:-1]
    result = dict(value)
    result["evidence_checksum"] = {
        "algorithm": "fnv1a64",
        "value": _fnv(prefix),
    }
    return result


def _manifest(benchmark_id, latency):
    result = {
        "median_latency_ns": latency,
        "p95_latency_ns": latency + 10,
        "tokens_per_second_milli": 1000,
    }
    return {
        "schema_version": 1,
        "commit_sha": "a" * 40,
        "benchmark_id": benchmark_id,
        "command": "cargo run --release --example bkv6",
        "environment": {
            "device": "test-device",
            "backend": "Vulkan",
            "driver": "test-driver",
            "os": "linux",
            "arch": "x86_64",
        },
        "problem": {
            "precision": "f32",
            "batch": 1,
            "q_heads": 2,
            "kv_heads": 1,
            "query_len": 1,
            "kv_len": 4,
            "head_dim": 2,
            "causal": True,
        },
        "protocol": {"warmup_iterations": 1, "measured_iterations": 3},
        "result": result,
        "result_checksum": {"algorithm": "fnv1a64", "value": _fnv(_compact(result))},
    }


def _selection(**updates):
    value = {
        "schema": "flat.boolean-kv-selection.v2",
        "generation": 7,
        "signature_bits": 64,
        "max_distance": 8,
        "live_tokens": 4,
        "mapped_pages": 2,
        "boolean_pages_scanned": 2,
        "boolean_key_bytes_read": 16,
        "numerical_kv_bytes_per_token": 8,
        "full_numerical_kv_bytes": 32,
        "selected_numerical_kv_bytes": 16,
        "avoided_numerical_kv_bytes": 16,
        "selected_pages": [
            {
                "logical_page": 0,
                "physical_page": 10,
                "live_tokens": 2,
                "hamming_distance": 1,
                "xnor_matches": 63,
            }
        ],
    }
    value.update(updates)
    return _with_checksum(value)


def _qualification(**selection_updates):
    value = {
        "schema_version": 1,
        "candidate": _manifest("candidate", 100),
        "dense_baseline": _manifest("dense", 100),
        "selection": {
            "signature_bits": 64,
            "max_distance": 8,
            "policy": "hamming-threshold",
        },
        "scope": {
            "timing": "end-to-end-candidate",
            "q_device_resident": True,
            "q_host_mirror_retained": False,
            "kv_device_resident": True,
            "uploads_readbacks_excluded": True,
            "resident_only_production_claim": True,
            "gpu_timestamp_claim": False,
            "physical_dram_traffic_claim": False,
            "model_quality_claim": False,
        },
        "accounting": {
            "live_tokens": 4,
            "selected_live_tokens": 2,
            "mapped_pages": 2,
            "selected_pages": 1,
            "page_size": 2,
            "kv_heads": 1,
            "head_dim": 2,
            "scalar_bytes": 2,
            "boolean_index_bytes_read": 16,
            "kv_bytes_per_token": 8,
            "dense_numerical_kv_bytes": 32,
            "selected_numerical_kv_bytes": 16,
            "avoided_numerical_kv_bytes": 16,
        },
        "phase_medians_ns": {
            "signature_generation": 10,
            "boolean_search": 10,
            "selected_attention": 70,
            "synchronization": 10,
            "dense_attention": 100,
            "diagnostic_sum": 100,
        },
        "gates": {
            "all_accept_k6_vs_m16": True,
            "sparse_k6_vs_restricted_oracle": True,
            "correctness_gate_passed": True,
            "quality_gate_passed": True,
        },
        "promotion_decision": "fallback_no_latency_win",
    }
    value["selection"].update(selection_updates)
    return _with_checksum(value)


def _binding(selection=None, qualification=None):
    selection = selection or _selection()
    qualification = qualification or _qualification()
    prefix = (
        b'{"schema":"flat.bikv-selection-binding.v1","selection":'
        + _compact(selection)
        + b',"qualification":'
        + _compact(qualification)
    )
    checksum = _fnv(prefix)
    return (
        prefix
        + b',"binding_checksum":{"algorithm":"fnv1a64","value":"'
        + checksum.encode("ascii")
        + b'"}}'
    )


class FlatBikvSelectionBindingTests(unittest.TestCase):
    def test_accepts_canonical_binding_and_preserves_exact_bytes(self):
        payload = _binding()
        evidence = FlatBikvSelectionBindingV1.from_canonical_json_bytes(payload)
        self.assertEqual(evidence.canonical_bytes, payload)
        self.assertEqual(evidence.binding_sha256, hashlib.sha256(payload).hexdigest())
        self.assertEqual(len(evidence.selection_sha256), 64)
        self.assertEqual(len(evidence.qualification_sha256), 64)
        self.assertEqual(
            FLAT_BIKV_SELECTION_BINDING_REFERENCE_REVISION,
            "5f474c51bff4b782b3a124375cb0f4b651b4b708",
        )

    def test_rejects_binding_checksum_tamper(self):
        payload = bytearray(_binding())
        marker = b'"binding_checksum":{"algorithm":"fnv1a64","value":"'
        index = payload.index(marker) + len(marker)
        payload[index] = ord("0") if payload[index] != ord("0") else ord("1")
        with self.assertRaisesRegex(FlatBikvSelectionBindingError, "binding checksum"):
            FlatBikvSelectionBindingV1.from_canonical_json_bytes(bytes(payload))

    def test_rejects_selection_and_qualification_threshold_drift(self):
        payload = _binding(selection=_selection(max_distance=7), qualification=_qualification())
        with self.assertRaisesRegex(FlatBikvSelectionBindingError, "max_distance"):
            FlatBikvSelectionBindingV1.from_canonical_json_bytes(payload)

    def test_rejects_boolean_used_for_integer_accounting(self):
        payload = _binding(selection=_selection(generation=True))
        with self.assertRaisesRegex(FlatBikvSelectionBindingError, "integer u64"):
            FlatBikvSelectionBindingV1.from_canonical_json_bytes(payload)

    def test_rejects_duplicate_json_keys(self):
        payload = _binding().replace(
            b'{"schema":"flat.bikv-selection-binding.v1",',
            b'{"schema":"flat.bikv-selection-binding.v1","schema":"flat.bikv-selection-binding.v1",',
            1,
        )
        with self.assertRaisesRegex(FlatBikvSelectionBindingError, "duplicate JSON key"):
            FlatBikvSelectionBindingV1.from_canonical_json_bytes(payload)

    def test_rejects_noncanonical_whitespace_even_when_json_is_valid(self):
        payload = _binding() + b"\n"
        with self.assertRaisesRegex(FlatBikvSelectionBindingError, "canonical"):
            FlatBikvSelectionBindingV1.from_canonical_json_bytes(payload)

    def test_rejects_selection_checksum_tamper(self):
        selection = _selection()
        selection["evidence_checksum"]["value"] = "0" * 16
        payload = _binding(selection=selection)
        with self.assertRaisesRegex(FlatBikvSelectionBindingError, "selection checksum"):
            FlatBikvSelectionBindingV1.from_canonical_json_bytes(payload)

    def test_rejects_qualification_accounting_drift(self):
        qualification = _qualification()
        qualification["accounting"]["selected_live_tokens"] = 3
        qualification = _with_checksum(
            {key: value for key, value in qualification.items() if key != "evidence_checksum"}
        )
        payload = _binding(qualification=qualification)
        with self.assertRaisesRegex(
            FlatBikvSelectionBindingError,
            "selected numerical byte accounting|selected_live_tokens",
        ):
            FlatBikvSelectionBindingV1.from_canonical_json_bytes(payload)


if __name__ == "__main__":
    unittest.main()
