import json
import unittest

from kvlab.flat_boolean_kv_wgpu_parity import (
    FLAT_BOOLEAN_KV_WGPU_PARITY_REFERENCE_REVISION,
    FlatBooleanKvWgpuParityError,
    FlatBooleanKvWgpuParityV1,
)

POSITIVE = b'{"schema":"flat.boolean-kv-wgpu-parity.v1","execution":{"source_revision":"4a9a7cb065b067ecbdfecd3f03041e51971e856b","wgpu_runtime":"wgpu/30.0.1","adapter_name":"fixture-adapter","backend":"Vulkan","driver":"fixture-driver","driver_info":"fixture-info","vendor":4660,"device":22136},"signature_bits":65,"key_count":3,"words_per_signature":4,"max_distance":1,"query_words_u32":[11,0,1,0],"key_words_u32":[11,0,1,0,3,0,1,0,0,0,0,0],"cpu_admitted_blocks":[0,1],"wgpu_admitted_blocks":[0,1],"exact_candidate_set_match":true,"parity_checksum":{"algorithm":"fnv1a64","value":"4f398c5d2b62ee43"}}'
NEGATIVE = b'{"schema":"flat.boolean-kv-wgpu-parity.v1","execution":{"source_revision":"4a9a7cb065b067ecbdfecd3f03041e51971e856b","wgpu_runtime":"wgpu/30.0.1","adapter_name":"fixture-adapter","backend":"Vulkan","driver":"fixture-driver","driver_info":"fixture-info","vendor":4660,"device":22136},"signature_bits":65,"key_count":3,"words_per_signature":4,"max_distance":1,"query_words_u32":[11,0,1,0],"key_words_u32":[11,0,1,0,3,0,1,0,0,0,0,0],"cpu_admitted_blocks":[0,1],"wgpu_admitted_blocks":[0,2],"exact_candidate_set_match":false,"parity_checksum":{"algorithm":"fnv1a64","value":"849db7aca171c975"}}'

def canonical_payload(raw: dict) -> bytes:
    checksum_record = raw.setdefault(
        "parity_checksum", {"algorithm": "fnv1a64", "value": "0" * 16}
    )
    checksum_record["algorithm"] = "fnv1a64"
    prefix = json.dumps(
        {key: value for key, value in raw.items() if key != "parity_checksum"},
        separators=(",", ":"),
    ).encode()[:-1]
    checksum = 0xCBF29CE484222325
    for byte in prefix:
        checksum ^= byte
        checksum = (checksum * 0x100000001B3) & ((1 << 64) - 1)
    checksum_record["value"] = f"{checksum:016x}"
    return json.dumps(raw, separators=(",", ":")).encode()



class FlatBooleanKvWgpuParityTests(unittest.TestCase):
    def test_positive_reference_round_trips_and_allows_match_gate(self) -> None:
        evidence = FlatBooleanKvWgpuParityV1.from_canonical_json_bytes(POSITIVE)
        self.assertEqual(evidence.canonical_bytes, POSITIVE)
        self.assertEqual(evidence.cpu_admitted_blocks, (0, 1))
        self.assertEqual(evidence.wgpu_admitted_blocks, (0, 1))
        self.assertTrue(evidence.exact_candidate_set_match)
        self.assertEqual(len(evidence.parity_sha256), 64)
        self.assertEqual(evidence.source_revision, FLAT_BOOLEAN_KV_WGPU_PARITY_REFERENCE_REVISION)
        self.assertEqual(evidence.wgpu_runtime, "wgpu/30.0.1")
        self.assertEqual(evidence.adapter_name, "fixture-adapter")
        self.assertEqual(evidence.backend, "Vulkan")
        self.assertEqual(evidence.driver, "fixture-driver")
        self.assertEqual(evidence.vendor, 4660)
        self.assertEqual(evidence.device, 22136)
        evidence.require_exact_match()
        self.assertEqual(
            FLAT_BOOLEAN_KV_WGPU_PARITY_REFERENCE_REVISION,
            "4a9a7cb065b067ecbdfecd3f03041e51971e856b",
        )

    def test_negative_candidate_mismatch_is_retained_but_blocks_timing(self) -> None:
        evidence = FlatBooleanKvWgpuParityV1.from_canonical_json_bytes(NEGATIVE)
        self.assertFalse(evidence.exact_candidate_set_match)
        self.assertEqual(evidence.cpu_admitted_blocks, (0, 1))
        self.assertEqual(evidence.wgpu_admitted_blocks, (0, 2))
        with self.assertRaisesRegex(FlatBooleanKvWgpuParityError, "blocks performance"):
            evidence.require_exact_match()

    def test_declared_match_cannot_disagree_with_candidate_sets(self) -> None:
        raw = json.loads(NEGATIVE)
        raw["exact_candidate_set_match"] = True
        payload = canonical_payload(raw)
        with self.assertRaisesRegex(FlatBooleanKvWgpuParityError, "disagrees"):
            FlatBooleanKvWgpuParityV1.from_canonical_json_bytes(payload)

    def test_execution_provenance_is_required_and_strict(self) -> None:
        raw = json.loads(POSITIVE)
        raw["execution"]["source_revision"] = "0" * 40
        payload = canonical_payload(raw)
        with self.assertRaisesRegex(FlatBooleanKvWgpuParityError, "qualified FLAT reference"):
            FlatBooleanKvWgpuParityV1.from_canonical_json_bytes(payload)

        raw = json.loads(POSITIVE)
        raw["execution"]["wgpu_runtime"] = ""
        payload = canonical_payload(raw)
        with self.assertRaisesRegex(FlatBooleanKvWgpuParityError, "wgpu_runtime"):
            FlatBooleanKvWgpuParityV1.from_canonical_json_bytes(payload)

    def test_noncanonical_reordered_payload_is_rejected(self) -> None:
        raw = json.loads(POSITIVE)
        payload = json.dumps(raw, indent=2).encode()
        with self.assertRaises(FlatBooleanKvWgpuParityError):
            FlatBooleanKvWgpuParityV1.from_canonical_json_bytes(payload)

    def test_tail_bits_and_candidate_range_fail_closed(self) -> None:
        raw = json.loads(POSITIVE)
        raw["query_words_u32"][-1] = 2
        payload = canonical_payload(raw)
        with self.assertRaisesRegex(FlatBooleanKvWgpuParityError, "tail bits"):
            FlatBooleanKvWgpuParityV1.from_canonical_json_bytes(payload)

        raw = json.loads(POSITIVE)
        raw["wgpu_admitted_blocks"] = [0, 3]
        raw["exact_candidate_set_match"] = False
        payload = canonical_payload(raw)
        with self.assertRaisesRegex(FlatBooleanKvWgpuParityError, "outside key_count"):
            FlatBooleanKvWgpuParityV1.from_canonical_json_bytes(payload)


if __name__ == "__main__":
    unittest.main()
