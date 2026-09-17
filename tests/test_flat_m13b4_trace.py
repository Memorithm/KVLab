import json
import unittest

from kvlab.flat_m13b4_trace import (
    FLAT_M13B4_REFERENCE_REVISION,
    FLAT_M13B4_TRACE_SCHEMA,
    FlatM13B4TraceError,
    FlatM13B4TraceV1,
)


FLAT_REFERENCE_FIXTURE = (
    b'{"schema":"flat.m13b4-trace.v1","timing_source":"device_timestamp",'
    b'"scheduling_variant":"serial_matched","scope":"first_decode","events":['
    b'{"kind":"query_representation_ready","timestamp_ns":10},'
    b'{"kind":"q_signature_start","timestamp_ns":11},'
    b'{"kind":"q_signature_end","timestamp_ns":15},'
    b'{"kind":"boolean_routing_start","timestamp_ns":16},'
    b'{"kind":"boolean_routing_end","timestamp_ns":20},'
    b'{"kind":"survivor_metadata_ready","timestamp_ns":21},'
    b'{"kind":"numerical_kv_staging_start","timestamp_ns":22},'
    b'{"kind":"numerical_attention_start","timestamp_ns":23},'
    b'{"kind":"numerical_attention_end","timestamp_ns":30},'
    b'{"kind":"output_ready","timestamp_ns":31}]}'
)


class FlatM13B4TraceTests(unittest.TestCase):
    def parse_reference(self) -> FlatM13B4TraceV1:
        return FlatM13B4TraceV1.from_canonical_json_bytes(FLAT_REFERENCE_FIXTURE)

    def test_exact_flat_reference_vector_round_trips(self) -> None:
        trace = self.parse_reference()
        self.assertEqual(trace.schema, FLAT_M13B4_TRACE_SCHEMA)
        self.assertEqual(trace.canonical_json_bytes(), FLAT_REFERENCE_FIXTURE)
        self.assertEqual(len(trace.trace_sha256()), 64)
        self.assertEqual(
            FLAT_M13B4_REFERENCE_REVISION,
            "29c18275b5687b71599ef11f5badcc4571a52a41",
        )

    def test_noncanonical_json_is_rejected(self) -> None:
        parsed = json.loads(FLAT_REFERENCE_FIXTURE)
        pretty = json.dumps(parsed, indent=2).encode("utf-8")
        with self.assertRaisesRegex(FlatM13B4TraceError, "not in canonical"):
            FlatM13B4TraceV1.from_canonical_json_bytes(pretty)

    def test_unknown_top_level_field_is_rejected(self) -> None:
        parsed = json.loads(FLAT_REFERENCE_FIXTURE)
        parsed["claim"] = "overlap"
        payload = json.dumps(parsed, separators=(",", ":")).encode("utf-8")
        with self.assertRaisesRegex(FlatM13B4TraceError, "fields do not match"):
            FlatM13B4TraceV1.from_canonical_json_bytes(payload)

    def test_boolean_timestamp_is_not_accepted_as_integer(self) -> None:
        parsed = json.loads(FLAT_REFERENCE_FIXTURE)
        parsed["events"][0]["timestamp_ns"] = True
        payload = json.dumps(parsed, separators=(",", ":")).encode("utf-8")
        with self.assertRaisesRegex(FlatM13B4TraceError, "integer u64"):
            FlatM13B4TraceV1.from_canonical_json_bytes(payload)

    def test_timestamp_outside_u64_is_rejected(self) -> None:
        parsed = json.loads(FLAT_REFERENCE_FIXTURE)
        parsed["events"][-1]["timestamp_ns"] = 1 << 64
        payload = json.dumps(parsed, separators=(",", ":")).encode("utf-8")
        with self.assertRaisesRegex(FlatM13B4TraceError, "u64 range"):
            FlatM13B4TraceV1.from_canonical_json_bytes(payload)

    def test_duplicate_event_is_rejected(self) -> None:
        parsed = json.loads(FLAT_REFERENCE_FIXTURE)
        parsed["events"].insert(1, dict(parsed["events"][0]))
        payload = json.dumps(parsed, separators=(",", ":")).encode("utf-8")
        with self.assertRaisesRegex(FlatM13B4TraceError, "duplicate"):
            FlatM13B4TraceV1.from_canonical_json_bytes(payload)

    def test_backwards_timestamp_is_rejected(self) -> None:
        parsed = json.loads(FLAT_REFERENCE_FIXTURE)
        parsed["events"][3]["timestamp_ns"] = 9
        payload = json.dumps(parsed, separators=(",", ":")).encode("utf-8")
        with self.assertRaisesRegex(FlatM13B4TraceError, "non-decreasing"):
            FlatM13B4TraceV1.from_canonical_json_bytes(payload)

    def test_multi_dispatch_requires_explicit_sync_pair(self) -> None:
        parsed = json.loads(FLAT_REFERENCE_FIXTURE)
        parsed["scheduling_variant"] = "multi_dispatch_overlap_candidate"
        payload = json.dumps(parsed, separators=(",", ":")).encode("utf-8")
        with self.assertRaisesRegex(FlatM13B4TraceError, "explicit synchronization"):
            FlatM13B4TraceV1.from_canonical_json_bytes(payload)

    def test_multi_dispatch_sync_pair_is_accepted_inside_unit_bounds(self) -> None:
        parsed = json.loads(FLAT_REFERENCE_FIXTURE)
        parsed["scheduling_variant"] = "multi_dispatch_overlap_candidate"
        parsed["events"].insert(
            8, {"kind": "synchronization_wait_start", "timestamp_ns": 24}
        )
        parsed["events"].insert(
            9, {"kind": "synchronization_wait_end", "timestamp_ns": 25}
        )
        payload = json.dumps(parsed, separators=(",", ":")).encode("utf-8")
        trace = FlatM13B4TraceV1.from_canonical_json_bytes(payload)
        self.assertEqual(trace.canonical_json_bytes(), payload)

    def test_prefill_requires_boolean_and_numerical_commit_visibility(self) -> None:
        payload = (
            b'{"schema":"flat.m13b4-trace.v1","timing_source":"host_wall_clock",'
            b'"scheduling_variant":"serial_matched","scope":"prefill","events":['
            b'{"kind":"numerical_kv_commit","timestamp_ns":100},'
            b'{"kind":"boolean_signature_commit","timestamp_ns":110},'
            b'{"kind":"decode_visible","timestamp_ns":120}]}'
        )
        trace = FlatM13B4TraceV1.from_canonical_json_bytes(payload)
        self.assertEqual(trace.canonical_json_bytes(), payload)

        parsed = json.loads(payload)
        parsed["events"] = [event for event in parsed["events"] if event["kind"] != "boolean_signature_commit"]
        missing = json.dumps(parsed, separators=(",", ":")).encode("utf-8")
        with self.assertRaisesRegex(FlatM13B4TraceError, "missing required"):
            FlatM13B4TraceV1.from_canonical_json_bytes(missing)


if __name__ == "__main__":
    unittest.main()
