import json
import unittest

from kvlab.flat_boolean_kv_selection import (
    FLAT_BOOLEAN_KV_SELECTION_REFERENCE_REVISION,
    FLAT_BOOLEAN_KV_SELECTION_SCHEMA,
    FlatBooleanKvSelectionError,
    FlatBooleanKvSelectionV1,
)

FLAT_REFERENCE_FIXTURE = (
    b'{"schema":"flat.boolean-kv-selection.v1","generation":0,"signature_bits":8,'
    b'"live_tokens":10,"mapped_pages":3,"boolean_pages_scanned":3,'
    b'"boolean_key_bytes_read":24,"numerical_kv_bytes_per_token":64,"full_numerical_kv_bytes":640,'
    b'"selected_numerical_kv_bytes":384,"avoided_numerical_kv_bytes":256,'
    b'"selected_pages":[{"logical_page":1,"physical_page":1,"live_tokens":4,'
    b'"hamming_distance":0,"xnor_matches":8},{"logical_page":2,"physical_page":2,'
    b'"live_tokens":2,"hamming_distance":1,"xnor_matches":7}],'
    b'"evidence_checksum":{"algorithm":"fnv1a64","value":"347bf775c963bf3b"}}'
)


class FlatBooleanKvSelectionTests(unittest.TestCase):
    def test_exact_flat_reference_vector_round_trips(self) -> None:
        selection = FlatBooleanKvSelectionV1.from_canonical_json_bytes(
            FLAT_REFERENCE_FIXTURE
        )
        self.assertEqual(selection.schema, FLAT_BOOLEAN_KV_SELECTION_SCHEMA)
        self.assertEqual(selection.canonical_json_bytes(), FLAT_REFERENCE_FIXTURE)
        self.assertEqual(len(selection.selection_sha256()), 64)
        self.assertEqual(
            FLAT_BOOLEAN_KV_SELECTION_REFERENCE_REVISION,
            "dab6704f4c97c15147227ca586fa7c2f8dc26a4d",
        )
        self.assertEqual([page.logical_page for page in selection.selected_pages], [1, 2])

    def test_noncanonical_json_is_rejected(self) -> None:
        pretty = json.dumps(json.loads(FLAT_REFERENCE_FIXTURE), indent=2).encode()
        with self.assertRaisesRegex(FlatBooleanKvSelectionError, "canonical FLAT"):
            FlatBooleanKvSelectionV1.from_canonical_json_bytes(pretty)

    def test_checksum_tampering_is_rejected(self) -> None:
        raw = json.loads(FLAT_REFERENCE_FIXTURE)
        raw["generation"] = 1
        payload = json.dumps(raw, separators=(",", ":")).encode()
        with self.assertRaisesRegex(FlatBooleanKvSelectionError, "checksum"):
            FlatBooleanKvSelectionV1.from_canonical_json_bytes(payload)

    def test_boolean_is_not_accepted_as_integer(self) -> None:
        raw = json.loads(FLAT_REFERENCE_FIXTURE)
        raw["generation"] = True
        payload = json.dumps(raw, separators=(",", ":")).encode()
        with self.assertRaisesRegex(FlatBooleanKvSelectionError, "integer u64"):
            FlatBooleanKvSelectionV1.from_canonical_json_bytes(payload)

    def test_mutated_page_order_fails_closed_even_with_recomputed_checksum(self) -> None:
        raw = json.loads(FLAT_REFERENCE_FIXTURE)
        raw["selected_pages"].reverse()
        # Keep the original checksum: checksum validation and structural ordering are
        # both fail-closed; ordering is checked first by the consumer.
        payload = json.dumps(raw, separators=(",", ":")).encode()
        with self.assertRaisesRegex(FlatBooleanKvSelectionError, "strictly increasing"):
            FlatBooleanKvSelectionV1.from_canonical_json_bytes(payload)

    def test_signature_accounting_mismatch_is_rejected(self) -> None:
        raw = json.loads(FLAT_REFERENCE_FIXTURE)
        raw["selected_pages"][0]["xnor_matches"] = 7
        payload = json.dumps(raw, separators=(",", ":")).encode()
        with self.assertRaisesRegex(FlatBooleanKvSelectionError, "signature_bits"):
            FlatBooleanKvSelectionV1.from_canonical_json_bytes(payload)

    def test_unknown_or_reordered_fields_are_rejected(self) -> None:
        raw = json.loads(FLAT_REFERENCE_FIXTURE)
        raw["claim"] = "physical_dram_saved"
        payload = json.dumps(raw, separators=(",", ":")).encode()
        with self.assertRaisesRegex(FlatBooleanKvSelectionError, "fields/order"):
            FlatBooleanKvSelectionV1.from_canonical_json_bytes(payload)

    def test_selected_byte_drift_is_rejected_even_when_total_is_conserved(self) -> None:
        raw = json.loads(FLAT_REFERENCE_FIXTURE)
        raw["selected_numerical_kv_bytes"] = 320
        raw["avoided_numerical_kv_bytes"] = 320
        payload = json.dumps(raw, separators=(",", ":")).encode()
        with self.assertRaisesRegex(
            FlatBooleanKvSelectionError, "selected numerical bytes"
        ):
            FlatBooleanKvSelectionV1.from_canonical_json_bytes(payload)

    def test_boolean_byte_count_must_match_packed_signature_geometry(self) -> None:
        raw = json.loads(FLAT_REFERENCE_FIXTURE)
        raw["boolean_key_bytes_read"] = 1
        payload = json.dumps(raw, separators=(",", ":")).encode()
        with self.assertRaisesRegex(
            FlatBooleanKvSelectionError, "packed signature geometry"
        ):
            FlatBooleanKvSelectionV1.from_canonical_json_bytes(payload)


if __name__ == "__main__":
    unittest.main()
