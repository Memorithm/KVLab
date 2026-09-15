import hashlib
import json
import unittest

from kvlab.bikv_evidence_receipt import BikvEvidenceReceiptV1


FLAT_REVISION = "0123456789abcdef0123456789abcdef01234567"


class BikvEvidenceReceiptTests(unittest.TestCase):
    def test_capture_hashes_exact_source_bytes(self) -> None:
        payload = b'{"metrics":{"candidate_density":0.25},"schema":"example/v1"}\n'
        receipt = BikvEvidenceReceiptV1.from_json_bytes(
            producer_repo="Memorithm/FLAT-ATTENTION",
            producer_commit=FLAT_REVISION,
            payload=payload,
        )

        self.assertEqual(receipt.source_sha256, hashlib.sha256(payload).hexdigest())
        self.assertEqual(receipt.source_bytes, len(payload))
        receipt.verify_source_bytes(payload)

    def test_semantically_equal_but_byte_different_sources_do_not_alias(self) -> None:
        first_encoding = b'{"a":1,"b":2}'
        second_encoding = b'{"b":2,"a":1}'

        first = BikvEvidenceReceiptV1.from_json_bytes(
            producer_repo="Memorithm/FLAT-ATTENTION",
            producer_commit=FLAT_REVISION,
            payload=first_encoding,
        )
        second = BikvEvidenceReceiptV1.from_json_bytes(
            producer_repo="Memorithm/FLAT-ATTENTION",
            producer_commit=FLAT_REVISION,
            payload=second_encoding,
        )

        self.assertEqual(len(first_encoding), len(second_encoding))
        self.assertEqual(json.loads(first_encoding), json.loads(second_encoding))
        self.assertNotEqual(first.source_sha256, second.source_sha256)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            first.verify_source_bytes(second_encoding)

    def test_verify_source_rejects_length_mismatch_before_digest_identity(self) -> None:
        payload = b'{"schema":"example/v1"}'
        receipt = BikvEvidenceReceiptV1.from_json_bytes(
            producer_repo="Memorithm/FLAT-ATTENTION",
            producer_commit=FLAT_REVISION,
            payload=payload,
        )

        with self.assertRaisesRegex(ValueError, "byte length"):
            receipt.verify_source_bytes(payload + b"\n")

    def test_canonical_receipt_bytes_and_digest_are_deterministic(self) -> None:
        receipt = BikvEvidenceReceiptV1.from_json_bytes(
            producer_repo="Memorithm/FLAT-ATTENTION",
            producer_commit=FLAT_REVISION,
            payload=b'{"schema":"example/v1"}',
        )

        canonical = receipt.canonical_json_bytes()
        self.assertEqual(canonical, json.dumps(json.loads(canonical), sort_keys=True, separators=(",", ":")).encode())
        self.assertEqual(receipt.receipt_sha256(), hashlib.sha256(canonical).hexdigest())
        self.assertEqual(BikvEvidenceReceiptV1.from_canonical_json_bytes(canonical), receipt)

    def test_receipt_parser_rejects_noncanonical_or_schema_drift(self) -> None:
        receipt = BikvEvidenceReceiptV1.from_json_bytes(
            producer_repo="Memorithm/FLAT-ATTENTION",
            producer_commit=FLAT_REVISION,
            payload=b'{"schema":"example/v1"}',
        )
        canonical_object = json.loads(receipt.canonical_json_bytes())

        noncanonical = json.dumps(canonical_object, sort_keys=True, indent=2).encode()
        with self.assertRaisesRegex(ValueError, "canonical JSON"):
            BikvEvidenceReceiptV1.from_canonical_json_bytes(noncanonical)

        canonical_object["unexpected"] = "field"
        with self.assertRaisesRegex(ValueError, "fields"):
            BikvEvidenceReceiptV1.from_canonical_json_bytes(
                json.dumps(canonical_object, sort_keys=True, separators=(",", ":")).encode()
            )

    def test_receipt_parser_rejects_boolean_source_size(self) -> None:
        receipt = BikvEvidenceReceiptV1.from_json_bytes(
            producer_repo="Memorithm/FLAT-ATTENTION",
            producer_commit=FLAT_REVISION,
            payload=b"{}",
        )
        malformed = json.loads(receipt.canonical_json_bytes())
        malformed["source_bytes"] = True
        malformed_bytes = json.dumps(malformed, sort_keys=True, separators=(",", ":")).encode()

        with self.assertRaisesRegex(ValueError, "integer"):
            BikvEvidenceReceiptV1.from_canonical_json_bytes(malformed_bytes)

    def test_rejects_non_object_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "top-level JSON object"):
            BikvEvidenceReceiptV1.from_json_bytes(
                producer_repo="Memorithm/FLAT-ATTENTION",
                producer_commit=FLAT_REVISION,
                payload=b"[]",
            )

    def test_rejects_invalid_json_and_invalid_commit(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            BikvEvidenceReceiptV1.from_json_bytes(
                producer_repo="Memorithm/FLAT-ATTENTION",
                producer_commit=FLAT_REVISION,
                payload=b"{",
            )

        with self.assertRaisesRegex(ValueError, "40-hex"):
            BikvEvidenceReceiptV1.from_json_bytes(
                producer_repo="Memorithm/FLAT-ATTENTION",
                producer_commit="not-a-commit",
                payload=b"{}",
            )


if __name__ == "__main__":
    unittest.main()
