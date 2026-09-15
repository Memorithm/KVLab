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

    def test_semantically_equal_but_byte_different_sources_do_not_alias(self) -> None:
        compact = b'{"a":1,"b":2}'
        spaced = b'{"a": 1, "b": 2}'

        first = BikvEvidenceReceiptV1.from_json_bytes(
            producer_repo="Memorithm/FLAT-ATTENTION",
            producer_commit=FLAT_REVISION,
            payload=compact,
        )
        second = BikvEvidenceReceiptV1.from_json_bytes(
            producer_repo="Memorithm/FLAT-ATTENTION",
            producer_commit=FLAT_REVISION,
            payload=spaced,
        )

        self.assertEqual(json.loads(compact), json.loads(spaced))
        self.assertNotEqual(first.source_sha256, second.source_sha256)

    def test_canonical_receipt_bytes_and_digest_are_deterministic(self) -> None:
        receipt = BikvEvidenceReceiptV1.from_json_bytes(
            producer_repo="Memorithm/FLAT-ATTENTION",
            producer_commit=FLAT_REVISION,
            payload=b'{"schema":"example/v1"}',
        )

        canonical = receipt.canonical_json_bytes()
        self.assertEqual(canonical, json.dumps(json.loads(canonical), sort_keys=True, separators=(",", ":")).encode())
        self.assertEqual(receipt.receipt_sha256(), hashlib.sha256(canonical).hexdigest())

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
