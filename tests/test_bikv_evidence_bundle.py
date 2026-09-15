from __future__ import annotations

import json
import unittest

from kvlab.bikv_evidence_bundle import (
    BikvEvidenceBundleEntryV1,
    BikvEvidenceBundleV1,
)
from kvlab.bikv_evidence_receipt import BikvEvidenceReceiptV1


def _receipt(*, commit: str, payload: bytes) -> BikvEvidenceReceiptV1:
    return BikvEvidenceReceiptV1.from_json_bytes(
        producer_repo="Memorithm/FLAT-ATTENTION",
        producer_commit=commit,
        payload=payload,
    )


class BikvEvidenceBundleTests(unittest.TestCase):
    def test_bundle_canonicalizes_roles_and_round_trips(self) -> None:
        k64 = _receipt(
            commit="7a4eec7dbb90627dde800bc1c3c90dbdd890d6d1",
            payload=b'{"schema":"flat.k6.4"}',
        )
        trace = _receipt(
            commit="ca164387143cd58b5349d1e52850a2907071ed4c",
            payload=b'{"schema":"flat.m13b.4"}',
        )

        bundle = BikvEvidenceBundleV1.from_receipts(
            [
                ("m13b4-trace", trace),
                ("k6.4-envelope", k64),
            ]
        )

        self.assertEqual(
            [entry.role for entry in bundle.entries],
            ["k6.4-envelope", "m13b4-trace"],
        )
        encoded = bundle.canonical_json_bytes()
        decoded = BikvEvidenceBundleV1.from_canonical_json_bytes(encoded)
        self.assertEqual(decoded, bundle)
        self.assertEqual(decoded.bundle_sha256(), bundle.bundle_sha256())

    def test_entry_verifies_exact_receipt_identity(self) -> None:
        receipt = _receipt(
            commit="7a4eec7dbb90627dde800bc1c3c90dbdd890d6d1",
            payload=b'{"value":1}',
        )
        entry = BikvEvidenceBundleEntryV1.from_receipt(
            role="k6.4-envelope", receipt=receipt
        )

        entry.verify_receipt(receipt)

        replacement = _receipt(
            commit="7a4eec7dbb90627dde800bc1c3c90dbdd890d6d1",
            payload=b'{"value":2}',
        )
        with self.assertRaisesRegex(ValueError, "receipt SHA-256"):
            entry.verify_receipt(replacement)

    def test_bundle_rejects_duplicate_roles(self) -> None:
        first = _receipt(
            commit="7a4eec7dbb90627dde800bc1c3c90dbdd890d6d1",
            payload=b'{"a":1}',
        )
        second = _receipt(
            commit="ca164387143cd58b5349d1e52850a2907071ed4c",
            payload=b'{"b":2}',
        )

        with self.assertRaisesRegex(ValueError, "unique roles"):
            BikvEvidenceBundleV1.from_receipts(
                [("trace", first), ("trace", second)]
            )

    def test_bundle_rejects_same_receipt_under_multiple_roles(self) -> None:
        receipt = _receipt(
            commit="7a4eec7dbb90627dde800bc1c3c90dbdd890d6d1",
            payload=b'{"a":1}',
        )

        with self.assertRaisesRegex(ValueError, "multiple bundle roles"):
            BikvEvidenceBundleV1.from_receipts(
                [("candidate", receipt), ("dense-control", receipt)]
            )

    def test_bundle_rejects_noncanonical_json_key_order(self) -> None:
        receipt = _receipt(
            commit="7a4eec7dbb90627dde800bc1c3c90dbdd890d6d1",
            payload=b'{"a":1}',
        )
        bundle = BikvEvidenceBundleV1.from_receipts([("candidate", receipt)])
        canonical = bundle.canonical_json_bytes()
        parsed = json.loads(canonical)
        alternate = json.dumps(parsed, sort_keys=False, indent=2).encode("utf-8")
        self.assertNotEqual(alternate, canonical)

        with self.assertRaisesRegex(ValueError, "not in canonical JSON form"):
            BikvEvidenceBundleV1.from_canonical_json_bytes(alternate)

    def test_bundle_rejects_reserved_or_invalid_roles(self) -> None:
        receipt = _receipt(
            commit="7a4eec7dbb90627dde800bc1c3c90dbdd890d6d1",
            payload=b'{"a":1}',
        )

        for role in ("", "UPPER", "contains space", "x" * 65):
            with self.subTest(role=role), self.assertRaisesRegex(
                ValueError, "role must match"
            ):
                BikvEvidenceBundleV1.from_receipts([(role, receipt)])

    def test_bundle_is_provenance_only_and_does_not_require_scientific_fields(self) -> None:
        opaque = _receipt(
            commit="7a4eec7dbb90627dde800bc1c3c90dbdd890d6d1",
            payload=b'{"opaque":true}',
        )
        bundle = BikvEvidenceBundleV1.from_receipts([("opaque-evidence", opaque)])

        self.assertEqual(bundle.entries[0].source_bytes, len(b'{"opaque":true}'))
        bundle.entries[0].verify_receipt(opaque)


if __name__ == "__main__":
    unittest.main()
