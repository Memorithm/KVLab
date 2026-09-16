import json
import unittest

from kvlab.consumer_receipt import (
    BIKV_CONSUMER_RECEIPT_SCHEMA_V1,
    BikvConsumerReceiptError,
    BikvConsumerReceiptV1,
)
from kvlab.prospect_handoff import ProspectBkvHandoffV1


KVLAB_REVISION = "0fb5adc5babfaea9077342db881ce775eacc4442"
PROSPECT_REVISION = "b6d96469bcb92f50cb95e7512920cebea1c4dd41"
FLAT_EVIDENCE_SCHEMA = "prospect.flat-boolean-routing-evidence/v1"
HANDOFF = (
    '{"admitted_pages":[0,1,3],"generation":0,"max_distance":1,'
    '"page_words":[["000000000000000a"],["000000000000000e"],'
    '["0000000000000000"],["000000000000000b"]],'
    '"query_words":["000000000000000a"],'
    '"schema":"kvlab.prospect-bkv-handoff/v1","signature_bits":4}'
)


class BikvConsumerReceiptTests(unittest.TestCase):
    def handoff(self) -> ProspectBkvHandoffV1:
        return ProspectBkvHandoffV1.from_canonical_json(HANDOFF)

    def receipt(self) -> BikvConsumerReceiptV1:
        return BikvConsumerReceiptV1.capture(
            self.handoff(),
            producer_repository="Memorithm/KVLab",
            producer_revision=KVLAB_REVISION,
            consumer_repository="Memorithm/ProspectEngine",
            consumer_revision=PROSPECT_REVISION,
            consumer_evidence_schema=FLAT_EVIDENCE_SCHEMA,
        )

    def test_capture_binds_real_merged_cross_project_revisions(self) -> None:
        receipt = self.receipt()

        self.assertEqual(receipt.schema, BIKV_CONSUMER_RECEIPT_SCHEMA_V1)
        self.assertEqual(receipt.producer_revision, KVLAB_REVISION)
        self.assertEqual(receipt.consumer_revision, PROSPECT_REVISION)
        self.assertTrue(receipt.verifies_handoff(self.handoff()))

    def test_exact_exchange_verification_binds_all_endpoint_identity(self) -> None:
        self.assertTrue(
            self.receipt().verifies_exchange(
                self.handoff(),
                producer_repository="Memorithm/KVLab",
                producer_revision=KVLAB_REVISION,
                consumer_repository="Memorithm/ProspectEngine",
                consumer_revision=PROSPECT_REVISION,
                consumer_evidence_schema=FLAT_EVIDENCE_SCHEMA,
            )
        )

    def test_exact_exchange_verification_fails_closed_on_each_identity_drift(self) -> None:
        receipt = self.receipt()
        common = {
            "producer_repository": "Memorithm/KVLab",
            "producer_revision": KVLAB_REVISION,
            "consumer_repository": "Memorithm/ProspectEngine",
            "consumer_revision": PROSPECT_REVISION,
            "consumer_evidence_schema": FLAT_EVIDENCE_SCHEMA,
        }
        mutations = {
            "producer_repository": "Memorithm/OtherLab",
            "producer_revision": "1" * 40,
            "consumer_repository": "Memorithm/FLAT-ATTENTION",
            "consumer_revision": "2" * 40,
            "consumer_evidence_schema": "different-evidence/v1",
        }
        for field, changed in mutations.items():
            with self.subTest(field=field):
                actual = dict(common)
                actual[field] = changed
                self.assertFalse(receipt.verifies_exchange(self.handoff(), **actual))

    def test_canonical_json_round_trip_is_exact(self) -> None:
        receipt = self.receipt()
        payload = receipt.canonical_json()

        self.assertEqual(payload, json.dumps(json.loads(payload), sort_keys=True, separators=(",", ":")))
        self.assertEqual(BikvConsumerReceiptV1.from_canonical_json(payload), receipt)

    def test_receipt_rejects_noncanonical_json(self) -> None:
        payload = self.receipt().canonical_json().replace(",\"handoff_schema\"", ", \"handoff_schema\"")

        with self.assertRaisesRegex(BikvConsumerReceiptError, "not canonical"):
            BikvConsumerReceiptV1.from_canonical_json(payload)

    def test_receipt_rejects_revision_or_digest_shape_drift(self) -> None:
        decoded = json.loads(self.receipt().canonical_json())
        decoded["consumer_revision"] = "NOT-A-GIT-SHA"
        malformed_revision = json.dumps(decoded, sort_keys=True, separators=(",", ":"))
        with self.assertRaisesRegex(BikvConsumerReceiptError, "Git SHA"):
            BikvConsumerReceiptV1.from_canonical_json(malformed_revision)

        decoded = json.loads(self.receipt().canonical_json())
        decoded["handoff_sha256"] = "0" * 63
        malformed_digest = json.dumps(decoded, sort_keys=True, separators=(",", ":"))
        with self.assertRaisesRegex(BikvConsumerReceiptError, "SHA-256"):
            BikvConsumerReceiptV1.from_canonical_json(malformed_digest)

    def test_receipt_detects_different_handoff_payload(self) -> None:
        receipt = self.receipt()
        decoded = json.loads(HANDOFF)
        decoded["generation"] = 1
        changed = ProspectBkvHandoffV1.from_canonical_json(
            json.dumps(decoded, sort_keys=True, separators=(",", ":"))
        )

        self.assertFalse(receipt.verifies_handoff(changed))
        self.assertFalse(
            receipt.verifies_exchange(
                changed,
                producer_repository="Memorithm/KVLab",
                producer_revision=KVLAB_REVISION,
                consumer_repository="Memorithm/ProspectEngine",
                consumer_revision=PROSPECT_REVISION,
                consumer_evidence_schema=FLAT_EVIDENCE_SCHEMA,
            )
        )


if __name__ == "__main__":
    unittest.main()
