import json
import unittest

from kvlab.boolean_kv import BooleanKvCache, PackedBits
from kvlab.prospect_handoff import (
    PROSPECT_BKV_HANDOFF_SCHEMA_V1,
    ProspectBkvHandoffError,
    ProspectBkvHandoffV1,
)


def packed(value: int, bits: int = 4) -> PackedBits:
    return PackedBits(bit_length=bits, words=(value,))


class ProspectBkvHandoffTests(unittest.TestCase):
    def cache(self) -> BooleanKvCache:
        cache = BooleanKvCache(4)
        for value in (0b1010, 0b1110, 0b0000, 0b1011):
            cache.append(packed(value))
        return cache

    def test_capture_normalizes_matches_to_logical_page_order(self) -> None:
        handoff = ProspectBkvHandoffV1.capture(
            self.cache(),
            packed(0b1010),
            max_distance=1,
        )

        self.assertEqual(handoff.schema, PROSPECT_BKV_HANDOFF_SCHEMA_V1)
        self.assertEqual(handoff.signature_bits, 4)
        self.assertEqual(handoff.generation, 0)
        self.assertEqual(handoff.query_words, ("000000000000000a",))
        self.assertEqual(
            handoff.page_words,
            (
                ("000000000000000a",),
                ("000000000000000e",),
                ("0000000000000000",),
                ("000000000000000b",),
            ),
        )
        self.assertEqual(handoff.admitted_pages, (0, 1, 3))

    def test_canonical_json_round_trip_is_exact(self) -> None:
        handoff = ProspectBkvHandoffV1.capture(
            self.cache(),
            packed(0b1010),
            max_distance=1,
        )
        payload = handoff.canonical_json()

        self.assertEqual(payload, json.dumps(json.loads(payload), sort_keys=True, separators=(",", ":")))
        self.assertEqual(ProspectBkvHandoffV1.from_canonical_json(payload), handoff)

    def test_decoder_rejects_tampered_candidate_set(self) -> None:
        handoff = ProspectBkvHandoffV1.capture(
            self.cache(),
            packed(0b1010),
            max_distance=1,
        )
        decoded = json.loads(handoff.canonical_json())
        decoded["admitted_pages"] = [0, 1, 2, 3]
        tampered = json.dumps(decoded, sort_keys=True, separators=(",", ":"))

        with self.assertRaisesRegex(ProspectBkvHandoffError, "exact Hamming rule"):
            ProspectBkvHandoffV1.from_canonical_json(tampered)

    def test_decoder_rejects_noncanonical_or_non_u64_word_encoding(self) -> None:
        handoff = ProspectBkvHandoffV1.capture(
            self.cache(),
            packed(0b1010),
            max_distance=1,
        )
        decoded = json.loads(handoff.canonical_json())
        decoded["query_words"] = ["A"]
        malformed = json.dumps(decoded, sort_keys=True, separators=(",", ":"))

        with self.assertRaisesRegex(ProspectBkvHandoffError, "16-digit lowercase hexadecimal"):
            ProspectBkvHandoffV1.from_canonical_json(malformed)

        pretty = json.dumps(json.loads(handoff.canonical_json()), sort_keys=True, indent=2)
        with self.assertRaisesRegex(ProspectBkvHandoffError, "not canonical"):
            ProspectBkvHandoffV1.from_canonical_json(pretty)

    def test_capture_rejects_empty_cache_and_out_of_range_threshold(self) -> None:
        with self.assertRaisesRegex(ProspectBkvHandoffError, "empty Boolean KV cache"):
            ProspectBkvHandoffV1.capture(BooleanKvCache(4), packed(0), max_distance=0)

        with self.assertRaisesRegex(ProspectBkvHandoffError, "invalid Boolean KV handoff inputs"):
            ProspectBkvHandoffV1.capture(self.cache(), packed(0), max_distance=5)


if __name__ == "__main__":
    unittest.main()
