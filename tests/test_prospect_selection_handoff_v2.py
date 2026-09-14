import json
import unittest

from kvlab.prospect_selection_handoff_v2 import (
    PROSPECT_KV_SELECTION_HANDOFF_SCHEMA_V2,
    ProspectKvSelectionHandoffV2,
    ProspectKvSelectionHandoffV2Error,
)


class ProspectSelectionHandoffV2Tests(unittest.TestCase):
    def test_repeated_vocabulary_tokens_are_distinct_occurrences(self) -> None:
        handoff = ProspectKvSelectionHandoffV2.capture(
            token_ids=[7, 11, 7, 7, 19],
            bytes_per_token=64,
            policy="lru",
            retained_positions=[0, 2, 4],
        )

        self.assertEqual(handoff.schema, PROSPECT_KV_SELECTION_HANDOFF_SCHEMA_V2)
        self.assertEqual(handoff.input_token_ids, (7, 11, 7, 7, 19))
        self.assertEqual(handoff.retained_positions, (0, 2, 4))
        self.assertEqual(handoff.evicted_positions, (1, 3))
        self.assertEqual(handoff.retained_token_ids, (7, 7, 19))
        self.assertEqual(handoff.evicted_token_ids, (11, 7))
        self.assertEqual(handoff.logical_input_bytes, 320)
        self.assertEqual(handoff.logical_retained_bytes, 192)
        self.assertEqual(handoff.logical_evicted_bytes, 128)

        replayed = ProspectKvSelectionHandoffV2.from_canonical_json(
            handoff.canonical_json()
        )
        self.assertEqual(replayed, handoff)

    def test_two_equal_token_values_can_be_selected_independently(self) -> None:
        left = ProspectKvSelectionHandoffV2.capture(
            token_ids=[5, 5, 5],
            bytes_per_token=32,
            policy="keep_first",
            retained_positions=[0],
        )
        middle = ProspectKvSelectionHandoffV2.capture(
            token_ids=[5, 5, 5],
            bytes_per_token=32,
            policy="keep_middle",
            retained_positions=[1],
        )

        self.assertEqual(left.retained_token_ids, middle.retained_token_ids)
        self.assertNotEqual(left.retained_positions, middle.retained_positions)
        self.assertNotEqual(left.canonical_json(), middle.canonical_json())

    def test_positions_must_be_strict_unique_and_in_range(self) -> None:
        for retained in ([1, 1], [2, 1], [-1], [3]):
            with self.subTest(retained=retained):
                with self.assertRaises(ProspectKvSelectionHandoffV2Error):
                    ProspectKvSelectionHandoffV2.capture(
                        token_ids=[1, 2, 1],
                        bytes_per_token=8,
                        policy="test",
                        retained_positions=retained,
                    )

    def test_tampered_partition_is_rejected(self) -> None:
        handoff = ProspectKvSelectionHandoffV2.capture(
            token_ids=[3, 3, 4, 3],
            bytes_per_token=16,
            policy="magnitude",
            retained_positions=[0, 3],
        )
        raw = json.loads(handoff.canonical_json())
        raw["evicted_positions"] = [1]
        payload = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        with self.assertRaises(ProspectKvSelectionHandoffV2Error):
            ProspectKvSelectionHandoffV2.from_canonical_json(payload)

    def test_noncanonical_json_is_rejected(self) -> None:
        handoff = ProspectKvSelectionHandoffV2.capture(
            token_ids=[1, 1],
            bytes_per_token=4,
            policy="random",
            retained_positions=[1],
        )
        raw = json.loads(handoff.canonical_json())
        payload = json.dumps(raw, sort_keys=True, indent=2)
        with self.assertRaises(ProspectKvSelectionHandoffV2Error):
            ProspectKvSelectionHandoffV2.from_canonical_json(payload)


if __name__ == "__main__":
    unittest.main()
