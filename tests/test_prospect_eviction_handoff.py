import json
import unittest

from kvlab.eviction import EvictionPolicy
from kvlab.prospect_eviction_handoff import (
    PROSPECT_KV_EVICTION_HANDOFF_SCHEMA_V1,
    ProspectKvEvictionHandoffError,
    ProspectKvEvictionHandoffV1,
)


class ProspectKvEvictionHandoffTests(unittest.TestCase):
    def test_canonical_round_trip_replays_exact_oldest_first_eviction(self) -> None:
        handoff = ProspectKvEvictionHandoffV1.capture(
            token_ids=[10, 11, 12, 13, 14],
            bytes_per_token=2048,
            policy=EvictionPolicy(max_tokens=3),
        )

        self.assertEqual(handoff.schema, PROSPECT_KV_EVICTION_HANDOFF_SCHEMA_V1)
        self.assertEqual(handoff.evicted_token_ids, (10, 11))
        self.assertEqual(handoff.retained_token_ids, (12, 13, 14))
        self.assertEqual(handoff.logical_evicted_bytes, 4096)

        payload = handoff.canonical_json()
        decoded = ProspectKvEvictionHandoffV1.from_canonical_json(payload)
        self.assertEqual(decoded, handoff)
        self.assertEqual(
            payload,
            json.dumps(json.loads(payload), sort_keys=True, separators=(",", ":")),
        )

    def test_rejects_tampered_retained_tokens(self) -> None:
        handoff = ProspectKvEvictionHandoffV1.capture(
            token_ids=[0, 1, 2, 3],
            bytes_per_token=1024,
            policy=EvictionPolicy(max_tokens=2),
        )
        raw = json.loads(handoff.canonical_json())
        raw["retained_token_ids"] = [1, 2, 3]
        tampered = json.dumps(raw, sort_keys=True, separators=(",", ":"))

        with self.assertRaisesRegex(
            ProspectKvEvictionHandoffError,
            "recorded eviction outcome does not match replayed KVLab semantics",
        ):
            ProspectKvEvictionHandoffV1.from_canonical_json(tampered)

    def test_rejects_noncanonical_or_unknown_fields(self) -> None:
        handoff = ProspectKvEvictionHandoffV1.capture(
            token_ids=[0, 1],
            bytes_per_token=512,
            policy=EvictionPolicy(max_tokens=1),
        )
        canonical = handoff.canonical_json()

        with self.assertRaisesRegex(
            ProspectKvEvictionHandoffError, "handoff JSON is not canonical"
        ):
            ProspectKvEvictionHandoffV1.from_canonical_json(canonical.replace(",", ", ", 1))

        raw = json.loads(canonical)
        raw["physical_bytes_freed"] = 512
        unknown = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        with self.assertRaisesRegex(
            ProspectKvEvictionHandoffError, "handoff fields do not match schema v1"
        ):
            ProspectKvEvictionHandoffV1.from_canonical_json(unknown)

    def test_zero_eviction_is_explicit_and_replayable(self) -> None:
        handoff = ProspectKvEvictionHandoffV1.capture(
            token_ids=[5, 6],
            bytes_per_token=256,
            policy=EvictionPolicy(max_tokens=4),
        )
        self.assertEqual(handoff.evicted_token_ids, ())
        self.assertEqual(handoff.retained_token_ids, (5, 6))
        self.assertEqual(handoff.logical_evicted_bytes, 0)
        self.assertEqual(
            ProspectKvEvictionHandoffV1.from_canonical_json(handoff.canonical_json()),
            handoff,
        )


if __name__ == "__main__":
    unittest.main()
