import json
import unittest

from kvlab.eviction import EvictionPolicy
from kvlab.prospect_eviction_effect import (
    PROSPECT_KV_EVICTION_EFFECT_SCHEMA_V1,
    ProspectKvEvictionEffectError,
    ProspectKvEvictionEffectV1,
)
from kvlab.prospect_eviction_handoff import ProspectKvEvictionHandoffV1
from kvlab.synthetic_trace import KvRegion, SyntheticKvTrace


class ProspectKvEvictionEffectTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = SyntheticKvTrace(
            "eviction-fixture",
            (
                KvRegion("r10", 64, (1.0, 0.0)),
                KvRegion("r11", 64, (0.0, 2.0)),
                KvRegion("r12", 64, (3.0, 0.0)),
                KvRegion("r13", 64, (0.0, 4.0)),
                KvRegion("r14", 64, (-1.0, 1.0)),
            ),
        )
        self.eviction = ProspectKvEvictionHandoffV1.capture(
            token_ids=[10, 11, 12, 13, 14],
            bytes_per_token=64,
            policy=EvictionPolicy(max_tokens=3),
        )
        self.binding = {
            10: "r10",
            11: "r11",
            12: "r12",
            13: "r13",
            14: "r14",
        }

    def test_capture_binds_logical_eviction_to_synthetic_numerical_effect(self) -> None:
        effect = ProspectKvEvictionEffectV1.capture(
            trace=self.trace,
            eviction=self.eviction,
            token_to_region=self.binding,
        )

        self.assertEqual(effect.schema, PROSPECT_KV_EVICTION_EFFECT_SCHEMA_V1)
        self.assertEqual(effect.retained_region_ids, ("r12", "r13", "r14"))
        self.assertEqual(effect.evicted_region_ids, ("r10", "r11"))
        self.assertEqual(effect.full_cache_output, (3.0, 7.0))
        self.assertEqual(effect.retained_output, (2.0, 5.0))
        self.assertAlmostEqual(effect.output_l2_delta, 5**0.5)
        self.assertEqual(effect.logical_evicted_bytes, 128)

    def test_canonical_json_round_trips_and_replays(self) -> None:
        effect = ProspectKvEvictionEffectV1.capture(
            trace=self.trace,
            eviction=self.eviction,
            token_to_region=self.binding,
        )
        payload = effect.canonical_json()
        replay = ProspectKvEvictionEffectV1.from_canonical_json(payload)

        self.assertEqual(replay, effect)
        self.assertEqual(replay.canonical_json(), payload)

    def test_tampered_numerical_result_is_rejected(self) -> None:
        effect = ProspectKvEvictionEffectV1.capture(
            trace=self.trace,
            eviction=self.eviction,
            token_to_region=self.binding,
        )
        raw = json.loads(effect.canonical_json())
        raw["retained_output"][0] += 1.0
        tampered = json.dumps(raw, sort_keys=True, separators=(",", ":"))

        with self.assertRaisesRegex(ProspectKvEvictionEffectError, "retained output"):
            ProspectKvEvictionEffectV1.from_canonical_json(tampered)

    def test_binding_must_exactly_cover_trace_and_eviction(self) -> None:
        with self.assertRaisesRegex(ProspectKvEvictionEffectError, "exactly cover"):
            ProspectKvEvictionEffectV1.capture(
                trace=self.trace,
                eviction=self.eviction,
                token_to_region={10: "r10", 11: "r11"},
            )

    def test_storage_must_match_logical_bytes_per_token(self) -> None:
        mismatched_trace = SyntheticKvTrace(
            "mismatch",
            (
                KvRegion("r10", 64, (1.0, 0.0)),
                KvRegion("r11", 64, (0.0, 2.0)),
                KvRegion("r12", 32, (3.0, 0.0)),
                KvRegion("r13", 64, (0.0, 4.0)),
                KvRegion("r14", 64, (-1.0, 1.0)),
            ),
        )
        with self.assertRaisesRegex(ProspectKvEvictionEffectError, "bytes_per_token"):
            ProspectKvEvictionEffectV1.capture(
                trace=mismatched_trace,
                eviction=self.eviction,
                token_to_region=self.binding,
            )

    def test_pretty_or_unknown_field_json_fails_closed(self) -> None:
        effect = ProspectKvEvictionEffectV1.capture(
            trace=self.trace,
            eviction=self.eviction,
            token_to_region=self.binding,
        )
        raw = json.loads(effect.canonical_json())
        pretty = json.dumps(raw, indent=2, sort_keys=True)
        with self.assertRaisesRegex(ProspectKvEvictionEffectError, "not canonical"):
            ProspectKvEvictionEffectV1.from_canonical_json(pretty)

        raw["unexpected"] = True
        unknown = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        with self.assertRaisesRegex(ProspectKvEvictionEffectError, "schema v1"):
            ProspectKvEvictionEffectV1.from_canonical_json(unknown)


if __name__ == "__main__":
    unittest.main()
