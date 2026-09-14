import json
import unittest

from kvlab.eviction import EvictionPolicy
from kvlab.prospect_eviction_comparison import (
    PROSPECT_KV_HEURISTIC_COMPARISON_SCHEMA_V1,
    ProspectKvHeuristicComparisonError,
    ProspectKvHeuristicComparisonV1,
)
from kvlab.prospect_eviction_effect import ProspectKvEvictionEffectV1
from kvlab.prospect_eviction_handoff import ProspectKvEvictionHandoffV1
from kvlab.synthetic_trace import KvRegion, SyntheticKvTrace


class ProspectKvHeuristicComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        trace = SyntheticKvTrace(
            "comparison-fixture",
            (
                KvRegion("r10", 64, (1.0, 0.0)),
                KvRegion("r11", 64, (0.0, 2.0)),
                KvRegion("r12", 64, (3.0, 0.0)),
                KvRegion("r13", 64, (0.0, 4.0)),
                KvRegion("r14", 64, (-1.0, 1.0)),
            ),
        )
        eviction = ProspectKvEvictionHandoffV1.capture(
            token_ids=[10, 11, 12, 13, 14],
            bytes_per_token=64,
            policy=EvictionPolicy(max_tokens=3),
        )
        self.effect = ProspectKvEvictionEffectV1.capture(
            trace=trace,
            eviction=eviction,
            token_to_region={
                10: "r10",
                11: "r11",
                12: "r12",
                13: "r13",
                14: "r14",
            },
        )

    def test_compares_existing_heuristics_under_same_budget(self) -> None:
        comparison = ProspectKvHeuristicComparisonV1.capture(
            effect=self.effect,
            random_seed=7,
        )

        self.assertEqual(
            comparison.schema,
            PROSPECT_KV_HEURISTIC_COMPARISON_SCHEMA_V1,
        )
        self.assertEqual(comparison.budget_bytes, 192)
        by_policy = {result.policy: result for result in comparison.results}
        self.assertEqual(
            set(by_policy),
            {
                "oldest_first",
                "lru",
                "magnitude",
                "synthetic_sensitivity_per_byte",
                "random",
            },
        )
        self.assertEqual(
            by_policy["oldest_first"].retained_region_ids,
            ("r12", "r13", "r14"),
        )
        self.assertEqual(
            by_policy["lru"].retained_region_ids,
            by_policy["oldest_first"].retained_region_ids,
        )
        self.assertEqual(
            by_policy["magnitude"].retained_region_ids,
            ("r11", "r12", "r13"),
        )
        self.assertEqual(
            by_policy["synthetic_sensitivity_per_byte"].retained_region_ids,
            ("r11", "r12", "r13"),
        )
        self.assertAlmostEqual(by_policy["magnitude"].output_l2_delta, 1.0)
        self.assertEqual(comparison.best_policy, "magnitude")

    def test_canonical_comparison_round_trips(self) -> None:
        comparison = ProspectKvHeuristicComparisonV1.capture(
            effect=self.effect,
            random_seed=19,
        )
        payload = comparison.canonical_json()
        replay = ProspectKvHeuristicComparisonV1.from_canonical_json(payload)

        self.assertEqual(replay, comparison)
        self.assertEqual(replay.canonical_json(), payload)

    def test_random_policy_is_reproducible_for_recorded_seed(self) -> None:
        first = ProspectKvHeuristicComparisonV1.capture(
            effect=self.effect,
            random_seed=12345,
        )
        second = ProspectKvHeuristicComparisonV1.capture(
            effect=self.effect,
            random_seed=12345,
        )
        self.assertEqual(first.results, second.results)

    def test_tampered_policy_result_is_rejected(self) -> None:
        comparison = ProspectKvHeuristicComparisonV1.capture(
            effect=self.effect,
            random_seed=7,
        )
        raw = json.loads(comparison.canonical_json())
        raw["results"][0]["output_l2_delta"] += 1.0
        tampered = json.dumps(raw, sort_keys=True, separators=(",", ":"))

        with self.assertRaisesRegex(ProspectKvHeuristicComparisonError, "L2 delta"):
            ProspectKvHeuristicComparisonV1.from_canonical_json(tampered)

    def test_tampered_best_policy_is_rejected(self) -> None:
        comparison = ProspectKvHeuristicComparisonV1.capture(
            effect=self.effect,
            random_seed=7,
        )
        raw = json.loads(comparison.canonical_json())
        raw["best_policy"] = "oldest_first"
        tampered = json.dumps(raw, sort_keys=True, separators=(",", ":"))

        with self.assertRaisesRegex(ProspectKvHeuristicComparisonError, "best_policy"):
            ProspectKvHeuristicComparisonV1.from_canonical_json(tampered)

    def test_seed_must_be_explicit_u64(self) -> None:
        with self.assertRaisesRegex(ProspectKvHeuristicComparisonError, "unsigned 64-bit"):
            ProspectKvHeuristicComparisonV1.capture(
                effect=self.effect,
                random_seed=-1,
            )


if __name__ == "__main__":
    unittest.main()
