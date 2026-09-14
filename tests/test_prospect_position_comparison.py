import json
import unittest

from kvlab.prospect_position_comparison import (
    ProspectKvPositionComparisonError,
    preflight_budget_matched_campaign,
)
from kvlab.prospect_real_model_campaign_v4 import (
    PROSPECT_KV_POSITION_CAMPAIGN_SCHEMA_V1,
    PositionCampaignSpecV1,
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _campaign(selections):
    payload = _canonical(
        {
            "schema": PROSPECT_KV_POSITION_CAMPAIGN_SCHEMA_V1,
            "experiment_id": "budget-matched-comparison",
            "run_repository_revision": "a" * 40,
            "model_id": "example/model",
            "model_revision": "model-r1",
            "tokenizer_revision": "tokenizer-r1",
            "runtime_backend": "nnis-kvlab-v4",
            "runtime_revision": "b" * 40,
            "evaluation_id": "teacher-forced-001",
            "seed": 7,
            "bytes_per_token": 64,
            "model_input_token_ids": [7, 11, 7, 13, 19],
            "evaluation_token_ids": [23, 29],
            "selections": selections,
        }
    )
    return PositionCampaignSpecV1.from_canonical_json(payload)


class ProspectPositionComparisonTests(unittest.TestCase):
    def test_equal_logical_budgets_are_accepted_and_reported(self):
        spec = _campaign(
            [
                {"policy": "lru", "retained_positions": [0, 2, 4]},
                {"policy": "magnitude", "retained_positions": [1, 2, 4]},
                {"policy": "sensitivity_per_byte", "retained_positions": [0, 1, 4]},
            ]
        )

        summary = preflight_budget_matched_campaign(spec)

        self.assertEqual(
            summary.policies, ("lru", "magnitude", "sensitivity_per_byte")
        )
        self.assertEqual(summary.logical_input_bytes, 320)
        self.assertEqual(summary.logical_retained_bytes, 192)
        self.assertEqual(summary.logical_evicted_bytes, 128)
        self.assertEqual(summary.retained_position_count, 3)
        self.assertEqual(summary.campaign_spec_sha256, spec.sha256)
        self.assertEqual(
            json.loads(summary.canonical_json())["logical_retained_bytes"], 192
        )

    def test_mismatched_logical_budgets_fail_closed(self):
        spec = _campaign(
            [
                {"policy": "lru", "retained_positions": [0, 2, 4]},
                {"policy": "magnitude", "retained_positions": [1, 4]},
            ]
        )

        with self.assertRaisesRegex(
            ProspectKvPositionComparisonError, "retained-byte budget"
        ):
            preflight_budget_matched_campaign(spec)

    def test_comparison_preflight_does_not_modify_generic_campaign_validation(self):
        spec = _campaign(
            [
                {"policy": "lru", "retained_positions": [0, 2, 4]},
                {"policy": "magnitude", "retained_positions": [1, 4]},
            ]
        )

        spec.validate()
        self.assertEqual(len(spec.selections), 2)


if __name__ == "__main__":
    unittest.main()
