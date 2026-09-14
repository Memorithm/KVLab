from pathlib import Path
import unittest

from kvlab.prospect_position_baselines import budget_matched_lru_random
from kvlab.prospect_position_comparison import preflight_budget_matched_campaign
from kvlab.prospect_real_model_campaign_v4 import PositionCampaignSpecV1


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / "experiments" / "prospect" / "smollm2-r1"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
KVLAB_RUN_REVISION = "404577ce939093767dc75d2d67de2fe3c16fa4dc"
NNIS_RUNTIME_REVISION = "58e7db8e1c4b471a7fe82a4beba11904240c4e89"
TRACE = (
    22007,
    6463,
    314,
    260,
    3075,
    338,
    6650,
    260,
    2591,
    284,
    260,
    8872,
    1592,
    30,
    198,
    198,
    504,
    8872,
    314,
    253,
    8304,
    282,
    260,
    2591,
    30,
    657,
    314,
    253,
    19284,
    1248,
    338,
    21837,
    260,
    2591,
    30,
)


class SmolLm2R1PositionPreregistrationTests(unittest.TestCase):
    def test_preregistered_campaigns_are_canonical_budget_matched_and_reproducible(self):
        for retained_count in (7, 14, 20):
            path = CAMPAIGN_DIR / f"retain-{retained_count:02d}-of-27.json"
            payload = path.read_text(encoding="utf-8")
            spec = PositionCampaignSpecV1.from_canonical_json(payload)
            preflight = preflight_budget_matched_campaign(spec)

            self.assertEqual(spec.model_id, "HuggingFaceTB/SmolLM2-135M")
            self.assertEqual(spec.model_revision, MODEL_REVISION)
            self.assertEqual(spec.tokenizer_revision, MODEL_REVISION)
            self.assertEqual(spec.run_repository_revision, KVLAB_RUN_REVISION)
            self.assertEqual(spec.runtime_backend, "nnis-kvlab-v4")
            self.assertEqual(spec.runtime_revision, NNIS_RUNTIME_REVISION)
            self.assertEqual(spec.bytes_per_token, 46_080)
            self.assertEqual(spec.seed, 7)
            self.assertEqual(
                spec.model_input_token_ids + spec.evaluation_token_ids, TRACE
            )
            self.assertEqual(len(spec.model_input_token_ids), 27)
            self.assertEqual(len(spec.evaluation_token_ids), 8)

            expected = budget_matched_lru_random(
                input_position_count=27,
                retained_position_count=retained_count,
                seed=spec.seed,
            )
            self.assertEqual(
                tuple(selection.policy for selection in spec.selections),
                tuple(selection.policy for selection in expected),
            )
            self.assertEqual(
                tuple(selection.retained_positions for selection in spec.selections),
                tuple(selection.retained_positions for selection in expected),
            )
            self.assertEqual(preflight.retained_position_count, retained_count)
            self.assertEqual(
                preflight.logical_retained_bytes, retained_count * 46_080
            )
            self.assertEqual(
                preflight.logical_evicted_bytes, (27 - retained_count) * 46_080
            )


if __name__ == "__main__":
    unittest.main()
