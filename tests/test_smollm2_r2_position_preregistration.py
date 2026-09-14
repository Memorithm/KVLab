"""Input-contract checks only; these tests do not execute a model."""

import hashlib
import json
from pathlib import Path
import unittest

from kvlab.prospect_position_baselines import budget_matched_lru_random
from kvlab.prospect_position_comparison import preflight_budget_matched_campaign
from kvlab.prospect_real_model_campaign_v4 import PositionCampaignSpecV1


ROOT = Path(__file__).resolve().parents[1] / "experiments" / "prospect"
RUNTIME_R1 = "58e7db8e1c4b471a7fe82a4beba11904240c4e89"
RUNTIME_R2 = "091aabbb3e132627cf64716720aae530442d2a32"
EXECUTION_REVISION = "404577ce939093767dc75d2d67de2fe3c16fa4dc"
TRACE_SHA256 = "3411f378fb3c7010eb94361128c019206fb47bda4271f721498d517eb07ca65f"
R1_DIGESTS = {
    7: "5220e3910750ffad24dad1b023d6393491f9ef2ee3a14621b0dd122fd79d81bb",
    14: "0b0d461b5a29ed34e22803a287bdc054622008f87073b9f8956a8392cb4605b2",
    20: "909d0339fda4d1c1272a6da72aa14ab19fab971c68c560f7c82fb25d45dcbb28",
}
R2_DIGESTS = {
    7: "d826e0ca1869b6f3134e8b34bb65db14aa034d2518bf9559810ae80f19012346",
    14: "01eeec54e02bf3f56bbd2e75175e2f04fc4593701875a21b90981b40cfa4eff5",
    20: "e09b14c8479bac98b93625f3d98f667943d8318ff769dbbbf3db989959dcba07",
}


def campaign_bytes(generation, count):
    return (ROOT / generation / f"retain-{count:02d}-of-27.json").read_bytes()


class SmolLm2R2PreregistrationTests(unittest.TestCase):
    def test_r1_preregistration_bytes_are_unchanged(self):
        for count, expected in R1_DIGESTS.items():
            with self.subTest(count=count):
                self.assertEqual(
                    hashlib.sha256(campaign_bytes("smollm2-r1", count)).hexdigest(),
                    expected,
                )

    def test_r2_changes_only_runtime_and_experiment_identity(self):
        for count, expected in R2_DIGESTS.items():
            with self.subTest(count=count):
                old = json.loads(campaign_bytes("smollm2-r1", count))
                payload = campaign_bytes("smollm2-r2", count)
                new = json.loads(payload)
                self.assertEqual(hashlib.sha256(payload).hexdigest(), expected)
                self.assertEqual(set(old), set(new))
                self.assertEqual(
                    {key for key in old if old[key] != new[key]},
                    {"runtime_revision", "experiment_id"},
                )
                self.assertEqual(old["runtime_revision"], RUNTIME_R1)
                self.assertEqual(new["runtime_revision"], RUNTIME_R2)
                self.assertEqual(
                    new["experiment_id"],
                    f"smollm2-r2-position-retain-{count:02d}-of-27",
                )
                self.assertEqual(new["run_repository_revision"], EXECUTION_REVISION)

    def test_r2_replays_canonically_and_preserves_budget_matched_controls(self):
        for count in R2_DIGESTS:
            with self.subTest(count=count):
                payload = campaign_bytes("smollm2-r2", count).decode("utf-8")
                spec = PositionCampaignSpecV1.from_canonical_json(payload)
                self.assertEqual(spec.canonical_json(), payload)
                summary = preflight_budget_matched_campaign(spec)
                context, trace, selections = spec.build_execution_inputs()
                self.assertEqual(trace.sha256, TRACE_SHA256)
                self.assertEqual(context.trace_sha256, TRACE_SHA256)
                self.assertEqual(context.runtime_revision, RUNTIME_R2)
                expected = budget_matched_lru_random(
                    input_position_count=27,
                    retained_position_count=count,
                    seed=7,
                )
                self.assertEqual(
                    [(item.policy, item.retained_positions) for item in selections],
                    [(item.policy, item.retained_positions) for item in expected],
                )
                self.assertEqual(summary.logical_input_bytes, 27 * 46_080)
                self.assertEqual(summary.logical_retained_bytes, count * 46_080)
                self.assertEqual(summary.logical_evicted_bytes, (27 - count) * 46_080)
                self.assertEqual(len(trace.model_input_token_ids), 27)
                self.assertEqual(len(trace.evaluation_token_ids), 8)


if __name__ == "__main__":
    unittest.main()
