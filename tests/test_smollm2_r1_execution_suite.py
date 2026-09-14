from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from kvlab.prospect_smollm2_r1_suite import (
    BYTES_PER_TOKEN,
    CAMPAIGN_RETAIN_COUNTS,
    KVLAB_EXECUTION_REVISION,
    KVLAB_PREREGISTRATION_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    NNIS_RUNTIME_REVISION,
    PROSPECT_VERIFIER_REVISION,
    CampaignInput,
    SmolLm2R1SuiteError,
    backend_argv,
    campaign_repository_path,
    sha256_file,
    validate_campaign_payload,
    validate_suite_campaigns,
)


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / "experiments" / "prospect" / "smollm2-r1"


class SmolLm2R1ExecutionSuiteTests(unittest.TestCase):
    def _campaigns(self) -> tuple[CampaignInput, ...]:
        campaigns = []
        for retained_count in CAMPAIGN_RETAIN_COUNTS:
            repository_path = campaign_repository_path(retained_count)
            payload = (CAMPAIGN_DIR / Path(repository_path).name).read_text(encoding="utf-8")
            campaigns.append(
                validate_campaign_payload(retained_count, repository_path, payload)
            )
        return validate_suite_campaigns(campaigns)

    def test_current_preregistrations_match_pinned_suite_contract(self):
        campaigns = self._campaigns()
        self.assertEqual(tuple(item.retained_count for item in campaigns), (7, 14, 20))
        self.assertEqual(len({item.trace_sha256 for item in campaigns}), 1)
        self.assertTrue(all(item.policies == ("lru", "random_seeded") for item in campaigns))
        self.assertEqual(
            KVLAB_PREREGISTRATION_REVISION,
            "51f2f414c6ca3ef0260d885c72b8f5863bd66047",
        )
        self.assertEqual(
            KVLAB_EXECUTION_REVISION,
            "404577ce939093767dc75d2d67de2fe3c16fa4dc",
        )
        self.assertEqual(
            NNIS_RUNTIME_REVISION,
            "58e7db8e1c4b471a7fe82a4beba11904240c4e89",
        )
        self.assertEqual(
            PROSPECT_VERIFIER_REVISION,
            "328dfc0c2989b9cfb2dc6b251c181141844f5241",
        )
        self.assertEqual(BYTES_PER_TOKEN, 46_080)

    def test_campaign_drift_fails_closed(self):
        path = CAMPAIGN_DIR / "retain-07-of-27.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["runtime_revision"] = "0" * 40
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        with self.assertRaisesRegex(SmolLm2R1SuiteError, "runtime revision drifted"):
            validate_campaign_payload(7, campaign_repository_path(7), payload)

    def test_suite_requires_exact_budget_order_and_one_trace(self):
        campaigns = self._campaigns()
        with self.assertRaisesRegex(SmolLm2R1SuiteError, "order/budgets drifted"):
            validate_suite_campaigns((campaigns[1], campaigns[0], campaigns[2]))
        drifted = replace(campaigns[2], trace_sha256="0" * 64)
        with self.assertRaisesRegex(SmolLm2R1SuiteError, "one exact trace"):
            validate_suite_campaigns((campaigns[0], campaigns[1], drifted))

    def test_backend_argv_binds_exact_model_and_runtime(self):
        argv = backend_argv(
            Path("/tmp/nnis-kvlab-backend-v4"),
            Path("/models/smollm2"),
            device_ordinal=2,
        )
        self.assertEqual(argv[0], "/tmp/nnis-kvlab-backend-v4")
        self.assertEqual(argv[argv.index("--model-id") + 1], MODEL_ID)
        self.assertEqual(argv[argv.index("--model-revision") + 1], MODEL_REVISION)
        self.assertEqual(argv[argv.index("--tokenizer-revision") + 1], MODEL_REVISION)
        self.assertEqual(argv[argv.index("--runtime-revision") + 1], NNIS_RUNTIME_REVISION)
        self.assertEqual(argv[argv.index("--device") + 1], "2")

    def test_sha256_file_streams_exact_bytes(self):
        payload = b"smollm2-suite-test" * 1000
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.bin"
            path.write_bytes(payload)
            self.assertEqual(sha256_file(path), hashlib.sha256(payload).hexdigest())

    def test_unknown_retained_count_is_rejected(self):
        with self.assertRaisesRegex(SmolLm2R1SuiteError, "unsupported retained count"):
            campaign_repository_path(8)


if __name__ == "__main__":
    unittest.main()
