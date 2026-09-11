import copy
import unittest

from kvlab.manifest import KVExperimentManifest, ManifestError


VALID = {
    "schema_version": 1,
    "experiment_id": "k0-manifest-contract",
    "phase": "K0",
    "hypothesis_h0": "The candidate does not improve the preregistered resource-quality tradeoff.",
    "hypothesis_h1": "The candidate improves the preregistered resource-quality tradeoff.",
    "mechanism_family": 13,
    "taxonomy_axes": ["I", "J"],
    "applicability": "applicable",
    "applicability_reason": "Source and target are declared as a matched-KV within-family pair.",
    "source_model": "source@immutable-revision",
    "target_model": "target@immutable-revision",
    "tokenizer": "shared-tokenizer@immutable-revision",
    "runtime_backend": "reference-backend@immutable-revision",
    "hardware": "cpu-structural-ci",
    "precision": "fp32-reference",
    "context_tokens": 1024,
    "batch_size": 1,
    "concurrency": 1,
    "workload": "structural-contract-only",
    "seeds": [7, 11, 19],
    "warmup_runs": 0,
    "repetitions": 3,
    "kv_policy": "cross-model ridge baseline; no experimental result in K0",
    "repository_commit": "0123456789abcdef0123456789abcdef01234567",
    "dependency_revisions": [
        "Memorithm/SciRust@89abcdef0123456789abcdef0123456789abcdef"
    ],
    "oracle_baseline": "target native prefill with unmodified full KV cache",
    "success_rule": "preregistered candidate threshold; evaluated only after calibration",
    "failure_rule": "candidate fails threshold or violates correction/quality constraints",
    "holdout_policy": "holdout is never used to choose hyperparameters",
    "resource_claims": [
        {
            "resource": "prefill_compute",
            "measurement": "measured",
            "direction": "reduce",
        },
        {
            "resource": "logical_kv_bytes",
            "measurement": "measured",
            "direction": "neutral",
        },
    ],
    "notes": ["K0 validates manifest semantics only."],
}


class ManifestTests(unittest.TestCase):
    def test_round_trip_is_deterministic(self):
        manifest = KVExperimentManifest.from_mapping(VALID)
        encoded = manifest.to_json()
        self.assertEqual(KVExperimentManifest.from_json(encoded), manifest)
        self.assertEqual(KVExperimentManifest.from_json(encoded).to_json(), encoded)

    def test_unknown_field_fails_closed(self):
        raw = copy.deepcopy(VALID)
        raw["measured_gpu_bytes"] = 123
        with self.assertRaisesRegex(ManifestError, "unknown manifest fields"):
            KVExperimentManifest.from_mapping(raw)

    def test_requires_full_commit_sha(self):
        raw = copy.deepcopy(VALID)
        raw["repository_commit"] = "deadbeef"
        with self.assertRaisesRegex(ManifestError, "full 40-character"):
            KVExperimentManifest.from_mapping(raw)

    def test_dependency_must_be_immutable(self):
        raw = copy.deepcopy(VALID)
        raw["dependency_revisions"] = ["Memorithm/SciRust@main"]
        with self.assertRaisesRegex(ManifestError, "full 40-character"):
            KVExperimentManifest.from_mapping(raw)

    def test_architectural_non_applicability_is_explicit(self):
        raw = copy.deepcopy(VALID)
        raw["applicability"] = "not_applicable"
        raw["applicability_reason"] = "Target architecture does not expose the required mechanism."
        manifest = KVExperimentManifest.from_mapping(raw)
        self.assertEqual(manifest.applicability, "not_applicable")

    def test_resource_accounting_distinguishes_logical_size_from_compute(self):
        manifest = KVExperimentManifest.from_mapping(VALID)
        claims = {claim.resource: claim.direction for claim in manifest.resource_claims}
        self.assertEqual(claims["prefill_compute"], "reduce")
        self.assertEqual(claims["logical_kv_bytes"], "neutral")

    def test_measurement_must_mark_estimates_explicitly(self):
        raw = copy.deepcopy(VALID)
        raw["resource_claims"][0]["measurement"] = "assumed"
        with self.assertRaisesRegex(ManifestError, "unsupported measurement kind"):
            KVExperimentManifest.from_mapping(raw)

    def test_duplicate_seed_is_rejected(self):
        raw = copy.deepcopy(VALID)
        raw["seeds"] = [7, 7]
        with self.assertRaisesRegex(ManifestError, "seeds must not contain duplicates"):
            KVExperimentManifest.from_mapping(raw)

    def test_taxonomy_is_bounded_to_registered_axes(self):
        raw = copy.deepcopy(VALID)
        raw["taxonomy_axes"] = ["I", "K"]
        with self.assertRaisesRegex(ManifestError, "unsupported taxonomy axes"):
            KVExperimentManifest.from_mapping(raw)


if __name__ == "__main__":
    unittest.main()
