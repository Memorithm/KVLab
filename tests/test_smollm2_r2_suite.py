"""Orchestration tests with mocked execution; not observed model evidence."""

from contextlib import ExitStack, contextmanager, redirect_stderr
from dataclasses import asdict
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from kvlab import prospect_smollm2_r2_suite as suite
from kvlab.prospect_r2_publication_gate import PublicationGateError

ROOT = Path(__file__).resolve().parents[1]


def payload(count):
    return (ROOT / suite.campaign_path(count)).read_text(encoding="utf-8")


def summary(campaign):
    selections = json.loads(campaign.payload)["selections"]
    return {
        "campaign_spec_sha256": campaign.campaign_spec_sha256,
        "trace_sha256": suite.TRACE_SHA256,
        "policies": list(suite.POLICIES),
        "record_count": 2,
        "observations": [dict(
            policy=s["policy"], retained_positions=s["retained_positions"],
            logical_retained_bytes=campaign.retained_count * suite.BYTES_PER_TOKEN,
            logical_evicted_bytes=(27 - campaign.retained_count) * suite.BYTES_PER_TOKEN,
            metrics=[],  # Only the orchestrator is mocked, not scientific evidence.
        ) for s in selections],
    }


@contextmanager
def fake_worktree(repo, revision, destination):
    destination.mkdir()
    yield destination


class R2SuiteTests(unittest.TestCase):
    def test_exact_preregistered_bytes_and_trace(self):
        for count in suite.RETAIN_COUNTS:
            with self.subTest(count=count):
                campaign = suite.validate_campaign_payload(count, payload(count))
                self.assertEqual(campaign.campaign_spec_sha256, suite.CAMPAIGN_SHA256[count])
                self.assertEqual(campaign.trace_sha256, suite.TRACE_SHA256)
                self.assertEqual(campaign.policies, suite.POLICIES)

    def test_r1_and_coherent_substitutions_are_rejected(self):
        source = json.loads(payload(7))
        mutations = [
            {"runtime_revision": "58e7db8e1c4b471a7fe82a4beba11904240c4e89"},
            {"experiment_id": "replacement"},
            {"evaluation_id": "replacement"},
            {"model_input_token_ids": [1] * 27},
            {"selections": [{"policy": "lru", "retained_positions": list(range(7))}]},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                altered = dict(source, **mutation)
                with self.assertRaises(suite.SmolLm2R2SuiteError):
                    suite.validate_campaign_payload(7, suite.canonical_json(altered))
        with self.assertRaises(suite.SmolLm2R2SuiteError):
            suite.validate_campaign_payload(7, payload(7) + "\n")

    def test_input_digest_table_is_read_only(self):
        with self.assertRaises(TypeError):
            suite.CAMPAIGN_SHA256[7] = "0" * 64

    def test_retained_counts_are_strict(self):
        for count in (True, 7.0, "7", 0, 27):
            with self.subTest(count=count), self.assertRaises(suite.SmolLm2R2SuiteError):
                suite.campaign_path(count)

    def test_backend_argv_uses_repaired_runtime_and_preserves_path_spaces(self):
        args = suite.backend_argv(Path("/tmp/bin with spaces"), Path("/tmp/model dir"), 2)
        self.assertEqual(args[0], "/tmp/bin with spaces")
        self.assertEqual(args[args.index("--model") + 1], "/tmp/model dir")
        self.assertEqual(args[args.index("--runtime-revision") + 1], suite.NNIS_REVISION)
        self.assertEqual(args[-2:], ("--device", "2"))

    def test_locked_build_uses_worktree_cwd_toolchain_and_separate_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            binary = target / "release" / "prospect"
            binary.parent.mkdir(parents=True)
            binary.touch()
            with patch.object(suite, "_command") as command:
                self.assertEqual(suite._build(root, target, "prospect-cli", "prospect"), binary)
            args, kwargs = command.call_args
            self.assertEqual(args[0][:5], ("cargo", "+1.89.0", "build", "--locked", "--release"))
            self.assertEqual(kwargs["cwd"], root)
            self.assertEqual(kwargs["env"]["CARGO_TARGET_DIR"], str(target))

    def test_preflight_uses_isolated_python_and_never_invokes_backend(self):
        with patch.object(suite, "_command") as command:
            suite._preflight("python3", Path("/pinned"), Path("/prospect"), Path("/input.json"))
        calls = command.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].args[0][:4], ("python3", "-E", "-s", "-m"))
        self.assertEqual(calls[0].kwargs["cwd"], Path("/pinned"))
        self.assertEqual(calls[1].args[0][1], "verify-kv-campaign-spec")

    def test_verification_identity_counts_and_positions_are_checked(self):
        campaign = suite.validate_campaign_payload(7, payload(7))
        suite._verify_summary(campaign, summary(campaign))
        for field, value in (("campaign_spec_sha256", "0" * 64), ("trace_sha256", "0" * 64),
                             ("record_count", True), ("policies", ["random_seeded", "lru"])):
            with self.subTest(field=field), self.assertRaises(suite.SmolLm2R2SuiteError):
                suite._verify_summary(campaign, dict(summary(campaign), **{field: value}))
        for field, value in (("retained_positions", list(range(7))),
                             ("logical_retained_bytes", 1), ("logical_evicted_bytes", 1)):
            altered = summary(campaign)
            altered["observations"][0][field] = value
            with self.subTest(field=field), self.assertRaises(suite.SmolLm2R2SuiteError):
                suite._verify_summary(campaign, altered)

    def test_invalid_tool_json_fails_closed(self):
        for value in ("[]", "garbage", '{"x":NaN}', '{"x":Infinity}'):
            with self.subTest(value=value), self.assertRaises(suite.SmolLm2R2SuiteError):
                suite._json_object(value)

    def test_command_keeps_diagnostics_off_stdout(self):
        result = subprocess.CompletedProcess(["fixture"], 0, '{"ok":true}', "diagnostic\n")
        captured = io.StringIO()
        with patch.object(suite, "_run", return_value=result), redirect_stderr(captured):
            self.assertEqual(suite._command(("fixture",)).stdout, '{"ok":true}')
        self.assertEqual(captured.getvalue(), "diagnostic\n")

    def _inputs(self, root):
        return dict(kvlab_repo=root, nnis_repo=root, prospect_repo=root,
                    model_dir=root, output_directory=root / "published")

    def _mocks(self, stack, execute):
        stack.enter_context(patch.object(suite, "require_git_commit"))
        stack.enter_context(patch.object(suite, "require_model_artifact"))
        def read(repo, revision, path):
            self.assertEqual(revision, suite.PREREGISTRATION_REVISION)
            return (ROOT / path).read_text(encoding="utf-8")
        stack.enter_context(patch.object(suite, "read_git_file", side_effect=read))
        stack.enter_context(patch.object(suite, "detached_worktree", side_effect=fake_worktree))
        stack.enter_context(patch.object(suite, "_build", side_effect=lambda *args: Path("/fake") / args[-1]))
        preflight = stack.enter_context(patch.object(suite, "_preflight"))
        execution = stack.enter_context(patch.object(suite, "_execute", side_effect=execute))
        stack.enter_context(patch.object(suite, "verify_staged_r2_suite", create=True, return_value={}))
        return preflight, execution

    @staticmethod
    def _fake_execute(*args):
        campaign, stage = args[4], args[6]
        stem = Path(campaign.repository_path).stem
        (stage / stem).mkdir()
        verification = f"verification-{stem}.json"
        (stage / verification).write_text("{}", encoding="utf-8")
        return suite.CampaignVerification(
            campaign.retained_count, campaign.repository_path, stem,
            campaign.campaign_spec_sha256, campaign.trace_sha256, 2, suite.POLICIES, verification,
        )

    def test_preflight_only_has_no_execution_or_output_side_effects(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            preflight, execution = self._mocks(stack, self._fake_execute)
            result = suite.run_suite(**self._inputs(root), preflight_only=True)
            self.assertEqual(result["schema"], suite.PREFLIGHT_SCHEMA)
            self.assertEqual(len(result["campaigns"]), 3)
            self.assertEqual(preflight.call_count, 3)
            execution.assert_not_called()
            suite.verify_staged_r2_suite.assert_not_called()
            self.assertEqual(suite._build.call_count, 3)
            self.assertEqual(list(root.iterdir()), [])

    def test_success_publishes_manifest_only_after_three_verifications(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            _, execution = self._mocks(stack, self._fake_execute)
            result = suite.run_suite(**self._inputs(root))
            self.assertEqual(execution.call_count, 3)
            suite.verify_staged_r2_suite.assert_called_once()
            output = root / "published"
            actual = (output / "suite-manifest.json").read_text(encoding="utf-8")
            self.assertEqual(actual, suite.canonical_json(result))
            self.assertEqual(result["schema"], suite.SUITE_SCHEMA)
            self.assertEqual(result["prospect_verifier_revision"], suite.VERIFIER_REVISION)
            self.assertEqual(set(root.iterdir()), {output})

    def test_failure_or_interrupt_removes_stage_and_never_publishes(self):
        for error in (suite.SmolLm2R2SuiteError("fixture failure"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
                    root = Path(temporary)
                    def execute(*args):
                        self._fake_execute(*args)
                        if args[4].retained_count == 14:
                            raise error
                    self._mocks(stack, execute)
                    with self.assertRaises(type(error)):
                        suite.run_suite(**self._inputs(root))
                    self.assertEqual(list(root.iterdir()), [])

    def test_global_gate_rejection_prevents_publication(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            _, execution = self._mocks(stack, self._fake_execute)
            with patch.object(suite, "verify_staged_r2_suite", create=True, side_effect=PublicationGateError("cross-budget baseline drift")):
                with self.assertRaises(PublicationGateError):
                    suite.run_suite(**self._inputs(root))
            self.assertEqual(execution.call_count, 3)
            self.assertEqual(list(root.iterdir()), [])

    def test_global_gate_runs_on_complete_stage_before_rename(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            _, execution = self._mocks(stack, self._fake_execute)
            def check(binary, stage, manifest_json):
                self.assertEqual(execution.call_count, 3)
                self.assertFalse((root / "published").exists())
                self.assertEqual((stage / "suite-manifest.json").read_text(), manifest_json)
                self.assertEqual(len(list(stage.iterdir())), 7)
                self.assertNotIn("publication_verifier_revision", json.loads(manifest_json))
                return {"phase": "stage_verified", "publication_verifier_revision": suite.PUBLICATION_VERIFIER_REVISION}
            captured = io.StringIO()
            with patch.object(suite, "verify_staged_r2_suite", side_effect=check), redirect_stderr(captured):
                suite.run_suite(**self._inputs(root))
            receipt = json.loads(captured.getvalue())
            self.assertEqual(receipt["phase"], "stage_verified")
            self.assertEqual(receipt["publication_verifier_revision"], suite.PUBLICATION_VERIFIER_REVISION)
            self.assertTrue((root / "published" / "suite-manifest.json").is_file())

    def test_global_gate_interrupt_discards_staging_and_lock(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            self._mocks(stack, self._fake_execute)
            with patch.object(suite, "verify_staged_r2_suite", side_effect=KeyboardInterrupt()):
                with self.assertRaises(KeyboardInterrupt):
                    suite.run_suite(**self._inputs(root))
            self.assertEqual(list(root.iterdir()), [])

    def test_existing_output_and_lock_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = self._inputs(root)
            inputs["output_directory"].mkdir()
            with patch.object(suite, "require_git_commit") as probe:
                with self.assertRaises(suite.SmolLm2R2SuiteError):
                    suite.run_suite(**inputs)
                probe.assert_not_called()
            lock = root / ".other.r2-lock"
            lock.mkdir()
            with self.assertRaises(suite.SmolLm2R2SuiteError):
                with suite._publication_lock(root / "other"):
                    self.fail("pre-existing lock was ignored")
            self.assertTrue(lock.is_dir())

    def test_bad_scalars_fail_before_any_repository_or_model_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = self._inputs(Path(temporary))
            invalid = [dict(device_ordinal=v) for v in (True, -1, 2**31, "0")]
            invalid += [dict(timeout_seconds=v) for v in (True, 0, -1, float("nan"), float("inf"), 10**1000)]
            invalid += [dict(preflight_only=1), dict(python="  ")]
            with patch.object(suite, "require_git_commit") as probe:
                for kwargs in invalid:
                    with self.subTest(kwargs=kwargs), self.assertRaises(suite.SmolLm2R2SuiteError):
                        suite.run_suite(**base, **kwargs)
                probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
