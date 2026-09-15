"""Gate unit tests: subprocess responses are fixtures, not model observations."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from kvlab import prospect_r2_publication_gate as gate


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def manifest_fixture():
    return dict(
        schema="kvlab.smollm2-r2-position-suite-result/v1",
        kvlab_preregistration_revision="input-revision",
        kvlab_execution_revision="executor-revision", nnis_runtime_revision="runtime-revision",
        prospect_verifier_revision="historical-verifier-revision", model_id="fixture/model",
        model_revision="model-revision", source_model_sha256="1" * 64,
        runtime_backend="fixture-backend", bytes_per_token=46080,
        campaigns=[dict(retained_count=n, campaign_spec_sha256=str(i + 2) * 64,
                        trace_sha256="7" * 64) for i, n in enumerate((7, 14, 20))],
    )


def summary_fixture(manifest):
    value = {key: item for key, item in manifest.items() if key != "campaigns"}
    value["schema"] = gate.VERIFICATION_SCHEMA
    value["suite_manifest_sha256"] = hashlib.sha256(canonical(manifest).encode()).hexdigest()
    value["prospect_launch_verifier_revision"] = value.pop("prospect_verifier_revision")
    value["trace_sha256"] = "7" * 64
    value["campaigns"] = [dict(retained_count=c["retained_count"], campaign=dict(
        campaign_spec_sha256=c["campaign_spec_sha256"], trace_sha256=c["trace_sha256"],
    )) for c in manifest["campaigns"]]
    return value


class PublicationGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gate-fixture-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.binary = self.root / "verifier with spaces"
        self.binary.write_bytes(b"not a real executable; subprocess is mocked")
        self.stage = self.root / "stage"
        self.stage.mkdir()
        self.manifest = manifest_fixture()
        self.manifest_json = canonical(self.manifest)
        (self.stage / "suite-manifest.json").write_text(self.manifest_json)
        self.summary = summary_fixture(self.manifest)

    def verify_response(self, payload, returncode=0):
        response = subprocess.CompletedProcess([], returncode, payload, b"fixture diagnostic")
        with patch.object(gate.subprocess, "run", return_value=response) as run:
            result = gate.verify_staged_r2_suite(self.binary, self.stage, self.manifest_json)
        return result, run

    def test_receipt_binds_actual_binary_manifest_and_response_bytes(self):
        payload = canonical(self.summary).encode() + b"\n"
        result, run = self.verify_response(payload)
        self.assertEqual(result["schema"], gate.RECEIPT_SCHEMA)
        self.assertEqual(result["phase"], "stage_verified")
        self.assertEqual(result["evidence_kind"], "consistency_check_only")
        self.assertEqual(result["publication_verifier_revision"], gate.PUBLICATION_VERIFIER_REVISION)
        self.assertEqual(result["verifier_binary_sha256"], hashlib.sha256(self.binary.read_bytes()).hexdigest())
        self.assertEqual(result["suite_manifest_sha256"], hashlib.sha256(self.manifest_json.encode()).hexdigest())
        self.assertEqual(result["verification_stdout_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(run.call_args.args[0], [str(self.binary), "verify-kv-campaign-suite-r2", str(self.stage)])
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertEqual(run.call_args.kwargs["timeout"], 120.0)
        self.assertEqual(run.call_args.kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(list(self.stage.iterdir()), [self.stage / "suite-manifest.json"])

    def test_nonzero_status_is_never_treated_as_success(self):
        with self.assertRaises(gate.PublicationGateError):
            self.verify_response(canonical(self.summary).encode(), returncode=1)

    def test_rejects_malformed_duplicate_nonfinite_and_wrong_type_json(self):
        for value in (b"", b"[]", b"invalid", b"\xff", b'{"a":1,"a":2}',
                      b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e999}',
                      b"[" * 1500 + b"0" + b"]" * 1500):
            with self.subTest(value=value[:40]), self.assertRaises(gate.PublicationGateError):
                self.verify_response(value)

    def test_rejects_wrong_summary_identities_even_with_exit_zero(self):
        for field in ("schema", "suite_manifest_sha256", "nnis_runtime_revision",
                      "prospect_launch_verifier_revision", "model_id", "trace_sha256", "bytes_per_token"):
            altered = dict(self.summary, **{field: "wrong"})
            with self.subTest(field=field), self.assertRaises(gate.PublicationGateError):
                self.verify_response(canonical(altered).encode())

    def test_rejects_missing_reordered_and_mistyped_campaigns(self):
        for change in ("missing", "order", "float", "digest", "shape"):
            altered = json.loads(canonical(self.summary))
            if change == "missing":
                altered["campaigns"].pop()
            elif change == "order":
                altered["campaigns"].reverse()
            elif change == "float":
                altered["campaigns"][0]["retained_count"] = 7.0
            elif change == "digest":
                altered["campaigns"][0]["campaign"]["campaign_spec_sha256"] = "0" * 64
            else:
                altered["campaigns"][0] = None
            with self.subTest(change=change), self.assertRaises(gate.PublicationGateError):
                self.verify_response(canonical(altered).encode())

    def test_rejects_changed_staged_manifest_before_process_start(self):
        (self.stage / "suite-manifest.json").write_text(self.manifest_json + "\n")
        with patch.object(gate.subprocess, "run") as run:
            with self.assertRaises(gate.PublicationGateError):
                gate.verify_staged_r2_suite(self.binary, self.stage, self.manifest_json)
            run.assert_not_called()

    def test_rejects_manifest_symlink(self):
        path = self.stage / "suite-manifest.json"
        target = self.root / "outside.json"
        path.rename(target)
        path.symlink_to(target)
        with patch.object(gate.subprocess, "run") as run:
            with self.assertRaises(gate.PublicationGateError):
                gate.verify_staged_r2_suite(self.binary, self.stage, self.manifest_json)
            run.assert_not_called()

    def test_process_failure_and_timeout_fail_closed(self):
        for error in (OSError("fixture spawn failure"), subprocess.TimeoutExpired("fixture", 120)):
            with self.subTest(error=type(error).__name__), patch.object(gate.subprocess, "run", side_effect=error):
                with self.assertRaises(gate.PublicationGateError):
                    gate.verify_staged_r2_suite(self.binary, self.stage, self.manifest_json)

    def test_bad_timeouts_are_rejected_before_execution(self):
        for timeout in (True, -1, 0, "1", float("nan"), float("inf"), 10**1000):
            with self.subTest(timeout=timeout), patch.object(gate.subprocess, "run") as run:
                with self.assertRaises(gate.PublicationGateError):
                    gate.verify_staged_r2_suite(self.binary, self.stage, self.manifest_json, timeout_seconds=timeout)
                run.assert_not_called()

    def test_summary_and_manifest_size_limits(self):
        with self.assertRaises(gate.PublicationGateError):
            self.verify_response(b" " * (gate.MAX_SUMMARY_BYTES + 1))
        with patch.object(gate.subprocess, "run") as run:
            with self.assertRaises(gate.PublicationGateError):
                gate.verify_staged_r2_suite(self.binary, self.stage, " " * (gate.MAX_MANIFEST_BYTES + 1))
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
