"""Verify a staged R2 suite with the pinned Rust consumer before publication.

The returned receipt identifies a consistency check, not a model execution or
publication. Keep it outside the frozen v1 suite file set (for example in the
launch log). The binary and directories must remain trusted and unmodified.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import stat
import subprocess
from typing import Any

# Distinct from the historical per-campaign verifier recorded in the v1 manifest.
PUBLICATION_VERIFIER_REVISION = "ca9685cd98f3a0a23e8c4f7e368736bb3aa28d0c"
VERIFICATION_SCHEMA = "prospect.kv-campaign-suite-r2-verification/v1"
RECEIPT_SCHEMA = "kvlab.r2-publication-gate-receipt/v1"
MAX_MANIFEST_BYTES = 65_536
MAX_SUMMARY_BYTES = 1_048_576


class PublicationGateError(RuntimeError):
    """The global consistency check failed; the staged suite must not publish."""


def _json_object(payload: bytes) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"non-finite number: {value}")

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
                           parse_constant=nonfinite)
        # Also rejects exponent overflow (1e999) nested anywhere in the response.
        json.dumps(value, allow_nan=False)
    except (ValueError, RecursionError) as error:
        raise PublicationGateError("invalid global verification JSON") from error
    if not isinstance(value, dict):
        raise PublicationGateError("global verification JSON must be an object")
    return value


def verify_staged_r2_suite(binary: Path, stage: Path, manifest_json: str,
                           *, timeout_seconds: float = 120.0) -> dict[str, Any]:
    """Require the actual global verifier to accept the exact staged manifest.

    Subprocess output is size-checked after capture, not a hard child-memory
    sandbox. Failure, malformed output, timeout or identity drift raises
    PublicationGateError. This function never publishes or edits suite files.
    """
    if type(timeout_seconds) not in (int, float):
        raise PublicationGateError("verification timeout must be numeric")
    try:
        timeout = float(timeout_seconds)
    except OverflowError as error:
        raise PublicationGateError("verification timeout overflows") from error
    if not math.isfinite(timeout) or timeout <= 0:
        raise PublicationGateError("verification timeout must be positive and finite")
    payload = manifest_json.encode("utf-8")
    if len(payload) > MAX_MANIFEST_BYTES:
        raise PublicationGateError("suite manifest exceeds the gate size limit")
    manifest = _json_object(payload)
    manifest_sha = hashlib.sha256(payload).hexdigest()
    path = stage / "suite-manifest.json"
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise PublicationGateError("staged manifest must be a regular file")
        with path.open("rb") as stream:
            if stream.read(MAX_MANIFEST_BYTES + 1) != payload:
                raise PublicationGateError("staged manifest differs from the intended result")
        with binary.open("rb") as stream:
            binary_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        completed = subprocess.run(
            [str(binary), "verify-kv-campaign-suite-r2", str(stage)],
            stdin=subprocess.DEVNULL, capture_output=True, check=False,
            timeout=timeout, shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PublicationGateError("global suite verifier could not complete") from error
    if completed.returncode != 0:
        detail = completed.stderr[:4096].decode("utf-8", errors="replace").strip()
        raise PublicationGateError(f"global suite verification failed ({completed.returncode}): {detail}")
    if len(completed.stdout) > MAX_SUMMARY_BYTES:
        raise PublicationGateError("global verification summary exceeds the gate size limit")
    summary = _json_object(completed.stdout)
    try:
        expected = {
            "schema": VERIFICATION_SCHEMA,
            "suite_manifest_sha256": manifest_sha,
            "kvlab_preregistration_revision": manifest["kvlab_preregistration_revision"],
            "kvlab_execution_revision": manifest["kvlab_execution_revision"],
            "nnis_runtime_revision": manifest["nnis_runtime_revision"],
            "prospect_launch_verifier_revision": manifest["prospect_verifier_revision"],
            "model_id": manifest["model_id"], "model_revision": manifest["model_revision"],
            "source_model_sha256": manifest["source_model_sha256"],
            "runtime_backend": manifest["runtime_backend"],
            "bytes_per_token": manifest["bytes_per_token"],
        }
        for field, value in expected.items():
            if type(summary.get(field)) is not type(value) or summary[field] != value:
                raise PublicationGateError(f"global verification identity mismatch: {field}")
        entries = manifest["campaigns"]
        campaigns = summary["campaigns"]
        if not isinstance(campaigns, list) or len(campaigns) != len(entries):
            raise PublicationGateError("global verification campaign count mismatch")
        for actual, entry in zip(campaigns, entries, strict=True):
            if (type(actual["retained_count"]) is not int
                    or actual["retained_count"] != entry["retained_count"]
                    or actual["campaign"]["campaign_spec_sha256"] != entry["campaign_spec_sha256"]
                    or actual["campaign"]["trace_sha256"] != entry["trace_sha256"]
                    or summary["trace_sha256"] != entry["trace_sha256"]):
                raise PublicationGateError("global verification campaign identity mismatch")
    except (KeyError, TypeError, ValueError) as error:
        raise PublicationGateError("global verification summary has invalid structure") from error
    return {
        "schema": RECEIPT_SCHEMA,
        "phase": "stage_verified",
        "evidence_kind": "consistency_check_only",
        "publication_verifier_revision": PUBLICATION_VERIFIER_REVISION,
        "verifier_binary_sha256": binary_sha,
        "suite_manifest_sha256": manifest_sha,
        "verification_stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
    }
