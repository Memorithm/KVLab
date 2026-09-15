"""Run the frozen R2 campaigns without rewriting R1 or their execution pins.

This is orchestration, not a new model executor. The existing v4 executor runs
from its declared Git revision; the repaired NNIS and locked ProspectEngine
builds run from separate detached worktrees. Only successful execution followed
by verification publishes a result. Preflight never invokes the model backend.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType
from typing import Iterator, Sequence

from .prospect_r2_publication_gate import (
    PUBLICATION_VERIFIER_REVISION, PublicationGateError, verify_staged_r2_suite,
)
from .prospect_real_model_campaign_v4 import PositionCampaignSpecV1
from .prospect_position_comparison import preflight_budget_matched_campaign
from .prospect_smollm2_r1_suite import (
    BYTES_PER_TOKEN, MODEL_ID, MODEL_REVISION, MODEL_SHA256, RUNTIME_BACKEND,
    CampaignInput, CampaignVerification, SuiteResultManifest,
    SmolLm2R1SuiteError, require_git_commit,
    require_model_artifact, read_git_file, _run,
)


SUITE_SCHEMA = "kvlab.smollm2-r2-position-suite-result/v1"
PREFLIGHT_SCHEMA = "kvlab.smollm2-r2-position-suite-preflight/v1"
PREREGISTRATION_REVISION = "216b49ae4d62ed4c4c2edfd1e88f929d0a0fd9e5"
EXECUTION_REVISION = "404577ce939093767dc75d2d67de2fe3c16fa4dc"
NNIS_REVISION = "091aabbb3e132627cf64716720aae530442d2a32"
VERIFIER_REVISION = "298acdc91682ef1d09914b6f964e8934828825c0"
TRACE_SHA256 = "3411f378fb3c7010eb94361128c019206fb47bda4271f721498d517eb07ca65f"
CAMPAIGN_ROOT = "experiments/prospect/smollm2-r2"
RETAIN_COUNTS = (7, 14, 20)
POLICIES = ("lru", "random_seeded")
CAMPAIGN_SHA256 = MappingProxyType({
    7: "d826e0ca1869b6f3134e8b34bb65db14aa034d2518bf9559810ae80f19012346",
    14: "01eeec54e02bf3f56bbd2e75175e2f04fc4593701875a21b90981b40cfa4eff5",
    20: "e09b14c8479bac98b93625f3d98f667943d8318ff769dbbbf3db989959dcba07",
})


class SmolLm2R2SuiteError(RuntimeError):
    """Invalid R2 input, tool output, or publication state."""


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def campaign_path(count: int) -> str:
    if type(count) is not int or count not in RETAIN_COUNTS:
        raise SmolLm2R2SuiteError("unsupported R2 retained count")
    return f"{CAMPAIGN_ROOT}/retain-{count:02d}-of-27.json"


def validate_campaign_payload(count: int, payload: str) -> CampaignInput:
    path = campaign_path(count)
    if hashlib.sha256(payload.encode("utf-8")).hexdigest() != CAMPAIGN_SHA256[count]:
        raise SmolLm2R2SuiteError("R2 campaign bytes differ from the frozen preregistration")
    spec = PositionCampaignSpecV1.from_canonical_json(payload)
    preflight = preflight_budget_matched_campaign(spec)
    _, trace, _ = spec.build_execution_inputs()
    if trace.sha256 != TRACE_SHA256 or preflight.retained_position_count != count:
        raise SmolLm2R2SuiteError("R2 trace or retained budget is inconsistent")
    if preflight.policies != POLICIES:
        raise SmolLm2R2SuiteError("R2 policy order is inconsistent")
    return CampaignInput(count, path, spec.sha256, trace.sha256, POLICIES, payload)


def _command(argv, *, cwd=None, env=None):
    # Keep stdout reserved for the final JSON response from this CLI.
    completed = _run(argv, cwd=cwd, env=env, capture_output=True)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    return completed


@contextmanager
def detached_worktree(repo: Path, revision: str, destination: Path) -> Iterator[Path]:
    # Unlike R1's interactive helper, capture Git's checkout banner as well.
    _command(("git", "-C", repo, "worktree", "add", "--detach", destination, revision))
    try:
        yield destination
    finally:
        try:
            _command(("git", "-C", repo, "worktree", "remove", "--force", destination))
        except SmolLm2R1SuiteError as error:
            # Do not obscure a primary execution failure or recursively delete
            # a path that Git refused to remove. The workspace is temporary.
            print(f"temporary worktree cleanup failed: {error}", file=sys.stderr)


def _build(worktree: Path, target: Path, package: str, binary: str) -> Path:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target)
    _command((
        "cargo", "+1.89.0", "build", "--locked", "--release",
        "--manifest-path", worktree / "Cargo.toml", "-p", package, "--bin", binary,
    ), cwd=worktree, env=env)
    executable = target / "release" / (binary + (".exe" if os.name == "nt" else ""))
    if not executable.is_file():
        raise SmolLm2R2SuiteError(f"build did not produce {binary}")
    return executable


def backend_argv(binary: Path, model: Path, device: int) -> tuple[str, ...]:
    return (
        str(binary), "--model", str(model), "--model-id", MODEL_ID,
        "--model-revision", MODEL_REVISION, "--tokenizer-revision", MODEL_REVISION,
        "--runtime-backend", RUNTIME_BACKEND, "--runtime-revision", NNIS_REVISION,
        "--device", str(device),
    )


def _json_object(payload: str) -> dict:
    def reject_constant(value):
        raise SmolLm2R2SuiteError(f"non-finite tool response: {value}")
    try:
        value = json.loads(payload, parse_constant=reject_constant)
    except json.JSONDecodeError as error:
        raise SmolLm2R2SuiteError("invalid verification JSON") from error
    if not isinstance(value, dict):
        raise SmolLm2R2SuiteError("verification response must be an object")
    return value


def _verify_summary(campaign: CampaignInput, summary: dict) -> None:
    if (summary.get("campaign_spec_sha256") != campaign.campaign_spec_sha256
            or summary.get("trace_sha256") != TRACE_SHA256
            or summary.get("policies") != list(POLICIES)
            or type(summary.get("record_count")) is not int
            or summary["record_count"] != len(POLICIES)):
        raise SmolLm2R2SuiteError("verified output does not match the frozen R2 campaign")
    expected = json.loads(campaign.payload)["selections"]
    observations = summary.get("observations")
    if not isinstance(observations, list) or len(observations) != len(expected):
        raise SmolLm2R2SuiteError("verified output lacks per-policy observations")
    for actual, selection in zip(observations, expected, strict=True):
        if not isinstance(actual, dict):
            raise SmolLm2R2SuiteError("invalid per-policy observation")
        if (actual.get("policy") != selection["policy"]
                or actual.get("retained_positions") != selection["retained_positions"]
                or actual.get("logical_retained_bytes") != campaign.retained_count * BYTES_PER_TOKEN
                or actual.get("logical_evicted_bytes") != (27 - campaign.retained_count) * BYTES_PER_TOKEN):
            raise SmolLm2R2SuiteError("verified output has drifted positions or byte accounting")


def _preflight(python: str, worktree: Path, prospect: Path, path: Path) -> None:
    # -E ignores inherited PYTHONPATH/PYTHONHOME; -s disables user site packages.
    # cwd selects the exact declared KVLab checkout, not the launcher's checkout.
    _command((python, "-E", "-s", "-m", "kvlab.prospect_position_comparison", path), cwd=worktree)
    _command((prospect, "verify-kv-campaign-spec", path))


def _execute(python: str, worktree: Path, nnis: Path, prospect: Path,
             campaign: CampaignInput, source: Path, stage: Path, model: Path,
             device: int, timeout: float) -> CampaignVerification:
    stem = Path(campaign.repository_path).stem
    output = stage / stem
    _command((
        python, "-E", "-s", "-m", "kvlab.prospect_real_model_campaign_v4",
        "--campaign", source, "--output-dir", output,
        "--timeout-seconds", str(timeout), "--", *backend_argv(nnis, model, device),
    ), cwd=worktree)
    summary = _json_object(_command((prospect, "verify-kv-campaign", output)).stdout)
    _verify_summary(campaign, summary)
    verification_name = f"verification-{stem}.json"
    (stage / verification_name).write_text(canonical_json(summary), encoding="utf-8")
    return CampaignVerification(
        campaign.retained_count, campaign.repository_path, stem,
        campaign.campaign_spec_sha256, campaign.trace_sha256,
        len(POLICIES), POLICIES, verification_name,
    )


@contextmanager
def _publication_lock(destination: Path) -> Iterator[None]:
    """Exclude cooperating launches; never erase an existing lock or destination.

    The parent directory must be trusted. This is not an adversarial filesystem
    sandbox or a crash-durability guarantee.
    """
    lock = destination.with_name(f".{destination.name}.r2-lock")
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise SmolLm2R2SuiteError(f"suite publication is already reserved: {lock}") from error
    try:
        if os.path.lexists(destination):
            raise SmolLm2R2SuiteError(f"output already exists: {destination}")
        yield
    finally:
        lock.rmdir()


def run_suite(*, kvlab_repo: Path, nnis_repo: Path, prospect_repo: Path,
              model_dir: Path, output_directory: Path, device_ordinal: int = 0,
              timeout_seconds: float = 300.0, python: str = sys.executable,
              preflight_only: bool = False) -> dict:
    if type(device_ordinal) is not int or not 0 <= device_ordinal <= 2**31 - 1:
        raise SmolLm2R2SuiteError("device must be a non-negative i32")
    if type(preflight_only) is not bool:
        raise SmolLm2R2SuiteError("preflight_only must be a bool")
    if type(timeout_seconds) not in (int, float):
        raise SmolLm2R2SuiteError("timeout must be numeric")
    try:
        timeout = float(timeout_seconds)
    except OverflowError as error:
        raise SmolLm2R2SuiteError("timeout exceeds finite float range") from error
    if not math.isfinite(timeout) or timeout <= 0:
        raise SmolLm2R2SuiteError("timeout must be positive and finite")
    if not isinstance(python, str) or not python.strip():
        raise SmolLm2R2SuiteError("python executable must be non-empty")
    kvlab_repo, nnis_repo, prospect_repo, model_dir = (
        path.resolve() for path in (kvlab_repo, nnis_repo, prospect_repo, model_dir)
    )
    # Do not resolve the last component: a dangling output symlink is occupied.
    output_directory = output_directory.parent.resolve() / output_directory.name
    if os.path.lexists(output_directory):
        raise SmolLm2R2SuiteError(f"output already exists: {output_directory}")
    for repo, revision in (
        (kvlab_repo, PREREGISTRATION_REVISION), (kvlab_repo, EXECUTION_REVISION),
        (nnis_repo, NNIS_REVISION), (prospect_repo, VERIFIER_REVISION),
        (prospect_repo, PUBLICATION_VERIFIER_REVISION),
    ):
        require_git_commit(repo, revision)
    require_model_artifact(model_dir)
    with tempfile.TemporaryDirectory(prefix="kvlab-smollm2-r2-") as temporary, ExitStack() as stack:
        root = Path(temporary)
        campaigns = tuple(validate_campaign_payload(count, read_git_file(
            kvlab_repo, PREREGISTRATION_REVISION, campaign_path(count),
        )) for count in RETAIN_COUNTS)
        sources = {}
        for campaign in campaigns:
            path = root / Path(campaign.repository_path).name
            path.write_text(campaign.payload, encoding="utf-8")
            sources[campaign.retained_count] = path
        worktrees = [stack.enter_context(detached_worktree(repo, revision, root / name))
                     for repo, revision, name in (
                         (kvlab_repo, EXECUTION_REVISION, "kvlab"),
                         (nnis_repo, NNIS_REVISION, "nnis"),
                         (prospect_repo, VERIFIER_REVISION, "prospect"),
                         (prospect_repo, PUBLICATION_VERIFIER_REVISION, "prospect-publication"),
                     )]
        nnis = _build(worktrees[1], root / "nnis-target", "nnis-cli", "nnis-kvlab-backend-v4")
        prospect = _build(worktrees[2], root / "prospect-target", "prospect-cli", "prospect")
        publication_verifier = _build(
            worktrees[3], root / "prospect-publication-target", "prospect-cli", "prospect",
        )
        for campaign in campaigns:
            _preflight(python, worktrees[0], prospect, sources[campaign.retained_count])
        identity = dict(
            kvlab_preregistration_revision=PREREGISTRATION_REVISION,
            kvlab_execution_revision=EXECUTION_REVISION, nnis_runtime_revision=NNIS_REVISION,
            prospect_verifier_revision=VERIFIER_REVISION, model_id=MODEL_ID,
            model_revision=MODEL_REVISION, source_model_sha256=MODEL_SHA256,
            runtime_backend=RUNTIME_BACKEND, bytes_per_token=BYTES_PER_TOKEN,
            device_ordinal=device_ordinal,
        )
        if preflight_only:
            return dict(schema=PREFLIGHT_SCHEMA, **identity, campaigns=[dict(
                retained_count=c.retained_count, campaign_path=c.repository_path,
                campaign_spec_sha256=c.campaign_spec_sha256, trace_sha256=c.trace_sha256,
                policies=list(c.policies),
            ) for c in campaigns])
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        with _publication_lock(output_directory), tempfile.TemporaryDirectory(
            prefix=f".{output_directory.name}.r2-stage-", dir=output_directory.parent,
        ) as stage_root:
            stage = Path(stage_root) / "suite"
            stage.mkdir()
            records = tuple(_execute(
                python, worktrees[0], nnis, prospect, c, sources[c.retained_count],
                stage, model_dir, device_ordinal, timeout,
            ) for c in campaigns)
            result = SuiteResultManifest(schema=SUITE_SCHEMA, campaigns=records, **identity)
            (stage / "suite-manifest.json").write_text(result.canonical_json(), encoding="utf-8")
            receipt = verify_staged_r2_suite(publication_verifier, stage, result.canonical_json())
            # A stage-consistency receipt is not a claim that publication/model execution occurred.
            # Preserve the frozen v1 result and strict file set; log this additional verifier separately.
            print(canonical_json(receipt), file=sys.stderr)
            if os.path.lexists(output_directory):
                raise SmolLm2R2SuiteError("output appeared during suite execution")
            stage.rename(output_directory)
            return asdict(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("nnis-repo", "prospect-repo", "model-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--kvlab-repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_suite(
            kvlab_repo=args.kvlab_repo, nnis_repo=args.nnis_repo, prospect_repo=args.prospect_repo,
            model_dir=args.model_dir, output_directory=args.output_dir, device_ordinal=args.device,
            timeout_seconds=args.timeout_seconds, python=args.python, preflight_only=args.preflight_only,
        )
    except (SmolLm2R2SuiteError, SmolLm2R1SuiteError, PublicationGateError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
