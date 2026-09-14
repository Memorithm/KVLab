"""Execute the preregistered SmolLM2 R1 KV suite with exact repository revisions.

The suite deliberately separates three identities:

* campaign inputs are read from the immutable KVLab preregistration merge;
* campaign execution imports KVLab from the revision declared by those inputs;
* NNIS and ProspectEngine run from their own pinned revisions in detached worktrees.

This prevents a newer checkout from silently producing evidence while claiming an older
``run_repository_revision``. The suite publishes its output directory only after all
three campaigns execute and ProspectEngine verifies every self-contained evidence
bundle.

The resulting observations are quality evidence for the exact teacher-forced campaign
only. Logical KV byte accounting is not allocator release, HBM residency reduction,
avoided physical traffic, latency improvement, or throughput improvement.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Iterator, Sequence

from .prospect_position_comparison import preflight_budget_matched_campaign
from .prospect_real_model_campaign_v4 import PositionCampaignSpecV1


SUITE_RESULT_SCHEMA_V1 = "kvlab.smollm2-r1-position-suite-result/v1"
KVLAB_PREREGISTRATION_REVISION = "51f2f414c6ca3ef0260d885c72b8f5863bd66047"
KVLAB_EXECUTION_REVISION = "404577ce939093767dc75d2d67de2fe3c16fa4dc"
NNIS_RUNTIME_REVISION = "58e7db8e1c4b471a7fe82a4beba11904240c4e89"
PROSPECT_VERIFIER_REVISION = "328dfc0c2989b9cfb2dc6b251c181141844f5241"
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
MODEL_SHA256 = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
RUNTIME_BACKEND = "nnis-kvlab-v4"
BYTES_PER_TOKEN = 46_080
CAMPAIGN_RETAIN_COUNTS = (7, 14, 20)
CAMPAIGN_ROOT = "experiments/prospect/smollm2-r1"


class SmolLm2R1SuiteError(RuntimeError):
    """Raised when suite provenance, preflight, execution, or publication fails."""


@dataclass(frozen=True, slots=True)
class CampaignInput:
    retained_count: int
    repository_path: str
    campaign_spec_sha256: str
    trace_sha256: str
    policies: tuple[str, ...]
    payload: str


@dataclass(frozen=True, slots=True)
class CampaignVerification:
    retained_count: int
    campaign_path: str
    output_directory: str
    campaign_spec_sha256: str
    trace_sha256: str
    record_count: int
    policies: tuple[str, ...]
    verification_file: str


@dataclass(frozen=True, slots=True)
class SuiteResultManifest:
    schema: str
    kvlab_preregistration_revision: str
    kvlab_execution_revision: str
    nnis_runtime_revision: str
    prospect_verifier_revision: str
    model_id: str
    model_revision: str
    source_model_sha256: str
    runtime_backend: str
    bytes_per_token: int
    device_ordinal: int
    campaigns: tuple[CampaignVerification, ...]

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class SuitePreflight:
    kvlab_preregistration_revision: str
    kvlab_execution_revision: str
    nnis_runtime_revision: str
    prospect_verifier_revision: str
    model_id: str
    model_revision: str
    source_model_sha256: str
    runtime_backend: str
    bytes_per_token: int
    device_ordinal: int
    campaigns: tuple[dict[str, object], ...]

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def campaign_repository_path(retained_count: int) -> str:
    if retained_count not in CAMPAIGN_RETAIN_COUNTS:
        raise SmolLm2R1SuiteError(f"unsupported retained count {retained_count}")
    return f"{CAMPAIGN_ROOT}/retain-{retained_count:02d}-of-27.json"


def validate_campaign_payload(
    retained_count: int,
    repository_path: str,
    payload: str,
) -> CampaignInput:
    spec = PositionCampaignSpecV1.from_canonical_json(payload)
    preflight = preflight_budget_matched_campaign(spec)
    if spec.run_repository_revision != KVLAB_EXECUTION_REVISION:
        raise SmolLm2R1SuiteError("campaign KVLab execution revision drifted")
    if spec.model_id != MODEL_ID or spec.model_revision != MODEL_REVISION:
        raise SmolLm2R1SuiteError("campaign model identity drifted")
    if spec.tokenizer_revision != MODEL_REVISION:
        raise SmolLm2R1SuiteError("campaign tokenizer revision drifted")
    if spec.runtime_backend != RUNTIME_BACKEND:
        raise SmolLm2R1SuiteError("campaign runtime backend drifted")
    if spec.runtime_revision != NNIS_RUNTIME_REVISION:
        raise SmolLm2R1SuiteError("campaign NNIS runtime revision drifted")
    if spec.bytes_per_token != BYTES_PER_TOKEN:
        raise SmolLm2R1SuiteError("campaign logical bytes-per-token drifted")
    if spec.seed != 7:
        raise SmolLm2R1SuiteError("campaign seed drifted")
    if len(spec.model_input_token_ids) != 27 or len(spec.evaluation_token_ids) != 8:
        raise SmolLm2R1SuiteError("campaign trace partition drifted")
    if preflight.retained_position_count != retained_count:
        raise SmolLm2R1SuiteError("campaign retained-position budget drifted")
    if preflight.logical_retained_bytes != retained_count * BYTES_PER_TOKEN:
        raise SmolLm2R1SuiteError("campaign retained-byte budget drifted")
    expected_policies = ("lru", "random_seeded")
    if preflight.policies != expected_policies:
        raise SmolLm2R1SuiteError("campaign policy set/order drifted")
    _context, trace, _selections = spec.build_execution_inputs()
    return CampaignInput(
        retained_count=retained_count,
        repository_path=repository_path,
        campaign_spec_sha256=spec.sha256,
        trace_sha256=trace.sha256,
        policies=preflight.policies,
        payload=payload,
    )


def validate_suite_campaigns(campaigns: Sequence[CampaignInput]) -> tuple[CampaignInput, ...]:
    campaigns = tuple(campaigns)
    if tuple(campaign.retained_count for campaign in campaigns) != CAMPAIGN_RETAIN_COUNTS:
        raise SmolLm2R1SuiteError("suite campaign order/budgets drifted")
    trace_hashes = {campaign.trace_sha256 for campaign in campaigns}
    if len(trace_hashes) != 1:
        raise SmolLm2R1SuiteError("suite campaigns do not share one exact trace")
    return campaigns


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise SmolLm2R1SuiteError(f"failed to hash {path}: {error}") from error
    return digest.hexdigest()


def require_model_artifact(model_dir: Path) -> None:
    path = model_dir / "model.safetensors"
    actual = sha256_file(path)
    if actual != MODEL_SHA256:
        raise SmolLm2R1SuiteError(
            f"SmolLM2 model.safetensors SHA-256 mismatch: got {actual}, expected {MODEL_SHA256}"
        )


def _run(
    argv: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [os.fspath(part) for part in argv]
    try:
        return subprocess.run(
            command,
            cwd=None if cwd is None else str(cwd),
            env=env,
            text=True,
            capture_output=capture_output,
            check=True,
            shell=False,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        rendered = " ".join(command)
        if isinstance(error, subprocess.CalledProcessError):
            stderr = (error.stderr or "").strip()
            if stderr:
                rendered = f"{rendered}: {stderr}"
        raise SmolLm2R1SuiteError(f"command failed: {rendered}") from error


def _git(repo: Path, *arguments: str, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return _run(("git", "-C", repo, *arguments), capture_output=capture_output)


def require_git_commit(repo: Path, revision: str) -> None:
    if not repo.is_dir():
        raise SmolLm2R1SuiteError(f"repository path does not exist: {repo}")
    _git(repo, "cat-file", "-e", f"{revision}^{{commit}}")


def read_git_file(repo: Path, revision: str, repository_path: str) -> str:
    completed = _git(repo, "show", f"{revision}:{repository_path}", capture_output=True)
    return completed.stdout


@contextmanager
def detached_worktree(repo: Path, revision: str, destination: Path) -> Iterator[Path]:
    _git(repo, "worktree", "add", "--detach", str(destination), revision)
    try:
        yield destination
    finally:
        try:
            _git(repo, "worktree", "remove", "--force", str(destination))
        except SmolLm2R1SuiteError:
            shutil.rmtree(destination, ignore_errors=True)
            try:
                _git(repo, "worktree", "prune")
            except SmolLm2R1SuiteError:
                pass


def _python_env(worktree: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(worktree) if not existing else f"{worktree}{os.pathsep}{existing}"
    return env


def _binary_path(target_dir: Path, name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return target_dir / "release" / f"{name}{suffix}"


def _build_nnis(nnis_worktree: Path, target_dir: Path) -> Path:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target_dir)
    _run(
        (
            "cargo",
            "build",
            "--locked",
            "--release",
            "--manifest-path",
            nnis_worktree / "Cargo.toml",
            "-p",
            "nnis-cli",
            "--bin",
            "nnis-kvlab-backend-v4",
        ),
        env=env,
    )
    binary = _binary_path(target_dir, "nnis-kvlab-backend-v4")
    if not binary.is_file():
        raise SmolLm2R1SuiteError("NNIS KVLab v4 backend binary was not produced")
    return binary


def _build_prospect(prospect_worktree: Path, target_dir: Path) -> Path:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target_dir)
    _run(
        (
            "cargo",
            "build",
            "--locked",
            "--release",
            "--manifest-path",
            prospect_worktree / "Cargo.toml",
            "-p",
            "prospect-cli",
            "--bin",
            "prospect",
        ),
        env=env,
    )
    binary = _binary_path(target_dir, "prospect")
    if not binary.is_file():
        raise SmolLm2R1SuiteError("ProspectEngine verifier binary was not produced")
    return binary


def backend_argv(
    backend_binary: Path,
    model_dir: Path,
    *,
    device_ordinal: int,
) -> tuple[str, ...]:
    return (
        str(backend_binary),
        "--model",
        str(model_dir),
        "--model-id",
        MODEL_ID,
        "--model-revision",
        MODEL_REVISION,
        "--tokenizer-revision",
        MODEL_REVISION,
        "--runtime-backend",
        RUNTIME_BACKEND,
        "--runtime-revision",
        NNIS_RUNTIME_REVISION,
        "--device",
        str(device_ordinal),
    )


def _write_campaign_files(
    kvlab_repo: Path,
    directory: Path,
) -> tuple[CampaignInput, ...]:
    directory.mkdir(parents=True)
    campaigns: list[CampaignInput] = []
    for retained_count in CAMPAIGN_RETAIN_COUNTS:
        repository_path = campaign_repository_path(retained_count)
        payload = read_git_file(
            kvlab_repo,
            KVLAB_PREREGISTRATION_REVISION,
            repository_path,
        )
        campaign = validate_campaign_payload(retained_count, repository_path, payload)
        destination = directory / Path(repository_path).name
        destination.write_text(payload, encoding="utf-8")
        campaigns.append(campaign)
    return validate_suite_campaigns(campaigns)


def _preflight_campaign_with_pinned_tools(
    *,
    python: str,
    kvlab_worktree: Path,
    prospect_binary: Path,
    campaign_file: Path,
) -> None:
    env = _python_env(kvlab_worktree)
    _run(
        (python, "-m", "kvlab.prospect_position_comparison", campaign_file),
        cwd=kvlab_worktree,
        env=env,
    )
    _run((prospect_binary, "verify-kv-campaign-spec", campaign_file))


def _execute_campaign(
    *,
    python: str,
    kvlab_worktree: Path,
    backend_binary: Path,
    prospect_binary: Path,
    campaign: CampaignInput,
    campaign_file: Path,
    output_directory: Path,
    model_dir: Path,
    device_ordinal: int,
    timeout_seconds: float,
    verification_file: Path,
) -> CampaignVerification:
    env = _python_env(kvlab_worktree)
    argv = (
        python,
        "-m",
        "kvlab.prospect_real_model_campaign_v4",
        "--campaign",
        str(campaign_file),
        "--output-dir",
        str(output_directory),
        "--timeout-seconds",
        str(timeout_seconds),
        "--",
        *backend_argv(backend_binary, model_dir, device_ordinal=device_ordinal),
    )
    _run(argv, cwd=kvlab_worktree, env=env)
    verified = _run(
        (prospect_binary, "verify-kv-campaign", output_directory),
        capture_output=True,
    )
    try:
        summary = json.loads(verified.stdout)
    except json.JSONDecodeError as error:
        raise SmolLm2R1SuiteError("ProspectEngine returned invalid verification JSON") from error
    if summary.get("campaign_spec_sha256") != campaign.campaign_spec_sha256:
        raise SmolLm2R1SuiteError("verified campaign SHA-256 does not match preregistration")
    if summary.get("trace_sha256") != campaign.trace_sha256:
        raise SmolLm2R1SuiteError("verified trace SHA-256 does not match preregistration")
    policies = tuple(summary.get("policies", ()))
    if policies != campaign.policies:
        raise SmolLm2R1SuiteError("verified campaign policies do not match preregistration")
    record_count = summary.get("record_count")
    if type(record_count) is not int or record_count != len(campaign.policies):
        raise SmolLm2R1SuiteError("verified campaign record count is inconsistent")
    verification_file.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return CampaignVerification(
        retained_count=campaign.retained_count,
        campaign_path=campaign.repository_path,
        output_directory=output_directory.name,
        campaign_spec_sha256=campaign.campaign_spec_sha256,
        trace_sha256=campaign.trace_sha256,
        record_count=record_count,
        policies=policies,
        verification_file=verification_file.name,
    )


def run_suite(
    *,
    kvlab_repo: Path,
    nnis_repo: Path,
    prospect_repo: Path,
    model_dir: Path,
    output_directory: Path,
    device_ordinal: int = 0,
    timeout_seconds: float = 300.0,
    python: str = sys.executable,
    preflight_only: bool = False,
) -> SuitePreflight | SuiteResultManifest:
    kvlab_repo = kvlab_repo.resolve()
    nnis_repo = nnis_repo.resolve()
    prospect_repo = prospect_repo.resolve()
    model_dir = model_dir.resolve()
    output_directory = output_directory.resolve()
    if type(device_ordinal) is not int or device_ordinal < 0:
        raise SmolLm2R1SuiteError("device ordinal must be a non-negative integer")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise SmolLm2R1SuiteError("timeout must be numeric")
    if not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0:
        raise SmolLm2R1SuiteError("timeout must be positive and finite")
    if not python:
        raise SmolLm2R1SuiteError("python executable must be non-empty")

    for repo, revision in (
        (kvlab_repo, KVLAB_PREREGISTRATION_REVISION),
        (kvlab_repo, KVLAB_EXECUTION_REVISION),
        (nnis_repo, NNIS_RUNTIME_REVISION),
        (prospect_repo, PROSPECT_VERIFIER_REVISION),
    ):
        require_git_commit(repo, revision)
    require_model_artifact(model_dir)
    if not preflight_only and output_directory.exists():
        raise SmolLm2R1SuiteError(f"output directory already exists: {output_directory}")

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kvlab-smollm2-r1-suite-") as temporary:
        temporary_root = Path(temporary)
        campaign_dir = temporary_root / "campaigns"
        campaigns = _write_campaign_files(kvlab_repo, campaign_dir)

        with ExitStack() as stack:
            kvlab_worktree = stack.enter_context(
                detached_worktree(
                    kvlab_repo,
                    KVLAB_EXECUTION_REVISION,
                    temporary_root / "kvlab-execution",
                )
            )
            nnis_worktree = stack.enter_context(
                detached_worktree(
                    nnis_repo,
                    NNIS_RUNTIME_REVISION,
                    temporary_root / "nnis-runtime",
                )
            )
            prospect_worktree = stack.enter_context(
                detached_worktree(
                    prospect_repo,
                    PROSPECT_VERIFIER_REVISION,
                    temporary_root / "prospect-verifier",
                )
            )
            nnis_binary = _build_nnis(nnis_worktree, temporary_root / "nnis-target")
            prospect_binary = _build_prospect(
                prospect_worktree, temporary_root / "prospect-target"
            )
            for campaign in campaigns:
                _preflight_campaign_with_pinned_tools(
                    python=python,
                    kvlab_worktree=kvlab_worktree,
                    prospect_binary=prospect_binary,
                    campaign_file=campaign_dir / Path(campaign.repository_path).name,
                )

            if preflight_only:
                return SuitePreflight(
                    kvlab_preregistration_revision=KVLAB_PREREGISTRATION_REVISION,
                    kvlab_execution_revision=KVLAB_EXECUTION_REVISION,
                    nnis_runtime_revision=NNIS_RUNTIME_REVISION,
                    prospect_verifier_revision=PROSPECT_VERIFIER_REVISION,
                    model_id=MODEL_ID,
                    model_revision=MODEL_REVISION,
                    source_model_sha256=MODEL_SHA256,
                    runtime_backend=RUNTIME_BACKEND,
                    bytes_per_token=BYTES_PER_TOKEN,
                    device_ordinal=device_ordinal,
                    campaigns=tuple(
                        {
                            "retained_count": campaign.retained_count,
                            "campaign_path": campaign.repository_path,
                            "campaign_spec_sha256": campaign.campaign_spec_sha256,
                            "trace_sha256": campaign.trace_sha256,
                            "policies": campaign.policies,
                        }
                        for campaign in campaigns
                    ),
                )

            stage = Path(
                tempfile.mkdtemp(
                    prefix=f".{output_directory.name}.tmp-",
                    dir=str(output_directory.parent),
                )
            )
            try:
                verified_campaigns: list[CampaignVerification] = []
                for campaign in campaigns:
                    stem = f"retain-{campaign.retained_count:02d}-of-27"
                    verification = _execute_campaign(
                        python=python,
                        kvlab_worktree=kvlab_worktree,
                        backend_binary=nnis_binary,
                        prospect_binary=prospect_binary,
                        campaign=campaign,
                        campaign_file=campaign_dir / f"{stem}.json",
                        output_directory=stage / stem,
                        model_dir=model_dir,
                        device_ordinal=device_ordinal,
                        timeout_seconds=float(timeout_seconds),
                        verification_file=stage / f"verification-{stem}.json",
                    )
                    verified_campaigns.append(verification)
                result = SuiteResultManifest(
                    schema=SUITE_RESULT_SCHEMA_V1,
                    kvlab_preregistration_revision=KVLAB_PREREGISTRATION_REVISION,
                    kvlab_execution_revision=KVLAB_EXECUTION_REVISION,
                    nnis_runtime_revision=NNIS_RUNTIME_REVISION,
                    prospect_verifier_revision=PROSPECT_VERIFIER_REVISION,
                    model_id=MODEL_ID,
                    model_revision=MODEL_REVISION,
                    source_model_sha256=MODEL_SHA256,
                    runtime_backend=RUNTIME_BACKEND,
                    bytes_per_token=BYTES_PER_TOKEN,
                    device_ordinal=device_ordinal,
                    campaigns=tuple(verified_campaigns),
                )
                (stage / "suite-manifest.json").write_text(
                    result.canonical_json(), encoding="utf-8"
                )
                if output_directory.exists():
                    raise SmolLm2R1SuiteError(
                        f"output directory appeared during suite publication: {output_directory}"
                    )
                stage.rename(output_directory)
                return result
            except Exception:
                shutil.rmtree(stage, ignore_errors=True)
                raise


def _parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the three preregistered SmolLM2 R1 position-native KV campaigns "
            "through exact pinned KVLab, NNIS, and ProspectEngine revisions."
        )
    )
    parser.add_argument("--nnis-repo", type=Path, required=True)
    parser.add_argument("--prospect-repo", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kvlab-repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate commits, model digest, build pinned tools and preflight all specs without CUDA execution",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_cli(argv)
    result = run_suite(
        kvlab_repo=args.kvlab_repo,
        nnis_repo=args.nnis_repo,
        prospect_repo=args.prospect_repo,
        model_dir=args.model_dir,
        output_directory=args.output_dir,
        device_ordinal=args.device,
        timeout_seconds=args.timeout_seconds,
        python=args.python,
        preflight_only=args.preflight_only,
    )
    print(result.canonical_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
