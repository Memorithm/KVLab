"""Atomic execution harness for position-native real-model KV campaigns.

A campaign specification contains scientific provenance, the exact model-token
trace, logical bytes-per-token and explicit position-selected candidates. Local
backend launch arguments are deliberately outside the scientific specification.

The harness executes the complete campaign before creating its output
directory. Only canonical, replayable observed evidence returned through the v4
backend path is persisted. Backend failure therefore leaves no partial observed
evidence directory. Logical KV bytes are not interpreted as allocator release,
HBM savings, memory-traffic reduction, latency/throughput improvement or
quality preservation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

from .prospect_real_model_runner import (
    ProspectKvRealModelRunnerError,
    RealModelRunContext,
)
from .prospect_real_model_runner_v4 import (
    ExternalJsonBackendV4,
    PositionModelEvaluationTraceV1,
    run_real_model_selection_campaign_v4,
)
from .prospect_real_model_selection_v2 import (
    PROSPECT_KV_REAL_MODEL_SELECTION_SCHEMA_V2,
    ProspectKvRealModelSelectionEvidenceV2,
)
from .prospect_selection_handoff_v2 import (
    ProspectKvSelectionHandoffV2,
    ProspectKvSelectionHandoffV2Error,
)


PROSPECT_KV_POSITION_CAMPAIGN_SCHEMA_V1 = (
    "kvlab.prospect-kv-real-model-position-campaign/v1"
)
PROSPECT_KV_POSITION_CAMPAIGN_RESULT_SCHEMA_V1 = (
    "kvlab.prospect-kv-real-model-position-campaign-result/v1"
)


class ProspectKvPositionCampaignError(RuntimeError):
    """Raised when a campaign specification or persistence step is invalid."""


@dataclass(frozen=True, slots=True)
class CampaignSelectionV1:
    policy: str
    retained_positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PositionCampaignSpecV1:
    schema: str
    experiment_id: str
    run_repository_revision: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    runtime_backend: str
    runtime_revision: str
    evaluation_id: str
    seed: int
    bytes_per_token: int
    model_input_token_ids: tuple[int, ...]
    evaluation_token_ids: tuple[int, ...]
    selections: tuple[CampaignSelectionV1, ...]

    @classmethod
    def from_canonical_json(cls, payload: str) -> "PositionCampaignSpecV1":
        try:
            raw = json.loads(payload, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ProspectKvPositionCampaignError) as error:
            raise ProspectKvPositionCampaignError("invalid campaign JSON") from error
        if not isinstance(raw, dict):
            raise ProspectKvPositionCampaignError("campaign must be a JSON object")
        if _canonical_json(raw) != payload:
            raise ProspectKvPositionCampaignError("campaign JSON must be canonical")

        expected = {
            "schema",
            "experiment_id",
            "run_repository_revision",
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "runtime_backend",
            "runtime_revision",
            "evaluation_id",
            "seed",
            "bytes_per_token",
            "model_input_token_ids",
            "evaluation_token_ids",
            "selections",
        }
        if set(raw) != expected:
            raise ProspectKvPositionCampaignError(
                "campaign fields do not match schema v1"
            )

        selections_raw = raw["selections"]
        if not isinstance(selections_raw, list) or not selections_raw:
            raise ProspectKvPositionCampaignError(
                "campaign selections must be a non-empty array"
            )
        selections = tuple(_selection_from_raw(item) for item in selections_raw)

        spec = cls(
            schema=_require_text("schema", raw["schema"]),
            experiment_id=_require_text("experiment_id", raw["experiment_id"]),
            run_repository_revision=_require_text(
                "run_repository_revision", raw["run_repository_revision"]
            ),
            model_id=_require_text("model_id", raw["model_id"]),
            model_revision=_require_text("model_revision", raw["model_revision"]),
            tokenizer_revision=_require_text(
                "tokenizer_revision", raw["tokenizer_revision"]
            ),
            runtime_backend=_require_text("runtime_backend", raw["runtime_backend"]),
            runtime_revision=_require_text(
                "runtime_revision", raw["runtime_revision"]
            ),
            evaluation_id=_require_text("evaluation_id", raw["evaluation_id"]),
            seed=_require_u64("seed", raw["seed"]),
            bytes_per_token=_require_positive_int(
                "bytes_per_token", raw["bytes_per_token"]
            ),
            model_input_token_ids=_token_ids(
                "model_input_token_ids", raw["model_input_token_ids"]
            ),
            evaluation_token_ids=_token_ids(
                "evaluation_token_ids", raw["evaluation_token_ids"]
            ),
            selections=selections,
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.schema != PROSPECT_KV_POSITION_CAMPAIGN_SCHEMA_V1:
            raise ProspectKvPositionCampaignError("unsupported campaign schema")
        for name, value in (
            ("experiment_id", self.experiment_id),
            ("model_id", self.model_id),
            ("model_revision", self.model_revision),
            ("tokenizer_revision", self.tokenizer_revision),
            ("runtime_backend", self.runtime_backend),
            ("runtime_revision", self.runtime_revision),
            ("evaluation_id", self.evaluation_id),
        ):
            _require_text(name, value)
        if not _is_lower_hex(self.run_repository_revision, 40):
            raise ProspectKvPositionCampaignError(
                "run_repository_revision must be a lowercase full Git SHA"
            )
        _require_u64("seed", self.seed)
        _require_positive_int("bytes_per_token", self.bytes_per_token)
        _token_ids("model_input_token_ids", self.model_input_token_ids)
        _token_ids("evaluation_token_ids", self.evaluation_token_ids)
        if not self.selections:
            raise ProspectKvPositionCampaignError("campaign selections must not be empty")

        seen_policies: set[str] = set()
        full_positions = tuple(range(len(self.model_input_token_ids)))
        for selection in self.selections:
            _require_text("selection.policy", selection.policy)
            if selection.policy in seen_policies:
                raise ProspectKvPositionCampaignError(
                    f"duplicate campaign policy: {selection.policy}"
                )
            seen_policies.add(selection.policy)
            try:
                handoff = ProspectKvSelectionHandoffV2.capture(
                    token_ids=self.model_input_token_ids,
                    bytes_per_token=self.bytes_per_token,
                    policy=selection.policy,
                    retained_positions=selection.retained_positions,
                )
            except ProspectKvSelectionHandoffV2Error as error:
                raise ProspectKvPositionCampaignError(
                    f"invalid selection {selection.policy!r}"
                ) from error
            if handoff.retained_positions == full_positions:
                raise ProspectKvPositionCampaignError(
                    f"selection {selection.policy!r} duplicates the full-cache baseline"
                )

    def canonical_json(self) -> str:
        self.validate()
        return _canonical_json(asdict(self))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def build_execution_inputs(
        self,
    ) -> tuple[
        RealModelRunContext,
        PositionModelEvaluationTraceV1,
        tuple[ProspectKvSelectionHandoffV2, ...],
    ]:
        self.validate()
        trace = PositionModelEvaluationTraceV1.capture(
            model_input_token_ids=self.model_input_token_ids,
            evaluation_token_ids=self.evaluation_token_ids,
        )
        context = RealModelRunContext(
            experiment_id=self.experiment_id,
            run_repository_revision=self.run_repository_revision,
            model_id=self.model_id,
            model_revision=self.model_revision,
            tokenizer_revision=self.tokenizer_revision,
            runtime_backend=self.runtime_backend,
            runtime_revision=self.runtime_revision,
            evaluation_id=self.evaluation_id,
            trace_sha256=trace.sha256,
            seed=self.seed,
        )
        selections = tuple(
            ProspectKvSelectionHandoffV2.capture(
                token_ids=self.model_input_token_ids,
                bytes_per_token=self.bytes_per_token,
                policy=selection.policy,
                retained_positions=selection.retained_positions,
            )
            for selection in self.selections
        )
        return context, trace, selections


@dataclass(frozen=True, slots=True)
class CampaignRecordManifestV1:
    index: int
    policy: str
    filename: str
    sha256: str


@dataclass(frozen=True, slots=True)
class PositionCampaignResultManifestV1:
    schema: str
    campaign_spec_sha256: str
    trace_sha256: str
    evidence_schema: str
    records: tuple[CampaignRecordManifestV1, ...]

    def canonical_json(self) -> str:
        return _canonical_json(asdict(self))


def execute_position_campaign_v4(
    *,
    spec: PositionCampaignSpecV1,
    backend: Any,
    output_dir: str | os.PathLike[str],
) -> PositionCampaignResultManifestV1:
    """Execute all backend requests first, then atomically publish evidence files."""

    spec.validate()
    destination = Path(output_dir)
    if destination.exists():
        raise ProspectKvPositionCampaignError(
            f"output directory already exists: {destination}"
        )

    context, trace, selections = spec.build_execution_inputs()
    try:
        evidence = run_real_model_selection_campaign_v4(
            context=context,
            trace=trace,
            selections=selections,
            backend=backend,
        )
    except ProspectKvRealModelRunnerError:
        raise
    except Exception as error:
        raise ProspectKvPositionCampaignError("campaign backend execution failed") from error

    if len(evidence) != len(selections):
        raise ProspectKvPositionCampaignError("campaign evidence count mismatch")

    encoded_records: list[tuple[str, str, str]] = []
    record_manifest: list[CampaignRecordManifestV1] = []
    for index, (selection, record) in enumerate(zip(selections, evidence, strict=True)):
        payload = record.canonical_json()
        replayed = ProspectKvRealModelSelectionEvidenceV2.from_canonical_json(payload)
        if replayed != record:
            raise ProspectKvPositionCampaignError(
                "generated evidence does not replay identically"
            )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        filename = f"selection-{index:03d}.json"
        encoded_records.append((filename, payload, digest))
        record_manifest.append(
            CampaignRecordManifestV1(
                index=index,
                policy=selection.policy,
                filename=filename,
                sha256=digest,
            )
        )

    manifest = PositionCampaignResultManifestV1(
        schema=PROSPECT_KV_POSITION_CAMPAIGN_RESULT_SCHEMA_V1,
        campaign_spec_sha256=spec.sha256,
        trace_sha256=trace.sha256,
        evidence_schema=PROSPECT_KV_REAL_MODEL_SELECTION_SCHEMA_V2,
        records=tuple(record_manifest),
    )
    _publish_atomically(destination, encoded_records, manifest)
    return manifest


def _publish_atomically(
    destination: Path,
    records: Sequence[tuple[str, str, str]],
    manifest: PositionCampaignResultManifestV1,
) -> None:
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(parent))
    )
    try:
        for filename, payload, _digest in records:
            (stage / filename).write_text(payload, encoding="utf-8")
        (stage / "manifest.json").write_text(manifest.canonical_json(), encoding="utf-8")
        if destination.exists():
            raise ProspectKvPositionCampaignError(
                f"output directory appeared during publication: {destination}"
            )
        stage.rename(destination)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _selection_from_raw(raw: Any) -> CampaignSelectionV1:
    if not isinstance(raw, dict) or set(raw) != {"policy", "retained_positions"}:
        raise ProspectKvPositionCampaignError(
            "selection fields must be policy and retained_positions"
        )
    policy = _require_text("selection.policy", raw["policy"])
    positions_raw = raw["retained_positions"]
    if not isinstance(positions_raw, list):
        raise ProspectKvPositionCampaignError(
            "selection.retained_positions must be an array"
        )
    positions: list[int] = []
    previous = None
    for value in positions_raw:
        if type(value) is not int or value < 0:
            raise ProspectKvPositionCampaignError(
                "selection.retained_positions must contain non-negative integers"
            )
        if previous is not None and value <= previous:
            raise ProspectKvPositionCampaignError(
                "selection.retained_positions must be strictly increasing and unique"
            )
        positions.append(value)
        previous = value
    return CampaignSelectionV1(policy=policy, retained_positions=tuple(positions))


def _token_ids(name: str, value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ProspectKvPositionCampaignError(f"{name} must be a non-empty array")
    tokens = tuple(value)
    if any(type(token) is not int or token < 0 for token in tokens):
        raise ProspectKvPositionCampaignError(
            f"{name} must contain non-negative integer token ids"
        )
    return tokens


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProspectKvPositionCampaignError(f"{name} must be a non-empty string")
    return value


def _require_u64(name: str, value: Any) -> int:
    if type(value) is not int or not 0 <= value <= (2**64 - 1):
        raise ProspectKvPositionCampaignError(
            f"{name} must be an unsigned 64-bit integer"
        )
    return value


def _require_positive_int(name: str, value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ProspectKvPositionCampaignError(f"{name} must be a positive integer")
    return value


def _is_lower_hex(value: str, length: int) -> bool:
    return len(value) == length and all(
        character.isdigit() or "a" <= character <= "f" for character in value
    )


def _reject_json_constant(value: str) -> Any:
    raise ProspectKvPositionCampaignError(f"non-finite JSON constant is forbidden: {value}")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute a canonical KVLab position-native real-model campaign."
    )
    parser.add_argument("--campaign", required=True, help="canonical campaign JSON file")
    parser.add_argument("--output-dir", required=True, help="new evidence output directory")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="per backend invocation timeout",
    )
    parser.add_argument(
        "backend_command",
        nargs=argparse.REMAINDER,
        help="backend argv after --, for example nnis-kvlab-backend-v4 --model ...",
    )
    args = parser.parse_args(argv)
    if args.backend_command and args.backend_command[0] == "--":
        args.backend_command = args.backend_command[1:]
    if not args.backend_command:
        parser.error("backend command is required after --")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive and finite")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_cli(argv)
    payload = Path(args.campaign).read_text(encoding="utf-8")
    spec = PositionCampaignSpecV1.from_canonical_json(payload)
    backend = ExternalJsonBackendV4(
        command=tuple(args.backend_command), timeout_seconds=args.timeout_seconds
    )
    manifest = execute_position_campaign_v4(
        spec=spec, backend=backend, output_dir=args.output_dir
    )
    print(manifest.canonical_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
