"""Versioned, fail-closed experiment manifests for KVLab.

The manifest records experimental intent and provenance before execution.  It
never turns estimates into measurements and it never infers architectural
compatibility from a model name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{40}$")

TAXONOMY_AXES = frozenset("ABCDEFGHIJ")
MECHANISM_FAMILIES = frozenset(range(1, 14))
RESOURCE_KINDS = frozenset(
    {
        "logical_kv_bytes",
        "resident_gpu_bytes",
        "host_ram_bytes",
        "secondary_storage_bytes",
        "gpu_cpu_transfer_bytes",
        "kv_read_bytes",
        "kv_write_bytes",
        "fragmentation_bytes",
        "duplicate_bytes",
        "prefill_compute",
        "decode_compute",
        "recomputation",
    }
)
APPLICABILITY = frozenset({"applicable", "not_applicable"})
MEASUREMENT_KINDS = frozenset({"measured", "estimated", "not_exposed"})


class ManifestError(ValueError):
    """Raised when a manifest is incomplete, ambiguous, or unsafe to execute."""


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{name} must be a non-empty string")
    return value.strip()


def _require_sha(name: str, value: object) -> str:
    text = _require_text(name, value).lower()
    if not _SHA256_RE.fullmatch(text):
        raise ManifestError(f"{name} must be a full 40-character Git commit SHA")
    return text


def _string_tuple(name: str, values: object, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ManifestError(f"{name} must be a list")
    normalized = tuple(_require_text(f"{name}[]", value) for value in values)
    if not allow_empty and not normalized:
        raise ManifestError(f"{name} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ManifestError(f"{name} must not contain duplicates")
    return normalized


@dataclass(frozen=True)
class ResourceClaim:
    """One declared resource effect; values are recorded later by result artefacts."""

    resource: str
    measurement: str
    direction: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ResourceClaim":
        allowed = {"resource", "measurement", "direction"}
        unknown = set(raw) - allowed
        if unknown:
            raise ManifestError(f"unknown resource claim fields: {sorted(unknown)}")
        resource = _require_text("resource", raw.get("resource"))
        measurement = _require_text("measurement", raw.get("measurement"))
        direction = _require_text("direction", raw.get("direction"))
        if resource not in RESOURCE_KINDS:
            raise ManifestError(f"unsupported resource kind: {resource}")
        if measurement not in MEASUREMENT_KINDS:
            raise ManifestError(f"unsupported measurement kind: {measurement}")
        if direction not in {"reduce", "increase", "neutral", "unknown"}:
            raise ManifestError(f"unsupported resource direction: {direction}")
        return cls(resource=resource, measurement=measurement, direction=direction)

    def to_dict(self) -> dict[str, str]:
        return {
            "resource": self.resource,
            "measurement": self.measurement,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class KVExperimentManifest:
    experiment_id: str
    phase: str
    hypothesis_h0: str
    hypothesis_h1: str
    mechanism_family: int
    taxonomy_axes: tuple[str, ...]
    applicability: str
    applicability_reason: str
    source_model: str
    target_model: str
    tokenizer: str
    runtime_backend: str
    hardware: str
    precision: str
    context_tokens: int
    batch_size: int
    concurrency: int
    workload: str
    seeds: tuple[int, ...]
    warmup_runs: int
    repetitions: int
    kv_policy: str
    repository_commit: str
    dependency_revisions: tuple[str, ...]
    oracle_baseline: str
    success_rule: str
    failure_rule: str
    holdout_policy: str
    resource_claims: tuple[ResourceClaim, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "KVExperimentManifest":
        if not isinstance(raw, Mapping):
            raise ManifestError("manifest must be an object")

        fields = {
            "schema_version",
            "experiment_id",
            "phase",
            "hypothesis_h0",
            "hypothesis_h1",
            "mechanism_family",
            "taxonomy_axes",
            "applicability",
            "applicability_reason",
            "source_model",
            "target_model",
            "tokenizer",
            "runtime_backend",
            "hardware",
            "precision",
            "context_tokens",
            "batch_size",
            "concurrency",
            "workload",
            "seeds",
            "warmup_runs",
            "repetitions",
            "kv_policy",
            "repository_commit",
            "dependency_revisions",
            "oracle_baseline",
            "success_rule",
            "failure_rule",
            "holdout_policy",
            "resource_claims",
            "notes",
        }
        unknown = set(raw) - fields
        if unknown:
            raise ManifestError(f"unknown manifest fields: {sorted(unknown)}")
        missing = fields - {"notes"} - set(raw)
        if missing:
            raise ManifestError(f"missing manifest fields: {sorted(missing)}")

        schema_version = raw.get("schema_version")
        if schema_version != SCHEMA_VERSION:
            raise ManifestError(f"unsupported schema_version: {schema_version!r}")

        mechanism_family = raw.get("mechanism_family")
        if not isinstance(mechanism_family, int) or isinstance(mechanism_family, bool):
            raise ManifestError("mechanism_family must be an integer")
        if mechanism_family not in MECHANISM_FAMILIES:
            raise ManifestError("mechanism_family must be in 1..13")

        axes = _string_tuple("taxonomy_axes", raw.get("taxonomy_axes"))
        invalid_axes = set(axes) - TAXONOMY_AXES
        if invalid_axes:
            raise ManifestError(f"unsupported taxonomy axes: {sorted(invalid_axes)}")

        applicability = _require_text("applicability", raw.get("applicability"))
        if applicability not in APPLICABILITY:
            raise ManifestError(f"unsupported applicability: {applicability}")

        def positive_int(name: str, *, minimum: int = 1) -> int:
            value = raw.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ManifestError(f"{name} must be an integer >= {minimum}")
            return value

        seeds_raw = raw.get("seeds")
        if not isinstance(seeds_raw, (list, tuple)) or not seeds_raw:
            raise ManifestError("seeds must be a non-empty list")
        if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds_raw):
            raise ManifestError("seeds must contain non-negative integers")
        seeds = tuple(seeds_raw)
        if len(set(seeds)) != len(seeds):
            raise ManifestError("seeds must not contain duplicates")

        deps = _string_tuple("dependency_revisions", raw.get("dependency_revisions"), allow_empty=True)
        for dependency in deps:
            if "@" not in dependency:
                raise ManifestError(
                    "dependency_revisions entries must bind a dependency to an immutable revision with '@'"
                )
            _, revision = dependency.rsplit("@", 1)
            _require_sha("dependency revision", revision)

        claims_raw = raw.get("resource_claims")
        if not isinstance(claims_raw, (list, tuple)) or not claims_raw:
            raise ManifestError("resource_claims must be a non-empty list")
        claims = tuple(ResourceClaim.from_mapping(item) for item in claims_raw)
        resources = [claim.resource for claim in claims]
        if len(set(resources)) != len(resources):
            raise ManifestError("resource_claims must not repeat a resource")

        notes = _string_tuple("notes", raw.get("notes", []), allow_empty=True)

        return cls(
            schema_version=SCHEMA_VERSION,
            experiment_id=_require_text("experiment_id", raw.get("experiment_id")),
            phase=_require_text("phase", raw.get("phase")),
            hypothesis_h0=_require_text("hypothesis_h0", raw.get("hypothesis_h0")),
            hypothesis_h1=_require_text("hypothesis_h1", raw.get("hypothesis_h1")),
            mechanism_family=mechanism_family,
            taxonomy_axes=axes,
            applicability=applicability,
            applicability_reason=_require_text("applicability_reason", raw.get("applicability_reason")),
            source_model=_require_text("source_model", raw.get("source_model")),
            target_model=_require_text("target_model", raw.get("target_model")),
            tokenizer=_require_text("tokenizer", raw.get("tokenizer")),
            runtime_backend=_require_text("runtime_backend", raw.get("runtime_backend")),
            hardware=_require_text("hardware", raw.get("hardware")),
            precision=_require_text("precision", raw.get("precision")),
            context_tokens=positive_int("context_tokens"),
            batch_size=positive_int("batch_size"),
            concurrency=positive_int("concurrency"),
            workload=_require_text("workload", raw.get("workload")),
            seeds=seeds,
            warmup_runs=positive_int("warmup_runs", minimum=0),
            repetitions=positive_int("repetitions"),
            kv_policy=_require_text("kv_policy", raw.get("kv_policy")),
            repository_commit=_require_sha("repository_commit", raw.get("repository_commit")),
            dependency_revisions=deps,
            oracle_baseline=_require_text("oracle_baseline", raw.get("oracle_baseline")),
            success_rule=_require_text("success_rule", raw.get("success_rule")),
            failure_rule=_require_text("failure_rule", raw.get("failure_rule")),
            holdout_policy=_require_text("holdout_policy", raw.get("holdout_policy")),
            resource_claims=claims,
            notes=notes,
        )

    @classmethod
    def from_json(cls, payload: str) -> "KVExperimentManifest":
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"invalid manifest JSON: {exc.msg}") from exc
        return cls.from_mapping(decoded)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "phase": self.phase,
            "hypothesis_h0": self.hypothesis_h0,
            "hypothesis_h1": self.hypothesis_h1,
            "mechanism_family": self.mechanism_family,
            "taxonomy_axes": list(self.taxonomy_axes),
            "applicability": self.applicability,
            "applicability_reason": self.applicability_reason,
            "source_model": self.source_model,
            "target_model": self.target_model,
            "tokenizer": self.tokenizer,
            "runtime_backend": self.runtime_backend,
            "hardware": self.hardware,
            "precision": self.precision,
            "context_tokens": self.context_tokens,
            "batch_size": self.batch_size,
            "concurrency": self.concurrency,
            "workload": self.workload,
            "seeds": list(self.seeds),
            "warmup_runs": self.warmup_runs,
            "repetitions": self.repetitions,
            "kv_policy": self.kv_policy,
            "repository_commit": self.repository_commit,
            "dependency_revisions": list(self.dependency_revisions),
            "oracle_baseline": self.oracle_baseline,
            "success_rule": self.success_rule,
            "failure_rule": self.failure_rule,
            "holdout_policy": self.holdout_policy,
            "resource_claims": [claim.to_dict() for claim in self.resource_claims],
            "notes": list(self.notes),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
