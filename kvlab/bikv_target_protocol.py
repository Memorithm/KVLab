"""Frozen target-host/model protocol for BIKV qualification.

This module records the identities and measurement obligations that must be
fixed before a target-host/model BIKV campaign is executed.  A valid protocol
is not a result and does not authorize BKV-K9/NBKV promotion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Mapping


BKV_TARGET_PROTOCOL_SCHEMA_V1 = "kvlab.bikv-target-protocol.v1"
BASELINE_FULL_CACHE_NATIVE_PREFILL = "full-cache/native-prefill"

PHASES = frozenset({"development", "validation", "confirmatory", "final"})
PARTITION_ROLES = frozenset({"development", "validation", "protected_holdout"})
TIMING_SOURCES = frozenset({"host_wall_clock", "device_timestamp"})
BYTE_EVIDENCE_KINDS = frozenset(
    {
        "logical_packed_payload",
        "host_observed_transfer",
        "device_observed_transfer",
        "physical_dram_counter",
    }
)

REQUIRED_METRICS = (
    # Selection/correctness/quality.
    "candidate_density",
    "candidate_recall",
    "candidate_false_negative_rate",
    "o_error",
    "lse_error",
    "downstream_quality",
    "reset_reuse_correctness",
    # Boolean representation and numerical-KV accounting.
    "boolean_bits_per_token",
    "boolean_bits_per_page",
    "boolean_index_bytes",
    "boolean_metadata_overhead_bytes",
    "numerical_kv_bytes_touched",
    "numerical_kv_bytes_avoided",
    "boolean_kv_bytes_read",
    "host_device_transfer_bytes",
    "numa_traffic_bytes_when_measurable",
    "fragmentation_bytes",
    "allocator_overhead_bytes",
    # BKV-K6 cooperative-pipeline transfer/synchronization surface.
    "query_signature_transfer_bytes",
    "candidate_bitmap_transfer_bytes",
    "synchronization_wait_ns",
    "dispatch_count",
    "backpressure_wait_ns",
    # Performance surface.
    "boolean_search_latency_ns",
    "boolean_frontend_ns",
    "first_token_latency_ns",
    "tpot_ns_per_token",
    "tokens_per_second",
    "pages_per_second",
    "bits_compared_per_second",
    "effective_bandwidth_bytes_per_second",
    "scaling_efficiency",
)

_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FIELDS = frozenset(
    {
        "schema",
        "campaign_id",
        "phase",
        "hypothesis_h0",
        "hypothesis_h1",
        "evidence_bundle_sha256",
        "kvlab_commit",
        "flat_commit",
        "model_id",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "runtime_id",
        "runtime_revision",
        "hardware_fingerprint_sha256",
        "precision",
        "context_tokens",
        "batch_size",
        "dataset_id",
        "dataset_revision",
        "partition_id",
        "partition_role",
        "tuning_permitted",
        "seeds",
        "warmup_runs",
        "repetitions",
        "boolean_policy",
        "baseline_policy",
        "timing_source",
        "byte_evidence_kind",
        "quality_metric",
        "quality_rule",
        "holdout_policy",
        "required_metrics",
    }
)


class BikvTargetProtocolError(ValueError):
    """Raised when a BIKV target-host/model protocol is unsafe or incomplete."""


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise BikvTargetProtocolError(f"{name} must be non-empty text")
    if value != value.strip():
        raise BikvTargetProtocolError(f"{name} must not contain leading/trailing whitespace")
    return value


def _sha1(name: str, value: object) -> str:
    if not isinstance(value, str) or not _SHA1_RE.fullmatch(value):
        raise BikvTargetProtocolError(f"{name} must be a full lowercase 40-hex revision")
    return value


def _sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise BikvTargetProtocolError(f"{name} must be a lowercase 64-hex SHA-256")
    return value


def _enum(name: str, value: object, allowed: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise BikvTargetProtocolError(f"{name} must be one of {sorted(allowed)}")
    return value


def _positive_int(name: str, value: object, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BikvTargetProtocolError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class BikvTargetProtocolV1:
    schema: str
    campaign_id: str
    phase: str
    hypothesis_h0: str
    hypothesis_h1: str
    evidence_bundle_sha256: str
    kvlab_commit: str
    flat_commit: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    runtime_id: str
    runtime_revision: str
    hardware_fingerprint_sha256: str
    precision: str
    context_tokens: int
    batch_size: int
    dataset_id: str
    dataset_revision: str
    partition_id: str
    partition_role: str
    tuning_permitted: bool
    seeds: tuple[int, ...]
    warmup_runs: int
    repetitions: int
    boolean_policy: str
    baseline_policy: str
    timing_source: str
    byte_evidence_kind: str
    quality_metric: str
    quality_rule: str
    holdout_policy: str
    required_metrics: tuple[str, ...] = REQUIRED_METRICS

    def validate(self) -> None:
        if self.schema != BKV_TARGET_PROTOCOL_SCHEMA_V1:
            raise BikvTargetProtocolError("unsupported BIKV target protocol schema")
        _text("campaign_id", self.campaign_id)
        _enum("phase", self.phase, PHASES)
        _text("hypothesis_h0", self.hypothesis_h0)
        _text("hypothesis_h1", self.hypothesis_h1)
        _sha256("evidence_bundle_sha256", self.evidence_bundle_sha256)
        _sha1("kvlab_commit", self.kvlab_commit)
        _sha1("flat_commit", self.flat_commit)
        _text("model_id", self.model_id)
        _sha1("model_revision", self.model_revision)
        _text("tokenizer_id", self.tokenizer_id)
        _sha1("tokenizer_revision", self.tokenizer_revision)
        _text("runtime_id", self.runtime_id)
        _sha1("runtime_revision", self.runtime_revision)
        _sha256("hardware_fingerprint_sha256", self.hardware_fingerprint_sha256)
        _text("precision", self.precision)
        _positive_int("context_tokens", self.context_tokens)
        _positive_int("batch_size", self.batch_size)
        _text("dataset_id", self.dataset_id)
        _sha1("dataset_revision", self.dataset_revision)
        _text("partition_id", self.partition_id)
        _enum("partition_role", self.partition_role, PARTITION_ROLES)
        if not isinstance(self.tuning_permitted, bool):
            raise BikvTargetProtocolError("tuning_permitted must be boolean")
        if self.partition_role == "protected_holdout" and self.tuning_permitted:
            raise BikvTargetProtocolError("protected holdout cannot permit tuning")
        if self.phase in {"confirmatory", "final"} and self.tuning_permitted:
            raise BikvTargetProtocolError(f"{self.phase} phase cannot permit tuning")
        if not isinstance(self.seeds, tuple) or not self.seeds:
            raise BikvTargetProtocolError("seeds must be a non-empty tuple")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in self.seeds):
            raise BikvTargetProtocolError("seeds must contain non-negative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise BikvTargetProtocolError("seeds must not contain duplicates")
        _positive_int("warmup_runs", self.warmup_runs, minimum=0)
        _positive_int("repetitions", self.repetitions)
        _text("boolean_policy", self.boolean_policy)
        if self.baseline_policy != BASELINE_FULL_CACHE_NATIVE_PREFILL:
            raise BikvTargetProtocolError(
                f"baseline_policy must be {BASELINE_FULL_CACHE_NATIVE_PREFILL!r}"
            )
        _enum("timing_source", self.timing_source, TIMING_SOURCES)
        _enum("byte_evidence_kind", self.byte_evidence_kind, BYTE_EVIDENCE_KINDS)
        _text("quality_metric", self.quality_metric)
        _text("quality_rule", self.quality_rule)
        _text("holdout_policy", self.holdout_policy)
        if self.required_metrics != REQUIRED_METRICS:
            raise BikvTargetProtocolError(
                "required_metrics must exactly match the frozen BIKV target metric set"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["seeds"] = list(self.seeds)
        payload["required_metrics"] = list(self.required_metrics)
        return payload

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    def protocol_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BikvTargetProtocolV1":
        if not isinstance(raw, Mapping):
            raise BikvTargetProtocolError("protocol must be a JSON object")
        unknown = set(raw) - _FIELDS
        missing = _FIELDS - set(raw)
        if unknown:
            raise BikvTargetProtocolError(f"unknown protocol fields: {sorted(unknown)}")
        if missing:
            raise BikvTargetProtocolError(f"missing protocol fields: {sorted(missing)}")
        seeds = raw["seeds"]
        metrics = raw["required_metrics"]
        if not isinstance(seeds, list):
            raise BikvTargetProtocolError("seeds must be a JSON array")
        if not isinstance(metrics, list):
            raise BikvTargetProtocolError("required_metrics must be a JSON array")
        protocol = cls(
            schema=raw["schema"],
            campaign_id=raw["campaign_id"],
            phase=raw["phase"],
            hypothesis_h0=raw["hypothesis_h0"],
            hypothesis_h1=raw["hypothesis_h1"],
            evidence_bundle_sha256=raw["evidence_bundle_sha256"],
            kvlab_commit=raw["kvlab_commit"],
            flat_commit=raw["flat_commit"],
            model_id=raw["model_id"],
            model_revision=raw["model_revision"],
            tokenizer_id=raw["tokenizer_id"],
            tokenizer_revision=raw["tokenizer_revision"],
            runtime_id=raw["runtime_id"],
            runtime_revision=raw["runtime_revision"],
            hardware_fingerprint_sha256=raw["hardware_fingerprint_sha256"],
            precision=raw["precision"],
            context_tokens=raw["context_tokens"],
            batch_size=raw["batch_size"],
            dataset_id=raw["dataset_id"],
            dataset_revision=raw["dataset_revision"],
            partition_id=raw["partition_id"],
            partition_role=raw["partition_role"],
            tuning_permitted=raw["tuning_permitted"],
            seeds=tuple(seeds),
            warmup_runs=raw["warmup_runs"],
            repetitions=raw["repetitions"],
            boolean_policy=raw["boolean_policy"],
            baseline_policy=raw["baseline_policy"],
            timing_source=raw["timing_source"],
            byte_evidence_kind=raw["byte_evidence_kind"],
            quality_metric=raw["quality_metric"],
            quality_rule=raw["quality_rule"],
            holdout_policy=raw["holdout_policy"],
            required_metrics=tuple(metrics),
        )
        protocol.validate()
        return protocol

    @classmethod
    def from_canonical_json(cls, payload: str) -> "BikvTargetProtocolV1":
        if not isinstance(payload, str) or not payload:
            raise BikvTargetProtocolError("protocol JSON must be non-empty text")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise BikvTargetProtocolError("protocol must be valid JSON") from exc
        protocol = cls.from_mapping(decoded)
        if protocol.canonical_json() != payload:
            raise BikvTargetProtocolError("protocol JSON is valid but not canonical")
        return protocol
