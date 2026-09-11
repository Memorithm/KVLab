"""Reproducible K1 experiment-run record tying oracle, resource, latency and trace evidence.

This module does not execute a model. It validates and serializes observations
already produced by a backend so unavailable telemetry remains explicitly
unavailable and the full-cache oracle identity cannot drift silently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json

from .instrumentation import KVResourceAccounting
from .latency import KVLatencyAccounting
from .oracle import FullCacheSnapshot
from .trace import KvTrace


class RunRecordError(ValueError):
    """Raised when a run record is internally inconsistent."""


@dataclass(frozen=True, slots=True)
class ExperimentRunRecord:
    schema_version: int
    experiment_id: str
    repository_revision: str
    oracle_digest_sha256: str
    oracle_logical_bytes: int
    resources: KVResourceAccounting
    latency: KVLatencyAccounting
    trace: KvTrace

    @classmethod
    def from_observations(
        cls,
        *,
        experiment_id: str,
        repository_revision: str,
        oracle: FullCacheSnapshot,
        resources: KVResourceAccounting,
        latency: KVLatencyAccounting,
        trace: KvTrace,
    ) -> "ExperimentRunRecord":
        if not experiment_id or any(ch.isspace() for ch in experiment_id):
            raise RunRecordError("experiment_id must be a non-empty canonical token")
        if len(repository_revision) != 40 or any(ch not in "0123456789abcdef" for ch in repository_revision):
            raise RunRecordError("repository_revision must be a lowercase full Git SHA")

        # Replay verifies the immutable oracle digest before the record accepts it.
        tuple(oracle.replay())
        resources.validate()
        latency.validate()

        logical = resources.logical_cache_bytes
        if logical.value is None:
            raise RunRecordError("logical cache size must be exposed for a K1 run")
        if logical.value != float(oracle.logical_bytes):
            raise RunRecordError("resource logical-cache bytes must equal the oracle snapshot")

        return cls(
            schema_version=1,
            experiment_id=experiment_id,
            repository_revision=repository_revision,
            oracle_digest_sha256=oracle.digest_sha256,
            oracle_logical_bytes=oracle.logical_bytes,
            resources=resources,
            latency=latency,
            trace=trace,
        )

    def to_json(self) -> str:
        """Emit deterministic machine-readable evidence without inventing missing values."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
