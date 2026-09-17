"""Canonical completeness manifest for a frozen BIKV target campaign.

The campaign manifest binds the exact retained baseline/candidate attempt records
for every frozen seed and measured repetition in :mod:`kvlab.bikv_target_run`.
It deliberately does not aggregate performance, choose a winner, open a holdout,
or turn unavailable/failed measurements into success.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping

from .bikv_target_protocol import BikvTargetProtocolV1
from .bikv_target_run import BikvTargetRunV1


BKV_TARGET_CAMPAIGN_SCHEMA_V1 = "kvlab.bikv-target-campaign.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CAMPAIGN_FIELDS = frozenset({"schema", "protocol_sha256", "campaign_id", "runs"})
_RUN_REF_FIELDS = frozenset(
    {
        "variant",
        "seed",
        "repetition_index",
        "attempt_id",
        "status",
        "run_sha256",
    }
)
_VARIANT_ORDER = {"baseline": 0, "candidate": 1}


class BikvTargetCampaignError(ValueError):
    """Raised when a target campaign is incomplete or provenance-inconsistent."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BikvTargetCampaignError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class BikvTargetRunRefV1:
    """Content-addressed identity of one retained target attempt."""

    variant: str
    seed: int
    repetition_index: int
    attempt_id: str
    status: str
    run_sha256: str

    @classmethod
    def from_run(cls, run: BikvTargetRunV1) -> "BikvTargetRunRefV1":
        run.validate()
        return cls(
            variant=run.variant,
            seed=run.seed,
            repetition_index=run.repetition_index,
            attempt_id=run.attempt_id,
            status=run.status,
            run_sha256=run.run_sha256(),
        )

    def validate(self) -> None:
        if not isinstance(self.variant, str) or self.variant not in _VARIANT_ORDER:
            raise BikvTargetCampaignError("run variant must be baseline or candidate")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise BikvTargetCampaignError("run seed must be a non-negative integer")
        if (
            isinstance(self.repetition_index, bool)
            or not isinstance(self.repetition_index, int)
            or self.repetition_index < 0
        ):
            raise BikvTargetCampaignError(
                "run repetition_index must be a non-negative integer"
            )
        if (
            not isinstance(self.attempt_id, str)
            or not self.attempt_id
            or self.attempt_id != self.attempt_id.strip()
        ):
            raise BikvTargetCampaignError("run attempt_id must be non-empty trimmed text")
        if not isinstance(self.status, str) or self.status not in {"completed", "failed"}:
            raise BikvTargetCampaignError("run status must be completed or failed")
        if not isinstance(self.run_sha256, str) or not _SHA256_RE.fullmatch(
            self.run_sha256
        ):
            raise BikvTargetCampaignError("run_sha256 must be lowercase 64-hex")

    def key(self) -> tuple[int, int, int]:
        self.validate()
        return (self.seed, self.repetition_index, _VARIANT_ORDER[self.variant])


@dataclass(frozen=True, slots=True)
class BikvTargetCampaignV1:
    """Canonical complete pairing of frozen baseline/candidate target attempts."""

    schema: str
    protocol_sha256: str
    campaign_id: str
    runs: tuple[BikvTargetRunRefV1, ...]

    @classmethod
    def from_runs(
        cls,
        *,
        protocol: BikvTargetProtocolV1,
        runs: Iterable[BikvTargetRunV1],
    ) -> "BikvTargetCampaignV1":
        protocol.validate()
        materialized = tuple(runs)
        for run in materialized:
            if not isinstance(run, BikvTargetRunV1):
                raise BikvTargetCampaignError("runs must contain BikvTargetRunV1 values")
            run.validate_against(protocol)

        refs = tuple(
            sorted(
                (BikvTargetRunRefV1.from_run(run) for run in materialized),
                key=BikvTargetRunRefV1.key,
            )
        )
        campaign = cls(
            schema=BKV_TARGET_CAMPAIGN_SCHEMA_V1,
            protocol_sha256=protocol.protocol_sha256(),
            campaign_id=protocol.campaign_id,
            runs=refs,
        )
        campaign.validate_against(protocol)
        return campaign

    def validate(self) -> None:
        if self.schema != BKV_TARGET_CAMPAIGN_SCHEMA_V1:
            raise BikvTargetCampaignError("unsupported BIKV target-campaign schema")
        if not isinstance(self.protocol_sha256, str) or not _SHA256_RE.fullmatch(
            self.protocol_sha256
        ):
            raise BikvTargetCampaignError("protocol_sha256 must be lowercase 64-hex")
        if (
            not isinstance(self.campaign_id, str)
            or not self.campaign_id
            or self.campaign_id != self.campaign_id.strip()
        ):
            raise BikvTargetCampaignError("campaign_id must be non-empty trimmed text")
        if not isinstance(self.runs, tuple) or not self.runs:
            raise BikvTargetCampaignError("runs must be a non-empty tuple")

        previous_key: tuple[int, int, int] | None = None
        attempts: set[str] = set()
        hashes: set[str] = set()
        logical_keys: set[tuple[str, int, int]] = set()
        for run in self.runs:
            if not isinstance(run, BikvTargetRunRefV1):
                raise BikvTargetCampaignError("runs must contain BikvTargetRunRefV1 values")
            run.validate()
            key = run.key()
            if previous_key is not None and key <= previous_key:
                raise BikvTargetCampaignError("runs must be in unique canonical order")
            logical_key = (run.variant, run.seed, run.repetition_index)
            if logical_key in logical_keys:
                raise BikvTargetCampaignError("duplicate variant/seed/repetition attempt")
            if run.attempt_id in attempts:
                raise BikvTargetCampaignError("attempt_id must be unique within campaign")
            if run.run_sha256 in hashes:
                raise BikvTargetCampaignError("one retained run cannot occupy two campaign slots")
            previous_key = key
            logical_keys.add(logical_key)
            attempts.add(run.attempt_id)
            hashes.add(run.run_sha256)

    def validate_against(self, protocol: BikvTargetProtocolV1) -> None:
        """Require exact protocol identity and complete paired measured repetitions."""

        self.validate()
        protocol.validate()
        if self.protocol_sha256 != protocol.protocol_sha256():
            raise BikvTargetCampaignError("campaign protocol_sha256 does not match protocol")
        if self.campaign_id != protocol.campaign_id:
            raise BikvTargetCampaignError("campaign_id does not match protocol")

        expected = {
            (variant, seed, repetition_index)
            for seed in protocol.seeds
            for repetition_index in range(protocol.repetitions)
            for variant in ("baseline", "candidate")
        }
        actual = {(run.variant, run.seed, run.repetition_index) for run in self.runs}
        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            raise BikvTargetCampaignError(
                f"campaign slots do not match frozen protocol: missing={len(missing)} extra={len(extra)}"
            )

    def verify_runs(
        self,
        *,
        protocol: BikvTargetProtocolV1,
        runs: Iterable[BikvTargetRunV1],
    ) -> None:
        """Fail closed unless supplied retained run payloads reproduce this manifest."""

        rebuilt = type(self).from_runs(protocol=protocol, runs=runs)
        if rebuilt != self:
            raise BikvTargetCampaignError("retained target runs do not match campaign manifest")

    def canonical_json_bytes(self) -> bytes:
        self.validate()
        payload = {
            "schema": self.schema,
            "protocol_sha256": self.protocol_sha256,
            "campaign_id": self.campaign_id,
            "runs": [asdict(run) for run in self.runs],
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    def campaign_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()

    @classmethod
    def from_canonical_json_bytes(cls, payload: bytes) -> "BikvTargetCampaignV1":
        if not isinstance(payload, bytes) or not payload:
            raise BikvTargetCampaignError("campaign payload must be non-empty bytes")
        try:
            raw: Any = json.loads(
                payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BikvTargetCampaignError("campaign payload must be UTF-8 JSON") from exc
        if not isinstance(raw, Mapping) or frozenset(raw) != _CAMPAIGN_FIELDS:
            raise BikvTargetCampaignError("campaign fields do not match v1 schema")
        raw_runs = raw["runs"]
        if not isinstance(raw_runs, list):
            raise BikvTargetCampaignError("runs must be a JSON array")
        refs: list[BikvTargetRunRefV1] = []
        for raw_run in raw_runs:
            if not isinstance(raw_run, Mapping) or frozenset(raw_run) != _RUN_REF_FIELDS:
                raise BikvTargetCampaignError("run reference fields do not match v1 schema")
            refs.append(BikvTargetRunRefV1(**raw_run))
        campaign = cls(
            schema=raw["schema"],
            protocol_sha256=raw["protocol_sha256"],
            campaign_id=raw["campaign_id"],
            runs=tuple(refs),
        )
        campaign.validate()
        if campaign.canonical_json_bytes() != payload:
            raise BikvTargetCampaignError("campaign payload is not canonical JSON")
        return campaign
