"""Canonical BKV-K8 binding of first-token evidence to a complete target campaign.

The receipt is provenance-only.  It proves that one first-token observation binds
to one exact completed candidate target run and that the run is one member of a
complete, content-addressed target campaign rebuilt from the retained run payloads.
It does not aggregate campaign outcomes or decide BKV-K9/BKV-K11 promotion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping

from .bikv_evidence_bundle import BikvEvidenceBundleV1
from .bikv_first_token import BikvFirstTokenObservationError
from .bikv_first_token_v2 import BikvK8FirstTokenObservationV2
from .bikv_target_campaign import (
    BikvTargetCampaignError,
    BikvTargetCampaignV1,
    BikvTargetRunRefV1,
)
from .bikv_target_protocol import BikvTargetProtocolV1
from .bikv_target_run import BikvTargetRunV1


BKV_K8_CAMPAIGN_BINDING_SCHEMA_V1 = "kvlab.bkv-k8-first-token-campaign-binding.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FIELDS = frozenset(
    {
        "schema",
        "evidence_bundle_sha256",
        "protocol_sha256",
        "campaign_sha256",
        "run_sha256",
        "observation_sha256",
        "attempt_id",
        "seed",
        "repetition_index",
    }
)


class BikvK8CampaignBindingError(ValueError):
    """Raised when K8 evidence cannot be bound to an exact retained campaign."""


def _sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise BikvK8CampaignBindingError(f"{name} must be lowercase 64-hex")
    return value


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise BikvK8CampaignBindingError(f"{name} must be non-empty trimmed text")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BikvK8CampaignBindingError(f"{name} must be a non-negative integer")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BikvK8CampaignBindingError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class BikvK8FirstTokenCampaignBindingV1:
    """Content-addressed link from a K8 observation to one exact campaign slot."""

    schema: str
    evidence_bundle_sha256: str
    protocol_sha256: str
    campaign_sha256: str
    run_sha256: str
    observation_sha256: str
    attempt_id: str
    seed: int
    repetition_index: int

    @classmethod
    def from_evidence(
        cls,
        *,
        evidence_bundle: BikvEvidenceBundleV1,
        protocol: BikvTargetProtocolV1,
        campaign: BikvTargetCampaignV1,
        retained_runs: Iterable[BikvTargetRunV1],
        candidate_run: BikvTargetRunV1,
        observation: BikvK8FirstTokenObservationV2,
    ) -> "BikvK8FirstTokenCampaignBindingV1":
        """Validate and bind one candidate first-token observation to a campaign.

        `retained_runs` must reproduce the complete campaign manifest exactly;
        supplying only the selected candidate run is therefore insufficient.
        Failed sibling attempts remain valid retained evidence and are not hidden.
        """

        try:
            observation.validate_against_target_run(
                evidence_bundle=evidence_bundle,
                protocol=protocol,
                run=candidate_run,
            )
        except BikvFirstTokenObservationError as exc:
            raise BikvK8CampaignBindingError(str(exc)) from exc

        materialized_runs = tuple(retained_runs)
        try:
            campaign.verify_runs(protocol=protocol, runs=materialized_runs)
        except BikvTargetCampaignError as exc:
            raise BikvK8CampaignBindingError(str(exc)) from exc

        expected_ref = BikvTargetRunRefV1.from_run(candidate_run)
        if expected_ref not in campaign.runs:
            raise BikvK8CampaignBindingError(
                "candidate target run is not retained in the supplied campaign manifest"
            )

        binding = cls(
            schema=BKV_K8_CAMPAIGN_BINDING_SCHEMA_V1,
            evidence_bundle_sha256=evidence_bundle.bundle_sha256(),
            protocol_sha256=protocol.protocol_sha256(),
            campaign_sha256=campaign.campaign_sha256(),
            run_sha256=candidate_run.run_sha256(),
            observation_sha256=observation.observation_sha256(),
            attempt_id=candidate_run.attempt_id,
            seed=candidate_run.seed,
            repetition_index=candidate_run.repetition_index,
        )
        binding.validate()
        return binding

    def validate(self) -> None:
        if self.schema != BKV_K8_CAMPAIGN_BINDING_SCHEMA_V1:
            raise BikvK8CampaignBindingError("unsupported BKV-K8 campaign-binding schema")
        _sha256("evidence_bundle_sha256", self.evidence_bundle_sha256)
        _sha256("protocol_sha256", self.protocol_sha256)
        _sha256("campaign_sha256", self.campaign_sha256)
        _sha256("run_sha256", self.run_sha256)
        _sha256("observation_sha256", self.observation_sha256)
        _text("attempt_id", self.attempt_id)
        _nonnegative_int("seed", self.seed)
        _nonnegative_int("repetition_index", self.repetition_index)

    def canonical_json(self) -> str:
        self.validate()
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def binding_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_canonical_json(cls, payload: str) -> "BikvK8FirstTokenCampaignBindingV1":
        if not isinstance(payload, str) or not payload:
            raise BikvK8CampaignBindingError("campaign-binding JSON must be non-empty text")
        try:
            raw: Any = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as exc:
            raise BikvK8CampaignBindingError("campaign-binding payload must be valid JSON") from exc
        if not isinstance(raw, Mapping) or frozenset(raw) != _FIELDS:
            raise BikvK8CampaignBindingError("campaign-binding fields do not match v1 schema")
        binding = cls(**raw)
        binding.validate()
        if binding.canonical_json() != payload:
            raise BikvK8CampaignBindingError("campaign-binding JSON is valid but not canonical")
        return binding
