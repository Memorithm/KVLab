"""Compatibility gating for architecture-native KV mechanisms.

K8 mechanisms must not be simulated on incompatible model families. This module
records declared architecture facts and returns an applicability decision before
an experiment is scheduled. It makes no performance claim and does not infer
features from model names.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ArchitectureMechanism(str, Enum):
    CROSS_LAYER_KV_SHARING = "cross-layer-kv-sharing"
    MLA = "mla"
    HYBRID_SSM = "hybrid-ssm"


class Applicability(str, Enum):
    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not-applicable"


@dataclass(frozen=True, slots=True)
class ModelArchitectureFeatures:
    """Explicit architecture facts taken from a pinned model revision."""

    model_id: str
    revision: str
    cross_layer_kv_sharing: bool = False
    latent_attention: bool = False
    state_space_layers: bool = False

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if not self.revision.strip():
            raise ValueError("revision must be non-empty")


@dataclass(frozen=True, slots=True)
class ArchitectureCompatibility:
    mechanism: ArchitectureMechanism
    applicability: Applicability
    reason: str

    @property
    def is_applicable(self) -> bool:
        return self.applicability is Applicability.APPLICABLE


def assess_architecture_compatibility(
    features: ModelArchitectureFeatures,
    mechanism: ArchitectureMechanism,
) -> ArchitectureCompatibility:
    """Gate a K8 mechanism using declared native architecture support only."""

    checks = {
        ArchitectureMechanism.CROSS_LAYER_KV_SHARING: (
            features.cross_layer_kv_sharing,
            "declared native cross-layer KV sharing",
        ),
        ArchitectureMechanism.MLA: (
            features.latent_attention,
            "declared native latent-attention representation",
        ),
        ArchitectureMechanism.HYBRID_SSM: (
            features.state_space_layers,
            "declared native state-space layers",
        ),
    }
    supported, fact = checks[mechanism]
    if supported:
        return ArchitectureCompatibility(
            mechanism=mechanism,
            applicability=Applicability.APPLICABLE,
            reason=fact,
        )
    return ArchitectureCompatibility(
        mechanism=mechanism,
        applicability=Applicability.NOT_APPLICABLE,
        reason=f"model revision does not declare {fact}",
    )
