"""Exact logical KV accounting for MHA, GQA, and MQA model topologies.

This module does not retrofit an incompatible model. It only accounts a model's
native attention topology from declared dimensions. Physical allocator overhead,
fragmentation, bandwidth, and latency remain separate measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AttentionTopology(str, Enum):
    MHA = "mha"
    GQA = "gqa"
    MQA = "mqa"


@dataclass(frozen=True, slots=True)
class KvTopology:
    query_heads: int
    kv_heads: int
    layers: int
    head_dim: int
    bytes_per_element: int

    def __post_init__(self) -> None:
        for name, value in (
            ("query_heads", self.query_heads),
            ("kv_heads", self.kv_heads),
            ("layers", self.layers),
            ("head_dim", self.head_dim),
            ("bytes_per_element", self.bytes_per_element),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.kv_heads > self.query_heads:
            raise ValueError("kv_heads cannot exceed query_heads")
        if self.query_heads % self.kv_heads != 0:
            raise ValueError("query_heads must be divisible by kv_heads")

    @property
    def topology(self) -> AttentionTopology:
        if self.kv_heads == self.query_heads:
            return AttentionTopology.MHA
        if self.kv_heads == 1:
            return AttentionTopology.MQA
        return AttentionTopology.GQA

    @property
    def query_heads_per_kv_head(self) -> int:
        return self.query_heads // self.kv_heads


@dataclass(frozen=True, slots=True)
class KvLogicalStorage:
    batch_size: int
    token_count: int
    topology: KvTopology
    logical_kv_bytes: int
    mha_reference_bytes: int

    @property
    def logical_fraction_of_mha(self) -> float:
        if self.mha_reference_bytes == 0:
            return 0.0
        return self.logical_kv_bytes / self.mha_reference_bytes

    @property
    def logical_bytes_avoided_vs_mha(self) -> int:
        return self.mha_reference_bytes - self.logical_kv_bytes


def account_logical_kv_storage(
    *, batch_size: int, token_count: int, topology: KvTopology
) -> KvLogicalStorage:
    """Return exact logical K+V bytes for a native attention topology.

    The MHA reference keeps all other declared dimensions fixed and changes only
    ``kv_heads`` to ``query_heads``. The result is a structural storage baseline,
    not a performance claim.
    """

    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if token_count < 0:
        raise ValueError("token_count must be non-negative")

    per_head = (
        batch_size
        * token_count
        * topology.layers
        * topology.head_dim
        * 2  # K and V
        * topology.bytes_per_element
    )
    logical = per_head * topology.kv_heads
    mha_reference = per_head * topology.query_heads
    return KvLogicalStorage(
        batch_size=batch_size,
        token_count=token_count,
        topology=topology,
        logical_kv_bytes=logical,
        mha_reference_bytes=mha_reference,
    )
