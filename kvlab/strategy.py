"""Backend-neutral KV strategy contract.

Strategies propose cache actions; they do not claim resource savings or quality
improvements. Experimental runners remain responsible for measuring outcomes
against the full-cache/native-prefill oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class StrategyAction(str, Enum):
    KEEP = "keep"
    EVICT = "evict"
    MOVE = "move"
    QUANTIZE = "quantize"
    REUSE = "reuse"
    READ = "read"


@dataclass(frozen=True, slots=True)
class StrategyContext:
    token_count: int
    logical_kv_bytes: int
    available_gpu_bytes: int | None = None
    available_host_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.token_count < 0:
            raise ValueError("token_count must be non-negative")
        if self.logical_kv_bytes < 0:
            raise ValueError("logical_kv_bytes must be non-negative")
        for name, value in (
            ("available_gpu_bytes", self.available_gpu_bytes),
            ("available_host_bytes", self.available_host_bytes),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative when exposed")


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    action: StrategyAction
    token_start: int
    token_count: int
    target_tier: str | None = None
    representation: str | None = None

    def __post_init__(self) -> None:
        if self.token_start < 0:
            raise ValueError("token_start must be non-negative")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if self.action is StrategyAction.MOVE and not self.target_tier:
            raise ValueError("move decisions require target_tier")
        if self.action is not StrategyAction.MOVE and self.target_tier is not None:
            raise ValueError("target_tier is only valid for move decisions")
        if self.action is StrategyAction.QUANTIZE and not self.representation:
            raise ValueError("quantize decisions require representation")
        if self.action is not StrategyAction.QUANTIZE and self.representation is not None:
            raise ValueError("representation is only valid for quantize decisions")


@runtime_checkable
class KvStrategy(Protocol):
    """Common deterministic strategy surface used by KVLab runners."""

    @property
    def name(self) -> str:
        """Stable machine-readable strategy identifier."""

    def decide(self, context: StrategyContext) -> tuple[StrategyDecision, ...]:
        """Return proposed actions for the current observed context."""


def validate_decisions(
    context: StrategyContext, decisions: tuple[StrategyDecision, ...]
) -> None:
    """Fail closed when a strategy addresses tokens outside current context."""

    for decision in decisions:
        end = decision.token_start + decision.token_count
        if end > context.token_count:
            raise ValueError("strategy decision exceeds observed token range")
