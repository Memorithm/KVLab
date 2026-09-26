"""Quality metrics shared by KVLab experiments that feed SLHAv2.

The module deliberately separates:
- ranking/top-k structure;
- retained softmax probability mass;
- selected density.

It does not infer model quality, latency, memory traffic, or speedup from these
mechanistic metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class SlhaQualityError(ValueError):
    """Raised when a SLHA-facing KV quality record is malformed."""


def _validated_scores(name: str, values: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    if not isinstance(values, (tuple, list)) or not values:
        raise SlhaQualityError(f"{name} must be a non-empty tuple/list")
    out: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SlhaQualityError(f"{name} values must be numeric")
        value = float(value)
        if not math.isfinite(value):
            raise SlhaQualityError(f"{name} values must be finite")
        out.append(value)
    return tuple(out)


def _validated_k(k: int, n: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise SlhaQualityError("top_k must be a positive integer")
    if k > n:
        raise SlhaQualityError("top_k cannot exceed the item count")
    return k


def _validated_selection(selected_ids: tuple[int, ...] | list[int], n: int) -> tuple[int, ...]:
    if not isinstance(selected_ids, (tuple, list)):
        raise SlhaQualityError("selected_ids must be a tuple/list")
    selected = tuple(selected_ids)
    previous = -1
    for item in selected:
        if isinstance(item, bool) or not isinstance(item, int):
            raise SlhaQualityError("selected ids must be integers")
        if item < 0 or item >= n:
            raise SlhaQualityError("selected id is outside the score universe")
        if item <= previous:
            raise SlhaQualityError("selected_ids must be strictly increasing and unique")
        previous = item
    return selected


def stable_rank(scores: tuple[float, ...] | list[float]) -> tuple[int, ...]:
    """Return descending score order with item-id ascending as the tie-break."""
    values = _validated_scores("scores", scores)
    return tuple(sorted(range(len(values)), key=lambda i: (-values[i], i)))


def softmax_probabilities(scores: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    """Stable f64-equivalent Python softmax over one reference score row."""
    values = _validated_scores("scores", scores)
    maximum = max(values)
    weights = tuple(math.exp(value - maximum) for value in values)
    denominator = math.fsum(weights)
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise SlhaQualityError("softmax denominator is invalid")
    return tuple(weight / denominator for weight in weights)


@dataclass(frozen=True, slots=True)
class SelectionQuality:
    total_items: int
    selected_count: int
    top_k: int
    top_k_hits: int
    top_k_recall: float
    retained_softmax_mass: float
    omitted_softmax_mass: float
    top_k_missed_softmax_mass: float
    candidate_density: float


def evaluate_selection(
    reference_scores: tuple[float, ...] | list[float],
    selected_ids: tuple[int, ...] | list[int],
    *,
    top_k: int,
) -> SelectionQuality:
    """Evaluate a survivor set against exact reference scores.

    Exact scores are an evaluation oracle here. This function does not imply
    that a deployable router may read them.
    """
    scores = _validated_scores("reference_scores", reference_scores)
    k = _validated_k(top_k, len(scores))
    selected = _validated_selection(selected_ids, len(scores))
    selected_set = frozenset(selected)

    reference_top = stable_rank(scores)[:k]
    hits = sum(1 for item in reference_top if item in selected_set)
    probabilities = softmax_probabilities(scores)
    retained = math.fsum(probabilities[item] for item in selected)
    missed_top_mass = math.fsum(
        probabilities[item] for item in reference_top if item not in selected_set
    )
    omitted = max(0.0, 1.0 - retained)

    return SelectionQuality(
        total_items=len(scores),
        selected_count=len(selected),
        top_k=k,
        top_k_hits=hits,
        top_k_recall=hits / k,
        retained_softmax_mass=retained,
        omitted_softmax_mass=omitted,
        top_k_missed_softmax_mass=missed_top_mass,
        candidate_density=len(selected) / len(scores),
    )


@dataclass(frozen=True, slots=True)
class RankingQuality:
    total_items: int
    top_k: int
    reference_top_k: tuple[int, ...]
    candidate_top_k: tuple[int, ...]
    top_k_overlap_count: int
    top_k_recall: float
    boundary_pairs: int
    boundary_pair_correct: int
    boundary_pair_accuracy: float


def evaluate_ranking(
    reference_scores: tuple[float, ...] | list[float],
    candidate_scores: tuple[float, ...] | list[float],
    *,
    top_k: int,
) -> RankingQuality:
    """Compare candidate ordering with the reference top-k boundary.

    The boundary metric evaluates every reference top-k item against every
    non-top-k item. It is intentionally stricter than top-k set overlap.
    """
    reference = _validated_scores("reference_scores", reference_scores)
    candidate = _validated_scores("candidate_scores", candidate_scores)
    if len(reference) != len(candidate):
        raise SlhaQualityError("reference_scores and candidate_scores must have equal length")
    k = _validated_k(top_k, len(reference))

    reference_rank = stable_rank(reference)
    candidate_rank = stable_rank(candidate)
    reference_top = reference_rank[:k]
    candidate_top = candidate_rank[:k]
    reference_top_set = frozenset(reference_top)
    candidate_top_set = frozenset(candidate_top)
    overlap = len(reference_top_set & candidate_top_set)

    candidate_position = {item: position for position, item in enumerate(candidate_rank)}
    negatives = reference_rank[k:]
    total_pairs = len(reference_top) * len(negatives)
    correct = 0
    for positive in reference_top:
        p = candidate_position[positive]
        for negative in negatives:
            if p < candidate_position[negative]:
                correct += 1

    accuracy = 1.0 if total_pairs == 0 else correct / total_pairs
    return RankingQuality(
        total_items=len(reference),
        top_k=k,
        reference_top_k=reference_top,
        candidate_top_k=candidate_top,
        top_k_overlap_count=overlap,
        top_k_recall=overlap / k,
        boundary_pairs=total_pairs,
        boundary_pair_correct=correct,
        boundary_pair_accuracy=accuracy,
    )
