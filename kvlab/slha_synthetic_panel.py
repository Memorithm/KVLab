"""Deterministic SKV-1 synthetic falsification panel.

The panel exists to falsify simplistic interpretations of top-k recall. It uses
exact reference score rows and deliberately controlled survivor perturbations.
It is not representative-model evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from .slha_evidence import SlhaQualityEvidenceV1
from .slha_quality import stable_rank


@dataclass(frozen=True, slots=True)
class SyntheticScoreCase:
    case_id: str
    scores: tuple[float, ...]
    top_k: int


CASES: tuple[SyntheticScoreCase, ...] = (
    SyntheticScoreCase(
        "dominant-head",
        (10.0, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4),
        2,
    ),
    SyntheticScoreCase(
        "near-tie-boundary",
        (3.00, 2.99, 2.98, 2.97, 0.0, -0.1, -0.2, -0.3),
        2,
    ),
    SyntheticScoreCase(
        "two-dominant",
        (8.0, 7.5, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
        2,
    ),
    SyntheticScoreCase(
        "flat",
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        2,
    ),
    SyntheticScoreCase(
        "late-relevance",
        (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 6.0),
        2,
    ),
    SyntheticScoreCase(
        "heavy-tail",
        (6.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0),
        2,
    ),
)


def _sorted_unique(items: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(set(items)))


def exact_top_k(case: SyntheticScoreCase) -> tuple[int, ...]:
    return tuple(sorted(stable_rank(case.scores)[: case.top_k]))


def tail_matched_density(case: SyntheticScoreCase) -> tuple[int, ...]:
    count = case.top_k
    return tuple(range(len(case.scores) - count, len(case.scores)))


def drop_top1_add_boundary(case: SyntheticScoreCase) -> tuple[int, ...]:
    """Keep k-1 reference winners but replace the strongest key by rank k+1."""
    rank = stable_rank(case.scores)
    if case.top_k >= len(rank):
        return tuple(sorted(rank))
    survivors = list(rank[: case.top_k])
    survivors.remove(rank[0])
    survivors.append(rank[case.top_k])
    return _sorted_unique(tuple(survivors))


def drop_boundary_add_next(case: SyntheticScoreCase) -> tuple[int, ...]:
    """Replace only the weakest reference top-k item by the next boundary item."""
    rank = stable_rank(case.scores)
    if case.top_k >= len(rank):
        return tuple(sorted(rank))
    survivors = list(rank[: case.top_k])
    survivors.remove(rank[case.top_k - 1])
    survivors.append(rank[case.top_k])
    return _sorted_unique(tuple(survivors))


def run_panel() -> tuple[SlhaQualityEvidenceV1, ...]:
    records: list[SlhaQualityEvidenceV1] = []
    policies = (
        ("exact-top-k", exact_top_k),
        ("tail-matched-density", tail_matched_density),
        ("drop-top1-add-boundary", drop_top1_add_boundary),
        ("drop-boundary-add-next", drop_boundary_add_next),
    )
    for case in CASES:
        for policy_id, policy in policies:
            records.append(
                SlhaQualityEvidenceV1.capture(
                    experiment_id="skv1-synthetic-mass-panel-v1",
                    query_id=case.case_id,
                    candidate_id=policy_id,
                    top_k=case.top_k,
                    reference_scores=case.scores,
                    selected_ids=policy(case),
                )
            )
    return tuple(records)
