"""Deterministic ridge baseline for K9 cross-model KV calibration.

This module fits only calibration matrices. It does not load model weights, remove
RoPE, select layers/heads, execute a final holdout, or make transfer-quality claims.
The caller must provide already prepared numeric rows from the preregistered K9
calibration split.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class RidgeFit:
    """A fitted multi-output linear ridge map with no intercept."""

    alpha: float
    feature_dim: int
    target_dim: int
    weights: tuple[tuple[float, ...], ...]

    def predict(self, rows: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
        prepared = _matrix("rows", rows, expected_width=self.feature_dim)
        return tuple(
            tuple(
                sum(row[column] * self.weights[column][target] for column in range(self.feature_dim))
                for target in range(self.target_dim)
            )
            for row in prepared
        )


def fit_ridge_calibration(
    features: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    *,
    alpha: float,
    calibration_ids: Iterable[str],
    final_holdout_ids: Iterable[str],
) -> RidgeFit:
    """Fit the K9 ridge baseline on calibration rows only.

    ``calibration_ids`` must identify exactly the supplied rows. Final-holdout IDs
    are accepted only to enforce split disjointness; no holdout values enter the
    fit. The implementation solves ``(X^T X + alpha I) W = X^T Y`` using
    deterministic Gauss-Jordan elimination with partial pivoting.
    """

    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")

    x = _matrix("features", features)
    y = _matrix("targets", targets)
    if len(x) != len(y):
        raise ValueError("features and targets must contain the same number of rows")

    calibration = _ids("calibration_ids", calibration_ids)
    holdout = _ids("final_holdout_ids", final_holdout_ids)
    if len(calibration) != len(x):
        raise ValueError("calibration_ids must identify exactly the supplied calibration rows")
    if set(calibration).intersection(holdout):
        raise ValueError("calibration_ids and final_holdout_ids must be disjoint")

    feature_dim = len(x[0])
    target_dim = len(y[0])
    gram = [[0.0 for _ in range(feature_dim)] for _ in range(feature_dim)]
    rhs = [[0.0 for _ in range(target_dim)] for _ in range(feature_dim)]

    for row_x, row_y in zip(x, y, strict=True):
        for left in range(feature_dim):
            for right in range(feature_dim):
                gram[left][right] += row_x[left] * row_x[right]
            for target in range(target_dim):
                rhs[left][target] += row_x[left] * row_y[target]

    for diagonal in range(feature_dim):
        gram[diagonal][diagonal] += alpha

    weights = _solve(gram, rhs)
    return RidgeFit(
        alpha=alpha,
        feature_dim=feature_dim,
        target_dim=target_dim,
        weights=tuple(tuple(row) for row in weights),
    )


def _matrix(
    name: str,
    rows: Sequence[Sequence[float]],
    *,
    expected_width: int | None = None,
) -> tuple[tuple[float, ...], ...]:
    if not rows:
        raise ValueError(f"{name} must be non-empty")
    prepared = tuple(tuple(float(value) for value in row) for row in rows)
    width = len(prepared[0])
    if width == 0:
        raise ValueError(f"{name} rows must be non-empty")
    if expected_width is not None and width != expected_width:
        raise ValueError(f"{name} row width does not match fitted feature dimension")
    if any(len(row) != width for row in prepared):
        raise ValueError(f"{name} must be rectangular")
    if any(not math.isfinite(value) for row in prepared for value in row):
        raise ValueError(f"{name} must contain only finite values")
    return prepared


def _ids(name: str, values: Iterable[str]) -> tuple[str, ...]:
    prepared = tuple(values)
    if not prepared or any(not value.strip() for value in prepared):
        raise ValueError(f"{name} must contain non-empty identifiers")
    if len(prepared) != len(set(prepared)):
        raise ValueError(f"{name} must not contain duplicates")
    return prepared


def _solve(
    matrix: list[list[float]], rhs: list[list[float]]
) -> list[list[float]]:
    size = len(matrix)
    outputs = len(rhs[0])
    augmented = [matrix[row][:] + rhs[row][:] for row in range(size)]

    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        pivot_value = augmented[pivot][column]
        if pivot_value == 0.0 or not math.isfinite(pivot_value):
            raise ValueError("ridge normal equations are singular; use a positive alpha or valid calibration matrix")
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]

        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [
                current - factor * pivoted
                for current, pivoted in zip(augmented[row], augmented[column], strict=True)
            ]

    solved = [row[size : size + outputs] for row in augmented]
    if any(not math.isfinite(value) for row in solved for value in row):
        raise ValueError("ridge fit produced non-finite weights")
    return solved
