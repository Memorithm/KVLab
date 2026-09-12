"""Small dependency-free uncertainty summaries for repeated KVLab measurements.

The helper reports descriptive uncertainty only; it does not decide scientific
significance and must not be used to tune against protected final holdouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import fmean, stdev


@dataclass(frozen=True, slots=True)
class UncertaintySummary:
    count: int
    mean: float
    sample_stddev: float
    standard_error: float
    confidence_level: float
    z_value: float
    interval_low: float
    interval_high: float

    def __post_init__(self) -> None:
        if self.count < 2:
            raise ValueError("uncertainty summary requires at least two samples")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between zero and one")
        if self.z_value <= 0.0:
            raise ValueError("z_value must be positive")
        if self.interval_low > self.interval_high:
            raise ValueError("interval bounds are inverted")


def summarize_normal_interval(
    values: tuple[float, ...] | list[float],
    *,
    confidence_level: float = 0.95,
    z_value: float = 1.959963984540054,
) -> UncertaintySummary:
    """Return a deterministic normal-approximation interval for repeated values.

    The default z value is the two-sided 95% standard-normal quantile. Callers
    must record this method in experiment provenance; for small/non-normal
    samples a preregistered bootstrap or other method may be more appropriate.
    """

    samples = tuple(float(value) for value in values)
    if len(samples) < 2:
        raise ValueError("at least two repeated measurements are required")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if z_value <= 0.0:
        raise ValueError("z_value must be positive")
    if any(value != value or value in (float("inf"), float("-inf")) for value in samples):
        raise ValueError("measurements must be finite")

    mean = fmean(samples)
    sample_stddev = stdev(samples)
    standard_error = sample_stddev / sqrt(len(samples))
    margin = z_value * standard_error

    return UncertaintySummary(
        count=len(samples),
        mean=mean,
        sample_stddev=sample_stddev,
        standard_error=standard_error,
        confidence_level=confidence_level,
        z_value=z_value,
        interval_low=mean - margin,
        interval_high=mean + margin,
    )
