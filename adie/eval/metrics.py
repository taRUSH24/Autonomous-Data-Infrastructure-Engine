"""Small statistics helpers for the research deliverables -- Wilson score
intervals for binomial proportions (repair success rate, false-repair
rate) computed honestly even at small sample sizes, rather than omitted or
overstated with a fabricated confidence interval."""
from __future__ import annotations

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Returns (point_estimate, lower, upper) for a 95% Wilson score interval.
    Appropriate for small-n binomial proportions, unlike the normal
    approximation this project's sample sizes would otherwise violate."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half_width = (z * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))) / denom
    return (p, max(0.0, center - half_width), min(1.0, center + half_width))
