"""The run-to-run noise floor.

Seeds, bootstrap draws, GPU nondeterminism and data loading order all move a
score. Two runs cannot estimate that spread: in one recorded case the first
two runs of identical code differed by 0.018 and the third moved the spread to
a standard deviation of 0.069. Measure the floor first; any difference smaller
than it is not a result.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

__all__ = ["NoiseFloor", "noise_floor"]


@dataclass
class NoiseFloor:
    """Scores from repeated runs of one unchanged setup."""

    scores: np.ndarray
    confidence: float = 0.95
    mean: float = field(init=False)
    sd: float = field(init=False)
    mde: float = field(init=False)

    def __post_init__(self):
        self.scores = np.asarray(self.scores, dtype=float)
        if self.scores.size < 3:
            raise ValueError("a noise floor needs at least 3 runs; two cannot estimate a spread")
        self.mean = float(self.scores.mean())
        self.sd = float(self.scores.std(ddof=1))
        # Smallest difference between two single runs that exceeds noise at
        # this confidence, using the t quantile for the runs we actually have.
        q = stats.t.ppf(0.5 + self.confidence / 2, df=self.scores.size - 1)
        self.mde = float(q * np.sqrt(2.0) * self.sd)

    def is_meaningful(self, difference: float) -> bool:
        """True if a single-run difference is larger than the noise floor."""
        return abs(difference) > self.mde

    def __str__(self):
        return (f"noise floor over {self.scores.size} runs: mean {self.mean:.4g}, "
                f"sd {self.sd:.4g}; differences under {self.mde:.4g} are noise "
                f"at {self.confidence:.0%}")


def noise_floor(run, seeds=(0, 1, 2, 3, 4), confidence=0.95) -> NoiseFloor:
    """Call ``run(seed) -> float`` once per seed and summarise the spread.

    Example
    -------
    >>> nf = noise_floor(lambda s: train_and_score(seed=s), seeds=range(5))
    >>> nf.is_meaningful(new_score - old_score)
    """
    return NoiseFloor(np.array([float(run(s)) for s in seeds]), confidence=confidence)
