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
    """Scores from repeated runs of one unchanged setup.

    ``critical_difference`` is the smallest difference between two single runs
    that is significant at ``confidence``: t · √2 · sd. It is not a power
    calculation; a real effect of exactly that size is detected only about
    half the time.
    """

    scores: np.ndarray
    confidence: float = 0.95
    mean: float = field(init=False)
    sd: float = field(init=False)
    critical_difference: float = field(init=False)

    def __post_init__(self):
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be between 0 and 1, e.g. 0.95")
        self.scores = np.asarray(self.scores, dtype=float)
        if not np.all(np.isfinite(self.scores)):
            raise ValueError("a run returned NaN or inf; fix the run before measuring its noise")
        if self.scores.size < 3:
            raise ValueError("a noise floor needs at least 3 runs; two cannot estimate a spread")
        self.mean = float(self.scores.mean())
        self.sd = float(self.scores.std(ddof=1))
        q = stats.t.ppf(0.5 + self.confidence / 2, df=self.scores.size - 1)
        self.critical_difference = float(q * np.sqrt(2.0) * self.sd)

    @property
    def mde(self) -> float:
        """Deprecated alias of ``critical_difference``."""
        return self.critical_difference

    def is_meaningful(self, difference: float) -> bool:
        """True if a single-run difference is larger than the noise floor."""
        return abs(difference) > self.critical_difference

    def __str__(self):
        return (f"noise floor over {self.scores.size} runs: mean {self.mean:.4g}, "
                f"sd {self.sd:.4g}; differences under {self.critical_difference:.4g} are noise "
                f"at {self.confidence:.0%}")


def noise_floor(run, seeds=(0, 1, 2, 3, 4), confidence=0.95) -> NoiseFloor:
    """Call ``run(seed) -> float`` once per seed and summarise the spread.

    Example
    -------
    >>> nf = noise_floor(lambda s: train_and_score(seed=s), seeds=range(5))
    >>> nf.is_meaningful(new_score - old_score)
    """
    return NoiseFloor(np.array([float(run(s)) for s in seeds]), confidence=confidence)
