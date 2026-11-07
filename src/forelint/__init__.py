"""forelint: lint your forecast backtests before you trust a score.

- `GroupTimeSplit`: hold out the future of entities the model never saw.
- `split_optimism`: how much each split strategy flatters the model.
- `noise_floor`: the run-to-run spread, below which a difference is noise.
- `diebold_mariano`, `dm_test`, `model_confidence_set`: is A really better than B.
- `audit`: per-item bias, tracking signal, error concentration, forecast value added.
"""
from .audit import AuditReport, audit
from .compare import DMResult, diebold_mariano, dm_test, model_confidence_set
from .noise import NoiseFloor, noise_floor
from .optimism import split_optimism
from .splits import GroupTimeSplit

__version__ = "0.1.0"

__all__ = [
    "AuditReport", "audit",
    "DMResult", "diebold_mariano", "dm_test", "model_confidence_set",
    "NoiseFloor", "noise_floor",
    "split_optimism",
    "GroupTimeSplit",
]
