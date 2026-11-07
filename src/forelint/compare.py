"""Is model A really better than model B, or is it noise?

`diebold_mariano` tests one pair on one loss series, with the Harvey,
Leybourne and Newbold (1997) small-sample correction. `dm_test` does the same
for a long-format forecast table with many series, by averaging the loss
difference across series at each time stamp. `model_confidence_set` keeps the
set of models that cannot be told apart from the best (Hansen, Lunde and
Nason 2011), using a moving-block bootstrap and the T_max statistic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["DMResult", "diebold_mariano", "dm_test", "model_confidence_set"]

_LOSSES = {
    "squared": lambda e: e ** 2,
    "absolute": lambda e: np.abs(e),
}


@dataclass
class DMResult:
    statistic: float
    p_value: float
    mean_loss_diff: float
    n: int

    def __str__(self):
        side = "A worse" if self.mean_loss_diff > 0 else "A better"
        return (f"DM = {self.statistic:.3f}, p = {self.p_value:.4f} over {self.n} points "
                f"(mean loss difference {self.mean_loss_diff:.4g}, {side})")


def _dm_from_diff(d, h):
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = d.size
    if n < 3:
        raise ValueError("need at least 3 loss differences")
    mean = d.mean()
    dc = d - mean
    var = dc @ dc / n
    for k in range(1, h):
        var += 2.0 * (dc[k:] @ dc[:-k]) / n
    if var <= 0:
        return DMResult(np.nan, np.nan, float(mean), n)
    stat = mean / np.sqrt(var / n)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)  # HLN correction
    p = 2.0 * stats.t.sf(abs(stat), df=n - 1)
    return DMResult(float(stat), float(p), float(mean), n)


def diebold_mariano(errors_a, errors_b, h=1, loss="squared") -> DMResult:
    """Diebold-Mariano test with the HLN correction.

    A positive statistic means model A has the larger loss. ``h`` is the
    forecast horizon; losses up to lag ``h - 1`` are allowed to be correlated.
    """
    if loss not in _LOSSES:
        raise ValueError(f"loss must be one of {list(_LOSSES)}")
    ea, eb = np.asarray(errors_a, dtype=float), np.asarray(errors_b, dtype=float)
    if ea.shape != eb.shape:
        raise ValueError("error arrays must have the same shape")
    return _dm_from_diff(_LOSSES[loss](ea) - _LOSSES[loss](eb), h)


def dm_test(df, model_a, model_b, *, y="y", id_col="unique_id", time_col="ds", h=1,
            loss="squared") -> DMResult:
    """Diebold-Mariano on a long-format table of many series.

    ``df`` has one row per series and time stamp, with the actual in ``y`` and
    one column per model (the layout of statsforecast/mlforecast
    ``cross_validation`` output). The loss difference is averaged across
    series at each time stamp, then tested as one series.
    """
    if loss not in _LOSSES:
        raise ValueError(f"loss must be one of {list(_LOSSES)}")
    f = _LOSSES[loss]
    d = f(df[model_a] - df[y]) - f(df[model_b] - df[y])
    per_time = d.groupby(df[time_col]).mean().sort_index()
    return _dm_from_diff(per_time.to_numpy(), h)


def _block_indices(n, block, reps, rng):
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(reps, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(reps, -1)[:, :n]


def model_confidence_set(losses, size=0.10, reps=2000, block_size=None, random_state=None):
    """Hansen's model confidence set with the T_max statistic.

    Parameters
    ----------
    losses : DataFrame of shape (n_times, n_models), one column per model,
        each cell that model's loss at that time. For many series, average the
        loss across series per time stamp first.
    size : the test size; models whose MCS p-value is at least ``size`` stay
        in the set.
    block_size : moving-block length for the bootstrap. Defaults to
        ``ceil(n_times ** (1/3))``.

    Returns
    -------
    DataFrame indexed by model with ``pvalue`` and ``in_set``, ordered from
    first eliminated to last.
    """
    L = pd.DataFrame(losses).astype(float)
    if L.isna().any().any():
        raise ValueError("losses contain missing values")
    n, m = L.shape
    if m < 2:
        raise ValueError("need at least two models")
    block = block_size or int(np.ceil(n ** (1 / 3)))
    rng = np.random.default_rng(random_state)
    idx = _block_indices(n, block, reps, rng)
    vals = L.to_numpy()
    lbar = vals.mean(axis=0)                       # (m,)
    lboot = vals[idx].mean(axis=1)                 # (reps, m)

    included = list(range(m))
    order, pvals = [], []
    running = 0.0
    while len(included) > 1:
        s = np.array(included)
        dbar = lbar[s] - lbar[s].mean()
        dboot = lboot[:, s] - lboot[:, s].mean(axis=1, keepdims=True)
        dev = dboot - dbar
        var = (dev ** 2).mean(axis=0)
        var[var <= 0] = np.finfo(float).tiny
        t = dbar / np.sqrt(var)
        tmax = t.max()
        tboot = (dev / np.sqrt(var)).max(axis=1)
        p = float((tboot >= tmax).mean())
        running = max(running, p)
        worst = int(s[np.argmax(t)])
        order.append(worst)
        pvals.append(running)
        included.remove(worst)
    order.append(included[0])
    pvals.append(1.0)
    out = pd.DataFrame({"pvalue": pvals}, index=L.columns[order])
    out["in_set"] = out["pvalue"] >= size
    out.index.name = "model"
    return out
