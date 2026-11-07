"""Is model A really better than model B, or is it noise?

`diebold_mariano` tests one pair on one loss series, with the Harvey,
Leybourne and Newbold (1997) small-sample correction; if the long-run
variance comes out negative with h > 1, Newey-West weights are used instead. `dm_test` does the same
for a long-format backtest with many series: with a ``cutoff`` column (as in
statsforecast / mlforecast cross-validation output) every forecast origin is
one observation, because the errors of the steps that share an origin are
correlated, and overlapping windows are thinned to non-overlapping ones. `model_confidence_set` keeps the set of models that cannot be told
apart from the best (Hansen, Lunde and Nason 2011), using a circular block
bootstrap and the T_max statistic.
"""
from __future__ import annotations

import math
import warnings
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
    h: int = 1

    def __str__(self):
        if self.mean_loss_diff > 0:
            side = "A worse"
        elif self.mean_loss_diff < 0:
            side = "A better"
        else:
            side = "no difference"
        return (f"DM = {self.statistic:.3f}, p = {self.p_value:.4f} over {self.n} points, h = {self.h} "
                f"(mean loss difference {self.mean_loss_diff:.4g}, {side})")


def _dm_from_diff(d, h):
    if h < 1:
        raise ValueError("h must be at least 1")
    d = np.asarray(d, dtype=float)
    if np.isinf(d).any():
        raise ValueError("loss differences contain inf")
    if np.isnan(d).any():
        if h > 1:
            raise ValueError("loss differences contain NaN; with h > 1 dropping them would misalign the lags")
        d = d[~np.isnan(d)]
    n = d.size
    if n < 3:
        raise ValueError("need at least 3 loss differences")
    if h >= n:
        raise ValueError(f"h = {h} needs more than {h} loss differences, got {n}")
    mean = d.mean()
    dc = d - mean
    gamma = [dc @ dc / n] + [(dc[k:] @ dc[:-k]) / n for k in range(1, h)]
    var = gamma[0] + 2.0 * sum(gamma[1:])
    if var <= 0 and h > 1:
        # The rectangular long-run variance can turn negative; fall back to
        # Newey-West (Bartlett) weights, which keep it positive.
        var = gamma[0] + 2.0 * sum((1 - k / h) * gamma[k] for k in range(1, h))
    if var <= 0:
        return DMResult(np.nan, np.nan, float(mean), n, h)
    stat = mean / np.sqrt(var / n)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)  # HLN correction
    p = 2.0 * stats.t.sf(abs(stat), df=n - 1)
    return DMResult(float(stat), float(p), float(mean), n, h)


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


def origin_means(values, cutoff, time, ids=None):
    """Mean of `values` per forecast window, keeping only windows that do not overlap.

    Each series' windows are thinned greedily from its latest cutoff backwards:
    a cutoff is kept only if its window ends no later than the start of the
    window kept after it, so kept windows share no target dates. Windows are
    then lined up across series by their position counted back from each
    series' latest kept window (0 = latest), which also handles series that
    end on different dates. Returns (means ordered oldest to latest, thinned),
    where `thinned` is True if any rows were dropped.
    """
    frame = pd.DataFrame({"v": np.asarray(values, dtype=float), "c": np.asarray(cutoff),
                          "t": np.asarray(time), "id": 0 if ids is None else np.asarray(ids)})
    horizon = (frame["t"] - frame["c"]).max()
    rank = {}
    for item, g in frame.groupby("id", sort=False, observed=True):
        last = None
        k = 0
        for cut in np.sort(g["c"].unique())[::-1]:
            if last is None or cut + horizon <= last:
                rank[(item, cut)] = k
                k += 1
                last = cut
    key = pd.Series([rank.get((i, x), -1) for i, x in zip(frame["id"], frame["c"])], index=frame.index)
    kept = key >= 0
    per = frame.loc[kept, "v"].groupby(key[kept]).mean().sort_index(ascending=False)
    per.index.name = "window_from_latest"
    return per, bool((~kept).any())


def dm_test(df, model_a, model_b, *, y="y", id_col="unique_id", time_col="ds", cutoff_col="cutoff",
            h=None, loss="squared") -> DMResult:
    """Diebold-Mariano on a long-format backtest of many series.

    With a ``cutoff_col`` column, the loss difference is averaged over all
    series and steps of each forecast origin, and the test runs across
    origins. If the windows overlap (origins closer together than the
    horizon, such as statsforecast's default ``step_size=1``), only origins
    whose windows do not overlap are kept, because overlapping windows share
    target dates and a lag allowance does not hold the test's size. At least
    3 kept origins are needed; for a useful test, backtest with
    ``step_size`` at least the horizon and ten or more windows.

    Without a cutoff column, every time stamp is treated as a separate forecast
    origin and ``h`` defaults to 1. Only use that for genuine one-step-ahead
    forecasts: steps of one multi-step forecast are correlated and the test
    would reject far too often.
    """
    if loss not in _LOSSES:
        raise ValueError(f"loss must be one of {list(_LOSSES)}")
    missing = [c for c in [y, id_col, time_col, model_a, model_b] if c not in df.columns]
    if missing:
        raise KeyError(f"columns not found: {missing}")
    f = _LOSSES[loss]
    d = f(df[model_a] - df[y]) - f(df[model_b] - df[y])
    if cutoff_col and cutoff_col in df.columns:
        per, thinned = origin_means(d, df[cutoff_col], df[time_col], df[id_col])
        if thinned:
            warnings.warn(f"forecast windows overlap; using {len(per)} non-overlapping windows per series",
                          stacklevel=2)
        if len(per) < 3:
            raise ValueError(f"only {len(per)} non-overlapping forecast origins; the test needs at least 3. "
                             "Backtest with more windows, spaced at least the horizon apart.")
        if len(per) < 10:
            warnings.warn(f"dm_test across only {len(per)} forecast origins has little power", stacklevel=2)
        return _dm_from_diff(per.to_numpy(), 1 if h is None else h)
    if df.duplicated([id_col, time_col]).any():
        raise ValueError("duplicate (id, time) rows, as from overlapping cross-validation windows; "
                         f"pass the forecast origin as {cutoff_col!r}")
    per = d.groupby(df[time_col], observed=True).mean().sort_index()
    return _dm_from_diff(per.to_numpy(), 1 if h is None else h)


def _boot_means(vals, block, reps, rng, chunk_cells=5_000_000):
    """Circular block bootstrap means, (reps, m), without materialising reps × n × m."""
    n, m = vals.shape
    n_blocks = int(np.ceil(n / block))
    out = np.empty((reps, m))
    chunk = max(1, chunk_cells // max(1, n * m))
    for start in range(0, reps, chunk):
        r = min(chunk, reps - start)
        starts = rng.integers(0, n, size=(r, n_blocks))
        idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(r, -1)[:, :n]
        out[start:start + r] = vals[idx].mean(axis=1)
    return out


def model_confidence_set(losses, size=0.10, reps=2000, block_size=None, random_state=None):
    """Hansen's model confidence set with the T_max statistic.

    Parameters
    ----------
    losses : DataFrame of shape (n_times, n_models), one column per model,
        each cell that model's loss at that time (or forecast origin), in time
        order. For many series, average the loss across series first.
    size : the test size; models whose MCS p-value is at least ``size`` stay
        in the set.
    block_size : block length for the circular block bootstrap. Defaults to
        ``ceil(n_times ** (1/3))``.

    Returns
    -------
    DataFrame indexed by model with ``pvalue`` and ``in_set``, ordered from
    first eliminated to last.
    """
    L = pd.DataFrame(losses).astype(float)
    if not np.isfinite(L.to_numpy()).all():
        raise ValueError("losses contain missing or infinite values")
    n, m = L.shape
    if m < 2:
        raise ValueError("need at least two models")
    if n < 3:
        raise ValueError("need at least 3 time points")
    block = block_size or int(np.ceil(n ** (1 / 3)))
    rng = np.random.default_rng(random_state)
    vals = L.to_numpy()
    lbar = vals.mean(axis=0)
    lboot = _boot_means(vals, block, reps, rng)

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
