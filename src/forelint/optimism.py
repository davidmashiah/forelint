"""How much does a leaky split flatter the model?

Each leaky strategy is compared with its honest counterpart on the *same test
rows* with the *same training-set size per group*, so the only difference is
what the training data is allowed to contain:

- ``random`` vs ``time`` (leak through time and through other series on the
  same dates): test on a random half of each test window. ``time`` trains on
  the past only; ``random`` draws the same number of rows per group from
  anything but the test rows, including the same dates and the future.
- ``group`` vs ``group_time`` (leak through the future of other entities):
  test on the window rows of held-out groups. ``group_time`` trains on the
  past of the other groups; ``group`` draws the same number of rows per group
  from those groups at any time.
- ``kfold`` vs ``group`` (leak through the entity itself, what plain K-fold
  does to related series): same test rows. ``group`` keeps the held-out
  groups out of training; ``kfold`` draws a same-size sample from every row
  but the test rows, including the held-out groups' other rows.

``optimism`` = 1 − leaky error / honest error, from the per-fold log error
ratios averaged with weights proportional to the honest training size, with a
95% t interval over folds. An interval that excludes 0 on the positive side is
a detected leak. Negative values mean no detectable leak, not a pessimistic
split: with small training sets, flexible models such as trees can do worse on
a scattered sample than on a contiguous block of the past even when the
sample contains the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.utils import _safe_indexing

from .splits import GroupTimeSplit

__all__ = ["split_optimism"]

_PAIRS = {"random": "time", "group": "group_time", "kfold": "group"}
_ALL = ("time", "random", "group_time", "group", "kfold")


def _match(rng, pool, pool_groups, counts):
    """Sample, per group, as many rows from ``pool`` as ``counts`` asks for."""
    out = []
    for g, n in counts.items():
        rows = pool[pool_groups == g]
        if len(rows):
            out.append(rng.choice(rows, size=min(n, len(rows)), replace=False))
    return np.concatenate(out) if out else np.array([], dtype=int)


def _folds(time, groups, n_splits, test_group_frac, gap, rng):
    time, groups = np.asarray(time), np.asarray(groups)
    idx = np.arange(len(time))
    windows = GroupTimeSplit(time, n_splits=n_splits, gap=gap, holdout="none").windows()
    uniq_groups = np.unique(groups)
    n_held = max(1, int(round(test_group_frac * len(uniq_groups))))
    if n_held >= len(uniq_groups):
        raise ValueError("test_group_frac leaves no groups for training")
    count = lambda rows: pd.Series(groups[rows]).value_counts().to_dict()
    for train_times, test_times in windows:
        in_past = np.isin(time, train_times)
        in_win = np.isin(time, test_times)
        win = idx[in_win]
        # random vs time: same test rows, same rows per group
        test_rt = np.sort(rng.choice(win, size=max(1, len(win) // 2), replace=False))
        train_t = idx[in_past]
        pool = np.setdiff1d(idx, test_rt)
        train_r = _match(rng, pool, groups[pool], count(train_t))
        # group vs group_time, and kfold vs group: same test rows (held groups, window)
        held = np.isin(groups, rng.choice(uniq_groups, size=n_held, replace=False))
        test_g = idx[in_win & held]
        train_gt = idx[in_past & ~held]
        pool_g = idx[~held]
        train_g = _match(rng, pool_g, groups[pool_g], count(train_gt))
        pool_k = np.setdiff1d(idx, test_g)
        # kfold: same total size, spread over all groups in proportion to what each has
        avail = pd.Series(groups[pool_k]).value_counts()
        share = (avail / avail.sum() * len(train_g)).round().astype(int).to_dict()
        train_k = _match(rng, pool_k, groups[pool_k], share)
        yield {"time": (train_t, test_rt), "random": (train_r, test_rt),
               "group_time": (train_gt, test_g), "group": (train_g, test_g), "kfold": (train_k, test_g)}


def split_optimism(estimator, X, y, *, groups, time, scoring="neg_mean_absolute_error",
                   strategies=_ALL, n_splits=5, test_group_frac=0.2, gap=0, random_state=0):
    """Measure how much leaky split strategies overstate accuracy.

    Parameters
    ----------
    estimator : scikit-learn estimator (cloned for every fit)
    X, y : data, aligned with ``groups`` and ``time``
    groups : array-like, the entity of each row (store, SKU, well, region)
    time : array-like, the time stamp of each row
    scoring : an error scorer name, i.e. one starting with ``neg_`` such as
        ``"neg_mean_absolute_error"`` or ``"neg_root_mean_squared_error"``.
    strategies : any of ``time``, ``random``, ``group_time``, ``group``,
        ``kfold``. A leaky strategy gets an optimism figure when its honest
        counterpart is also run.

    Returns
    -------
    DataFrame indexed by strategy with ``error`` (mean over folds),
    ``error_sd``, ``n_folds``, ``honest`` (the counterpart it is compared
    with), ``optimism`` (1 − error / counterpart error) and its 95% interval
    ``optimism_low``, ``optimism_high``.
    """
    if not (isinstance(scoring, str) and scoring.startswith("neg_")):
        raise ValueError("scoring must be an error scorer name starting with 'neg_', "
                         "e.g. 'neg_mean_absolute_error'; optimism is a ratio of errors")
    scorer = get_scorer(scoring)
    unknown = set(strategies) - set(_ALL)
    if unknown:
        raise ValueError(f"unknown strategies: {sorted(unknown)}")
    rng = np.random.default_rng(random_state)
    errs = {s: [] for s in strategies}
    sizes = {s: [] for s in strategies}
    for fold in _folds(time, groups, n_splits, test_group_frac, gap, rng):
        for s in strategies:
            tr, te = fold[s]
            sizes[s].append(len(tr))
            if len(tr) == 0 or len(te) == 0:
                raise ValueError("a fold came out empty; use fewer splits or a larger test_group_frac")
            est = clone(estimator).fit(_safe_indexing(X, tr), _safe_indexing(y, tr))
            errs[s].append(-scorer(est, _safe_indexing(X, te), _safe_indexing(y, te)))
    errs = {s: np.asarray(e, dtype=float) for s, e in errs.items()}
    rows = {}
    for s, e in errs.items():
        h = _PAIRS.get(s)
        opt = lo = hi = np.nan
        if h in errs:
            lr = np.log(e / errs[h])
            w = np.asarray(sizes[h], dtype=float)
            m = float(np.average(lr, weights=w))
            opt = 1 - np.exp(m)
            if len(lr) > 1:
                se = lr.std(ddof=1) / np.sqrt(len(lr))
                q = stats.t.ppf(0.975, df=len(lr) - 1)
                lo, hi = 1 - np.exp(m + q * se), 1 - np.exp(m - q * se)
        rows[s] = {"error": e.mean(), "error_sd": e.std(ddof=1) if len(e) > 1 else np.nan,
                   "n_folds": len(e), "honest": h or "", "optimism": opt,
                   "optimism_low": lo, "optimism_high": hi}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "strategy"
    return out
