"""How much does the way you split flatter the model?

Run one estimator under several validation schemes and report each scheme's
error next to the strictest one. If random K-fold reports half the error of a
split that holds out the future of unseen entities, the random score is not a
measurement of the deployed model; it is a measurement of the leak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, cross_validate

from .splits import GroupTimeSplit

__all__ = ["split_optimism"]

_STRATEGIES = ("random", "group", "time", "group_time")


def _splitter(name, *, time, n_splits, test_group_frac, gap, random_state):
    if name == "random":
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if name == "group":
        return GroupKFold(n_splits=n_splits)
    if name == "time":
        return GroupTimeSplit(time, n_splits=n_splits, gap=gap, holdout="none")
    if name == "group_time":
        return GroupTimeSplit(time, n_splits=n_splits, gap=gap, holdout="groups",
                              test_group_frac=test_group_frac, random_state=random_state)
    raise ValueError(f"unknown strategy {name!r}; choose from {_STRATEGIES}")


def split_optimism(estimator, X, y, *, groups, time, scoring="neg_mean_absolute_error",
                   strategies=_STRATEGIES, reference="group_time", n_splits=5,
                   test_group_frac=0.2, gap=0, random_state=0):
    """Score one estimator under several split strategies.

    Parameters
    ----------
    estimator : scikit-learn estimator
    X, y : training data
    groups : array-like, the entity of each row (store, SKU, well, region)
    time : array-like, the time stamp of each row
    scoring : str, a scikit-learn scorer. ``neg_*`` scorers are turned back
        into positive errors.
    strategies : subset of ``("random", "group", "time", "group_time")``
    reference : the strategy to compare against, normally the one that
        matches how the model will be used.

    Returns
    -------
    pandas.DataFrame indexed by strategy with ``error`` (mean over folds),
    ``error_sd``, ``n_folds`` and ``vs_reference``: the strategy's error as a
    share of the reference error. ``0.6`` means that strategy reports 40%
    less error than the reference, i.e. it is 40% optimistic.
    """
    if reference not in strategies:
        raise ValueError("reference must be one of the strategies")
    sign = -1.0 if scoring.startswith("neg_") else 1.0
    rows = {}
    for name in strategies:
        cv = _splitter(name, time=time, n_splits=n_splits, test_group_frac=test_group_frac,
                       gap=gap, random_state=random_state)
        res = cross_validate(estimator, X, y, groups=None if name == "random" else groups,
                             cv=cv, scoring=scoring)
        err = sign * np.asarray(res["test_score"], dtype=float)
        rows[name] = {"error": err.mean(), "error_sd": err.std(ddof=1) if len(err) > 1 else np.nan,
                      "n_folds": len(err)}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out["vs_reference"] = out["error"] / out.loc[reference, "error"]
    out.index.name = "strategy"
    return out
