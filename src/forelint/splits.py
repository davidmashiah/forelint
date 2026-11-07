"""Splitters that hold out the future *and* unseen entities.

scikit-learn ships `GroupKFold` (unseen entities, any time) and
`TimeSeriesSplit` (future rows, every entity seen). Forecasting new stores,
new products or new wells needs both at once: train on the past of some
entities, test on the future of entities the model never saw. Without that,
related series leak through validation and the score flatters the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

__all__ = ["GroupTimeSplit"]


class GroupTimeSplit(BaseCrossValidator):
    """Expanding-window splits that can also hold out whole groups.

    Parameters
    ----------
    time : array-like of shape (n_samples,)
        The time stamp of every row, aligned with ``X``. Any sortable type,
        no missing values.
    n_splits : int, default 5
        Number of test windows, taken from the end of the time range.
    test_size : int, optional
        Unique time points per test window. Defaults to
        ``n_unique_times // (n_splits + 1)``.
    gap : int, default 0
        Unique time points dropped between the end of training and the
        start of each test window, to mimic a forecast lead time.
    holdout : {"groups", "none"}, default "groups"
        ``"groups"`` tests only on groups that are absent from training
        (new entities). ``"none"`` keeps every group on both sides, which is
        an ordinary time-series split.
    test_group_frac : float, default 0.2
        Share of groups held out in each fold when ``holdout="groups"``.
    random_state : int, optional
        Seed for which groups are held out.

    Notes
    -----
    ``split`` needs ``groups`` when ``holdout="groups"``. Works with
    ``sklearn.model_selection.cross_validate(..., groups=groups)``. Because
    ``time`` is bound to the full data at construction, the splitter cannot be
    used for an inner loop of nested cross-validation, where the data passed to
    ``split`` is a subset.
    """

    def __init__(self, time, n_splits=5, test_size=None, gap=0, holdout="groups",
                 test_group_frac=0.2, random_state=None):
        if holdout not in ("groups", "none"):
            raise ValueError("holdout must be 'groups' or 'none'")
        if not 0 < test_group_frac < 1:
            raise ValueError("test_group_frac must be between 0 and 1")
        if n_splits < 1:
            raise ValueError("n_splits must be at least 1")
        if test_size is not None and test_size < 1:
            raise ValueError("test_size must be at least 1")
        if gap < 0:
            raise ValueError("gap cannot be negative")
        self.time = time
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.holdout = holdout
        self.test_group_frac = test_group_frac
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def windows(self):
        """The (train times, test times) of each fold, as arrays of unique times."""
        time = np.asarray(self.time)
        if pd.isna(time).any():
            raise ValueError("time contains missing values")
        uniq = np.unique(time)
        size = self.test_size if self.test_size is not None else len(uniq) // (self.n_splits + 1)
        if size < 1:
            raise ValueError("not enough unique time points for this many splits")
        first_test = len(uniq) - self.n_splits * size
        if first_test - self.gap < 1:
            raise ValueError("no training time left before the first test window")
        return [(uniq[: first_test + k * size - self.gap], uniq[first_test + k * size: first_test + (k + 1) * size])
                for k in range(self.n_splits)]

    def split(self, X, y=None, groups=None):
        time = np.asarray(self.time)
        n = len(time) if X is None else len(X)
        if len(time) != n:
            raise ValueError(f"time has {len(time)} rows but X has {n}")
        idx = np.arange(n)
        windows = self.windows()
        if self.holdout == "groups":
            if groups is None:
                raise ValueError("holdout='groups' needs the groups argument")
            groups = np.asarray(groups)
            uniq_groups = np.unique(groups)
            n_test = max(1, int(round(self.test_group_frac * len(uniq_groups))))
            if n_test >= len(uniq_groups):
                raise ValueError("test_group_frac leaves no groups for training")
            rng = np.random.default_rng(self.random_state)
        for train_times, test_times in windows:
            in_train_time = np.isin(time, train_times)
            in_test_time = np.isin(time, test_times)
            if self.holdout == "groups":
                held = rng.choice(uniq_groups, size=n_test, replace=False)
                is_held = np.isin(groups, held)
                train, test = idx[in_train_time & ~is_held], idx[in_test_time & is_held]
            else:
                train, test = idx[in_train_time], idx[in_test_time]
            if len(train) == 0 or len(test) == 0:
                raise ValueError("a fold came out empty; use fewer splits or a larger test_group_frac")
            yield train, test

    def _iter_test_indices(self, X=None, y=None, groups=None):  # pragma: no cover
        for _, test in self.split(X, y, groups):
            yield test
