"""Demand classes: smooth, erratic, intermittent, lumpy.

Syntetos, Boylan and Croston (2005) classify each item by two numbers:

- ADI, the average inter-demand interval: periods per period with demand,
  counted from the item's first demand, so the run of zeros before a product
  is launched does not make it look intermittent.
- CV², the squared coefficient of variation of the non-zero demand sizes.

With the cut-offs ADI = 1.32 and CV² = 0.49 this gives four classes, and the
classes behave very differently under the same model. An average error over
all items mixes them; the audit can break the error down by class instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["classify_demand", "ADI_CUTOFF", "CV2_CUTOFF"]

ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49


def _one(y):
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    pos = np.flatnonzero(y > 0)
    if pos.size == 0:
        return np.nan, np.nan, "no demand"
    nz = y[pos]
    adi = (y.size - pos[0]) / pos.size
    cv2 = (nz.std(ddof=0) / nz.mean()) ** 2
    if adi < ADI_CUTOFF:
        cls = "smooth" if cv2 < CV2_CUTOFF else "erratic"
    else:
        cls = "intermittent" if cv2 < CV2_CUTOFF else "lumpy"
    return adi, cv2, cls


def classify_demand(df, *, y="y", id_col="unique_id", time_col="ds"):
    """Classify every item's demand history.

    Parameters
    ----------
    df : long-format history, one row per item and period, demand in ``y``.
        Use the training history, not the test window, so the class is known
        before the forecast is made. Duplicate (item, period) rows, as in
        cross-validation output with overlapping windows, are counted once.

    Returns
    -------
    DataFrame indexed by item with ``n``, ``adi``, ``cv2`` and ``demand_class``
    (``smooth``, ``erratic``, ``intermittent``, ``lumpy`` or ``no demand``).

    Notes
    -----
    ADI is counted from the first period with demand. CV² uses the population
    standard deviation of the positive demand sizes. Zero and negative values
    (returns) both count as periods without demand. Missing values are
    ignored, so pass stockout periods as NaN if they do not measure demand.
    """
    d = df[[id_col, time_col, y]].drop_duplicates([id_col, time_col]).sort_values([id_col, time_col])
    rows = {}
    for item, g in d.groupby(id_col, sort=True, observed=True):
        adi, cv2, cls = _one(g[y].to_numpy())
        rows[item] = {"n": int(g[y].notna().sum()), "adi": adi, "cv2": cv2, "demand_class": cls}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = id_col
    return out
