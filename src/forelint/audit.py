"""One report on a backtest: bias, tracking signal, error concentration and
forecast value added, per item and overall.

Average error hides the two things that cost money: items whose forecast is
always too high or always too low, and the few items that carry most of the
error. The audit surfaces both, and compares every model with a baseline the
way forecast value added does, because a model that loses to a naive forecast
on most items should not ship whatever its average says.

Sign convention: error = forecast − actual, so positive bias is over-forecasting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["AuditReport", "audit"]


@dataclass
class AuditReport:
    per_item: pd.DataFrame
    summary: pd.DataFrame
    baseline: str | None
    top_frac: float
    ts_limit: float

    def __str__(self):
        lines = ["forelint audit", ""]
        for model, r in self.summary.iterrows():
            lines.append(f"{model}")
            lines.append(f"  MAE {r['mae']:.4g}   RMSE {r['rmse']:.4g}   bias {r['bias']:+.4g} "
                         f"({r['pct_bias']:+.1%} of actuals)")
            lines.append(f"  items with |tracking signal| > {self.ts_limit:g}: "
                         f"{int(r['n_biased_items'])} of {int(r['n_items'])} ({r['share_biased_items']:.0%})")
            lines.append(f"  top {self.top_frac:.0%} of items hold {r['error_concentration']:.0%} of squared error")
            if self.baseline is not None and model != self.baseline:
                lines.append(f"  forecast value added vs {self.baseline}: {r['fva']:+.1%} on MAE; "
                             f"worse than {self.baseline} on {r['share_items_worse']:.0%} of items")
            lines.append("")
        return "\n".join(lines).rstrip()


def _tracking_signal(e):
    mad = np.abs(e).mean()
    return np.nan if mad == 0 else e.sum() / mad


def audit(df, models, *, baseline=None, y="y", id_col="unique_id", time_col="ds",
          top_frac=0.10, ts_limit=4.0) -> AuditReport:
    """Audit a long-format backtest.

    Parameters
    ----------
    df : DataFrame with one row per item and time stamp, the actual in ``y``
        and one column per model. statsforecast / mlforecast
        ``cross_validation`` output works as is.
    models : list of model column names.
    baseline : optional model column to measure forecast value added against,
        typically a naive or seasonal-naive forecast.
    top_frac : share of items used for the error-concentration figure.
    ts_limit : tracking-signal alarm level; 4 is a common choice.

    Returns
    -------
    AuditReport with ``per_item`` (one row per item and model) and ``summary``
    (one row per model). ``print(report)`` gives a readable summary.
    """
    models = list(models)
    cols = models + ([baseline] if baseline and baseline not in models else [])
    missing = [c for c in [y, id_col, time_col, *cols] if c not in df.columns]
    if missing:
        raise KeyError(f"columns not found: {missing}")
    d = df.sort_values([id_col, time_col])
    items = []
    for model in cols:
        e = d[model] - d[y]
        g = pd.DataFrame({id_col: d[id_col], "e": e, "ae": e.abs(), "se": e ** 2, "y": d[y]}).groupby(id_col)
        part = pd.DataFrame({
            "n": g["e"].size(),
            "mae": g["ae"].mean(),
            "rmse": np.sqrt(g["se"].mean()),
            "bias": g["e"].mean(),
            "sse": g["se"].sum(),
            "sum_actual": g["y"].sum(),
            "tracking_signal": g["e"].apply(lambda s: _tracking_signal(s.to_numpy())),
        })
        part["model"] = model
        items.append(part.reset_index())
    per_item = pd.concat(items, ignore_index=True)
    if baseline:
        base_mae = per_item[per_item["model"] == baseline].set_index(id_col)["mae"]
        per_item["fva"] = 1 - per_item["mae"] / per_item[id_col].map(base_mae)

    rows = {}
    for model in cols:
        p = per_item[per_item["model"] == model]
        e = d[model] - d[y]
        sse_sorted = np.sort(p["sse"].to_numpy())[::-1]
        k = max(1, int(round(top_frac * len(sse_sorted))))
        biased = (p["tracking_signal"].abs() > ts_limit).sum()
        r = {
            "n_items": len(p),
            "mae": e.abs().mean(),
            "rmse": float(np.sqrt((e ** 2).mean())),
            "bias": e.mean(),
            "pct_bias": e.sum() / d[y].sum() if d[y].sum() != 0 else np.nan,
            "n_biased_items": int(biased),
            "share_biased_items": biased / len(p),
            "error_concentration": sse_sorted[:k].sum() / sse_sorted.sum() if sse_sorted.sum() else np.nan,
        }
        if baseline:
            base_e = d[baseline] - d[y]
            r["fva"] = 1 - e.abs().mean() / base_e.abs().mean()
            r["share_items_worse"] = float((p["fva"] < 0).mean())
        rows[model] = r
    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.index.name = "model"
    return AuditReport(per_item=per_item, summary=summary, baseline=baseline,
                       top_frac=top_frac, ts_limit=ts_limit)
