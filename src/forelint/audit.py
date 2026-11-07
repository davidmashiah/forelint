"""One report on a backtest: bias, tracking signal, error concentration,
scaled error and forecast value added, per item, per segment and overall.

Average error hides the things that cost money: items whose forecast is always
too high or always too low, the few items that carry most of the error, and
whole classes of items where a model loses to a naive forecast while winning on
average. The audit surfaces all three.

Sign convention: error = forecast − actual, so positive bias is over-forecasting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .segment import classify_demand

__all__ = ["AuditReport", "audit"]


@dataclass
class AuditReport:
    per_item: pd.DataFrame
    summary: pd.DataFrame
    baseline: str | None
    top_frac: float
    ts_limit: float
    by_segment: pd.DataFrame | None = None
    n_excluded: int = 0
    notes: list = field(default_factory=list)

    def __str__(self):
        lines = ["forelint audit", ""]
        if self.n_excluded:
            lines += [f"{self.n_excluded} rows excluded from scoring (e.g. stockouts)", ""]
        for model, r in self.summary.iterrows():
            lines.append(f"{model}")
            lines.append(f"  MAE {r['mae']:.4g}   RMSE {r['rmse']:.4g}   bias {r['bias']:+.4g} "
                         f"({r['pct_bias']:+.1%} of actuals)")
            if "rmsse" in r and pd.notna(r["rmsse"]):
                lines.append(f"  MASE {r['mase']:.4g}   RMSSE {r['rmsse']:.4g} (mean over items)")
            lines.append(f"  items with |tracking signal| > {self.ts_limit:g}: "
                         f"{int(r['n_biased_items'])} of {int(r['n_items'])} ({r['share_biased_items']:.0%})")
            lines.append(f"  top {self.top_frac:.0%} of items hold {r['error_concentration']:.0%} of squared error")
            if self.baseline is not None and model != self.baseline:
                lines.append(f"  forecast value added vs {self.baseline}: {r['fva']:+.1%} on MAE; "
                             f"worse than {self.baseline} on {r['share_items_worse']:.0%} of items")
            lines.append("")
        if self.by_segment is not None:
            lines.append("by segment")
            cols = [c for c in ["n_items", "mae", "bias", "rmsse", "fva", "share_items_worse"] if c in self.by_segment]
            lines.append(self.by_segment[cols].round(4).to_string())
            lines.append("")
        lines += self.notes
        return "\n".join(lines).rstrip()


def _tracking_signal(e):
    mad = np.abs(e).mean()
    return np.nan if mad == 0 else e.sum() / mad


def _scales(train, y, id_col, time_col, m):
    """In-sample (seasonal) naive MAE and MSE per item: the MASE / RMSSE denominators."""
    t = train.sort_values([id_col, time_col])
    diff = t.groupby(id_col)[y].diff(m)
    g = pd.DataFrame({id_col: t[id_col], "d": diff}).dropna().groupby(id_col)["d"]
    return g.apply(lambda s: s.abs().mean()), g.apply(lambda s: (s ** 2).mean())


def audit(df, models, *, baseline=None, y="y", id_col="unique_id", time_col="ds",
          exclude=None, train=None, season_length=1, segments=None,
          top_frac=0.10, ts_limit=4.0) -> AuditReport:
    """Audit a long-format backtest.

    Parameters
    ----------
    df : DataFrame with one row per item and time stamp, the actual in ``y``
        and one column per model. statsforecast / mlforecast
        ``cross_validation`` output works as is.
    models : list of model column names.
    baseline : optional model column for forecast value added, typically a
        naive or seasonal-naive forecast.
    exclude : optional name of a boolean column; rows where it is True are left
        out of scoring. Use it for stockout periods, where sales stopped at the
        stock on hand and do not measure demand.
    train : optional training history (same column names). When given, the
        audit adds MASE and RMSSE, scaled by each item's in-sample naive error.
    season_length : the naive lag used for scaling (1 = naive, 7 = weekly).
    segments : optional segment per item, as a Series or dict indexed by item,
        or ``"demand_class"`` to classify items as smooth / erratic /
        intermittent / lumpy from ``train`` (or from ``df`` without it).
    top_frac : share of items used for the error-concentration figure.
    ts_limit : tracking-signal alarm level; 4 is a common choice.

    Returns
    -------
    AuditReport with ``per_item``, ``summary`` and, with ``segments``,
    ``by_segment``. ``print(report)`` gives a readable summary.
    """
    models = list(models)
    cols = models + ([baseline] if baseline and baseline not in models else [])
    need = [y, id_col, time_col, *cols] + ([exclude] if exclude else [])
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"columns not found: {missing}")
    d = df.sort_values([id_col, time_col])
    n_excluded = 0
    if exclude:
        mask = d[exclude].astype(bool)
        n_excluded = int(mask.sum())
        d = d[~mask]

    scale_mae = scale_mse = None
    if train is not None:
        scale_mae, scale_mse = _scales(train, y, id_col, time_col, season_length)

    seg = None
    if segments is not None:
        if isinstance(segments, str) and segments == "demand_class":
            src = train if train is not None else df
            seg = classify_demand(src, y=y, id_col=id_col, time_col=time_col)["demand_class"]
        else:
            seg = pd.Series(segments)

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
        if scale_mae is not None:
            part["mase"] = part["mae"] / scale_mae.reindex(part.index).replace(0, np.nan)
            part["rmsse"] = np.sqrt(g["se"].mean() / scale_mse.reindex(part.index).replace(0, np.nan))
        part["model"] = model
        items.append(part.reset_index())
    per_item = pd.concat(items, ignore_index=True)
    if baseline:
        base_mae = per_item[per_item["model"] == baseline].set_index(id_col)["mae"]
        per_item["fva"] = 1 - per_item["mae"] / per_item[id_col].map(base_mae)
    if seg is not None:
        per_item["segment"] = per_item[id_col].map(seg)

    def summarise(p, dd):
        e = dd["_e"]
        sse_sorted = np.sort(p["sse"].to_numpy())[::-1]
        k = max(1, int(round(top_frac * len(sse_sorted))))
        biased = int((p["tracking_signal"].abs() > ts_limit).sum())
        r = {
            "n_items": len(p),
            "mae": e.abs().mean(),
            "rmse": float(np.sqrt((e ** 2).mean())),
            "bias": e.mean(),
            "pct_bias": e.sum() / dd[y].sum() if dd[y].sum() != 0 else np.nan,
            "n_biased_items": biased,
            "share_biased_items": biased / len(p) if len(p) else np.nan,
            "error_concentration": sse_sorted[:k].sum() / sse_sorted.sum() if sse_sorted.sum() else np.nan,
        }
        if "rmsse" in p:
            r["mase"] = p["mase"].mean()
            r["rmsse"] = p["rmsse"].mean()
        if baseline:
            r["fva"] = 1 - e.abs().mean() / dd["_be"].abs().mean()
            r["share_items_worse"] = float((p["fva"] < 0).mean())
        return r

    rows, seg_rows = {}, {}
    for model in cols:
        p = per_item[per_item["model"] == model]
        dd = pd.DataFrame({id_col: d[id_col], y: d[y], "_e": d[model] - d[y]})
        if baseline:
            dd["_be"] = d[baseline] - d[y]
        rows[model] = summarise(p, dd)
        if seg is not None:
            dd["_seg"] = dd[id_col].map(seg)
            for s, ps in p.groupby("segment"):
                seg_rows[(model, s)] = summarise(ps, dd[dd["_seg"] == s])
    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.index.name = "model"
    by_segment = None
    if seg_rows:
        by_segment = pd.DataFrame.from_dict(seg_rows, orient="index")
        by_segment.index.names = ["model", "segment"]
    notes = []
    if exclude:
        notes.append("Scored rows exclude flagged periods; error on them is not counted.")
    return AuditReport(per_item=per_item, summary=summary, baseline=baseline, top_frac=top_frac,
                       ts_limit=ts_limit, by_segment=by_segment, n_excluded=n_excluded, notes=notes)
