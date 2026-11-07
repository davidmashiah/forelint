"""One report on a backtest: bias, error concentration, scaled error and
forecast value added, per item, per segment and overall.

Average error hides the things that cost money: items whose forecast is always
too high or always too low, the few items that carry most of the error, and
whole classes of items where a model loses to a naive forecast while winning on
average. The audit surfaces all three.

Sign convention: error = forecast − actual, so positive bias is over-forecasting.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .compare import _dm_from_diff, origin_means
from .segment import classify_demand

__all__ = ["AuditReport", "audit"]


@dataclass
class AuditReport:
    per_item: pd.DataFrame
    summary: pd.DataFrame
    baseline: str | None
    top_frac: float
    bias_alpha: float
    by_segment: pd.DataFrame | None = None
    n_excluded: int = 0
    n_missing: int = 0
    notes: list = field(default_factory=list)

    def __str__(self):
        lines = ["forelint audit", ""]
        if self.n_excluded:
            lines.append(f"{self.n_excluded} rows excluded from scoring (flagged, e.g. stockouts)")
        if self.n_missing:
            lines.append(f"{self.n_missing} rows dropped for missing actuals or forecasts")
        if self.n_excluded or self.n_missing:
            lines.append("")
        for model, r in self.summary.iterrows():
            lines.append(f"{model}")
            lines.append(f"  MAE {r['mae']:.4g}   RMSE {r['rmse']:.4g}   bias {r['bias']:+.4g} "
                         f"({r['pct_bias']:+.1%} of actuals)")
            if "rmsse" in r and pd.notna(r["rmsse"]):
                lines.append(f"  MASE {r['mase']:.4g}   RMSSE {r['rmsse']:.4g} "
                             f"(mean over {int(r['n_items_scaled'])} scaled items)")
            lines.append(f"  items with significant bias (p < {self.bias_alpha:g}): "
                         f"{int(r['n_biased_items'])} of {int(r['n_items'])} ({r['share_biased_items']:.0%})")
            lines.append(f"  top {int(r['top_k'])} of {int(r['n_items'])} items hold "
                         f"{r['error_concentration']:.0%} of squared error")
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


def _bias_pvalue(e):
    e = e[np.isfinite(e)]
    if e.size < 3:
        return np.nan
    sd = e.std(ddof=1)
    if sd == 0:
        return 0.0 if e.mean() != 0 else 1.0
    t = e.mean() / (sd / np.sqrt(e.size))
    return float(2 * stats.t.sf(abs(t), df=e.size - 1))


def _origin_bias_pvalue(g):
    """Zero-mean test on one mean error per non-overlapping forecast origin."""
    per, _ = origin_means(g["e"].to_numpy(), g["cut"].to_numpy(), g["t"].to_numpy())
    per = per.to_numpy()
    if per.size < 3:
        return np.nan
    if np.ptp(per) <= 1e-12 * max(1.0, float(np.abs(per).max())):
        return 0.0 if per.mean() != 0 else 1.0
    return _dm_from_diff(per, 1).p_value


def _tracking_signal(e):
    mad = np.abs(e).mean()
    return np.nan if mad == 0 else e.sum() / mad


def _scales(train, y, id_col, time_col, m):
    """Per-item in-sample (seasonal) naive MAE and MSE: the MASE / RMSSE denominators.

    Each item's history starts at its first positive value (as in M5), and a
    lag-m difference is used only when the two rows are exactly m regular
    time steps apart, so gaps in the history do not count as one-step changes.
    """
    t = train[[id_col, time_col, y]].dropna().drop_duplicates([id_col, time_col]).sort_values([id_col, time_col])
    mae, mse = {}, {}
    for item, g in t.groupby(id_col, observed=True):
        v = g[y].to_numpy(dtype=float)
        pos = np.flatnonzero(v > 0)
        if pos.size == 0:
            continue
        v, tt = v[pos[0]:], g[time_col].to_numpy()[pos[0]:]
        if v.size <= m:
            continue
        steps = np.diff(tt)
        if steps.size == 0:
            continue
        unit = pd.Series(steps).median()
        if not unit:
            continue
        ok = (tt[m:] - tt[:-m]) == m * unit
        d = (v[m:] - v[:-m])[ok]
        if d.size:
            mae[item], mse[item] = np.abs(d).mean(), (d ** 2).mean()
    return pd.Series(mae, dtype=float), pd.Series(mse, dtype=float)


def _bool_mask(s, name):
    if s.dtype == bool:
        return s
    if pd.api.types.is_integer_dtype(s.dtype) and set(pd.unique(s.dropna())) <= {0, 1}:
        return s.fillna(0).astype(bool)
    if str(s.dtype) == "boolean":
        return s.fillna(False).astype(bool)
    if s.dtype == object and s.dropna().map(lambda v: isinstance(v, (bool, np.bool_))).all():
        return s.fillna(False).astype(bool)
    raise TypeError(f"exclude column {name!r} must be boolean (True = leave out), got {s.dtype}")


def audit(df, models, *, baseline=None, y="y", id_col="unique_id", time_col="ds", cutoff_col="cutoff",
          exclude=None, train=None, season_length=1, segments=None,
          top_frac=0.10, bias_alpha=0.05, ts_window=12) -> AuditReport:
    """Audit a long-format backtest.

    Parameters
    ----------
    df : DataFrame with one row per item, time stamp (and forecast origin, in
        ``cutoff_col``, if there are several), the actual in ``y`` and one
        column per model. statsforecast / mlforecast ``cross_validation``
        output works as is.
    models : list of model column names.
    baseline : optional model column for forecast value added, typically a
        naive or seasonal-naive forecast.
    exclude : optional boolean (or 0/1) column; rows where it is True are left
        out of scoring, and out of the scaling history and the demand
        classification if ``train`` has the column too. Use it for stockout periods, which do not measure demand.
    train : optional training history (same column names). When given, the
        audit adds MASE and RMSSE, scaled by each item's in-sample naive error
        from its first positive value onward.
    season_length : the naive lag used for scaling (1 = naive, 7 = weekly).
    segments : optional segment per item, as a Series or dict indexed by item,
        or ``"demand_class"`` to classify items as smooth / erratic /
        intermittent / lumpy from ``train`` (or from ``df`` without it).
    top_frac : share of items used for the error-concentration figure.
    bias_alpha : an item counts as biased when its mean error differs from
        zero at this level. With a cutoff column the test uses one mean error
        per forecast origin, keeping only origins whose windows do not overlap,
        and needs 3 of them (otherwise it is not run). Without a cutoff column
        it is a t-test on the item's errors, which is only valid for
        independent one-step errors.
    ts_window : the per-item tracking signal (sum of errors / MAD) is
        reported over each item's last ``ts_window`` scored periods, its
        classic use. It is descriptive and not used for the bias count.

    Rows where the actual or any scored forecast is missing are dropped for
    every model, so all models are scored on the same rows.
    """
    models = list(models)
    cols = models + ([baseline] if baseline is not None and baseline not in models else [])
    need = [y, id_col, time_col, *cols] + ([exclude] if exclude is not None else [])
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"columns not found: {missing}")
    notes = []
    has_cut = cutoff_col is not None and cutoff_col in df.columns
    order = [id_col] + ([cutoff_col] if has_cut else []) + [time_col]
    d = df.sort_values(order)
    if not has_cut and d.duplicated([id_col, time_col]).any():
        warnings.warn("duplicate (id, time) rows, as from overlapping cross-validation windows; pass the "
                      f"forecast origin as {cutoff_col!r} so they are kept apart", stacklevel=2)
    n_excluded = 0
    if exclude is not None:
        mask = _bool_mask(d[exclude], exclude)
        n_excluded = int(mask.sum())
        d = d[~mask.to_numpy()]
    valid = d[[y, *cols]].notna().all(axis=1) & np.isfinite(d[[y, *cols]].astype(float)).all(axis=1)
    n_missing = int((~valid).sum())
    d = d[valid]
    if d.empty:
        raise ValueError("no rows left to score")
    item_ids = pd.Index(d[id_col].unique())
    # Bias test: with a cutoff column, one mean error per non-overlapping
    # forecast origin; with fewer than 3 such origins there is no valid test.
    if has_cut:
        kept, overlap = origin_means(np.zeros(len(d)), d[cutoff_col], d[time_col])
        if len(kept) < 3:
            notes.append(f"bias test: only {len(kept)} non-overlapping forecast origin(s); "
                         "per-item bias is not tested (needs 3 or more).")
        elif overlap > 1:
            notes.append(f"bias test: forecast windows overlap, so it uses {len(kept)} non-overlapping "
                         f"origins of {d[cutoff_col].nunique()}.")
    else:
        notes.append("bias test: no cutoff column, so errors are treated as independent one-step "
                     "errors; for multi-step forecasts pass the forecast origin to avoid false alarms.")

    scale_mae = scale_mse = None
    if train is not None:
        tr = train
        if exclude is not None and exclude in train.columns:
            tr = train[~_bool_mask(train[exclude], exclude).to_numpy()]
        scale_mae, scale_mse = _scales(tr, y, id_col, time_col, season_length)
        unscaled = int((~item_ids.isin(scale_mae.index)).sum())
        if unscaled:
            notes.append(f"MASE/RMSSE: {unscaled} of {len(item_ids)} items have no usable history "
                         "(missing from train, constant, or too short) and are left out of the mean.")
            if unscaled == len(item_ids):
                warnings.warn("no item could be scaled; check that train and df use the same id values "
                              "and types", stacklevel=2)

    seg = None
    if segments is not None:
        if isinstance(segments, str) and segments == "demand_class":
            src = train if train is not None else df
            if exclude is not None and exclude in src.columns:
                # flagged periods (e.g. stockouts) do not measure demand: classify without them
                src = src.assign(**{y: src[y].where(~_bool_mask(src[exclude], exclude).to_numpy())})
            seg = classify_demand(src, y=y, id_col=id_col, time_col=time_col)["demand_class"]
        else:
            seg = pd.Series(segments)
        unmapped = int((~item_ids.isin(seg.index)).sum())
        if unmapped:
            notes.append(f"segments: {unmapped} of {len(item_ids)} items have no segment and are left out "
                         "of the breakdown.")
            if unmapped == len(item_ids):
                warnings.warn("no item matched a segment; check the id values and types", stacklevel=2)

    def per_model(model):
        e = (d[model] - d[y]).to_numpy(dtype=float)
        frame = pd.DataFrame({id_col: d[id_col].to_numpy(), "e": e, "y": d[y].to_numpy(dtype=float)})
        if has_cut:
            frame["cut"] = d[cutoff_col].to_numpy()
            frame["t"] = d[time_col].to_numpy()
        rows = {}
        for item, g in frame.groupby(id_col, sort=True, observed=True):
            ev = g["e"].to_numpy()
            r = {"n": ev.size, "mae": np.abs(ev).mean(), "rmse": float(np.sqrt((ev ** 2).mean())),
                 "bias": ev.mean(), "sse": (ev ** 2).sum(), "sum_actual": g["y"].sum(),
                 "bias_pvalue": _origin_bias_pvalue(g) if has_cut else _bias_pvalue(ev),
                 "tracking_signal": _tracking_signal(ev[-ts_window:])}
            if scale_mae is not None:
                sm, ss = scale_mae.get(item, np.nan), scale_mse.get(item, np.nan)
                r["mase"] = r["mae"] / sm if sm else np.nan
                r["rmsse"] = float(np.sqrt((ev ** 2).mean() / ss)) if ss else np.nan
            rows[item] = r
        part = pd.DataFrame.from_dict(rows, orient="index")
        part.index.name = id_col
        part["model"] = model
        return part.reset_index()

    per_item = pd.concat([per_model(m) for m in cols], ignore_index=True)
    if baseline is not None:
        base_mae = per_item[per_item["model"] == baseline].set_index(id_col)["mae"]
        bm = per_item[id_col].map(base_mae)
        per_item["fva"] = np.where(bm > 0, 1 - per_item["mae"] / bm, np.nan)
    if seg is not None:
        per_item["segment"] = per_item[id_col].map(seg)

    def summarise(p, dd, model):
        e = dd[model] - dd[y]
        sse_sorted = np.sort(p["sse"].to_numpy())[::-1]
        k = max(1, int(np.floor(top_frac * len(sse_sorted) + 0.5)))
        biased = int((p["bias_pvalue"] < bias_alpha).sum())
        r = {
            "n_items": len(p),
            "mae": e.abs().mean(),
            "rmse": float(np.sqrt((e ** 2).mean())),
            "bias": e.mean(),
            "pct_bias": e.sum() / dd[y].sum() if dd[y].sum() != 0 else np.nan,
            "n_biased_items": biased,
            "share_biased_items": biased / len(p) if len(p) else np.nan,
            "top_k": k,
            "error_concentration": sse_sorted[:k].sum() / sse_sorted.sum() if sse_sorted.sum() else np.nan,
        }
        if "rmsse" in p:
            r["n_items_scaled"] = int(p["rmsse"].notna().sum())
            r["mase"] = p["mase"].mean()
            r["rmsse"] = p["rmsse"].mean()
        if baseline is not None:
            be = (dd[baseline] - dd[y]).abs().mean()
            r["fva"] = 1 - e.abs().mean() / be if be > 0 else np.nan
            f = p["fva"].dropna()
            r["share_items_worse"] = float((f < 0).mean()) if len(f) else np.nan
        return r

    rows, seg_rows = {}, {}
    seg_of_row = d[id_col].map(seg) if seg is not None else None
    for model in cols:
        p = per_item[per_item["model"] == model]
        rows[model] = summarise(p, d, model)
        if seg is not None:
            for s, ps in p.groupby("segment", observed=True):
                seg_rows[(model, s)] = summarise(ps, d[(seg_of_row == s).to_numpy()], model)
    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.index.name = "model"
    by_segment = None
    if seg_rows:
        by_segment = pd.DataFrame.from_dict(seg_rows, orient="index")
        by_segment.index.names = ["model", "segment"]
    return AuditReport(per_item=per_item, summary=summary, baseline=baseline, top_frac=top_frac,
                       bias_alpha=bias_alpha, by_segment=by_segment, n_excluded=n_excluded,
                       n_missing=n_missing, notes=notes)
