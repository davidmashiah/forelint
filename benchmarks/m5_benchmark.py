"""forelint on real retail data: the M5 competition, one Walmart store.

Usage:  python benchmarks/m5_benchmark.py path/to/m5.zip [STORE] [DEPT]

Reads the public M5 sales file (from the Kaggle competition or Nixtla's
mirror), keeps one store and department, and builds a direct 28-day-ahead
global model with scikit-learn. Every number this prints is measured on that
data; nothing is assumed.
"""
from __future__ import annotations

import sys
import time
import zipfile

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import forelint as fl

H = 28          # forecast horizon, as in M5
HISTORY = 420   # days of history kept


def load(path, store, dept):
    with zipfile.ZipFile(path) as z, z.open("sales_train_evaluation.csv") as f:
        wide = pd.read_csv(f)
    wide = wide[(wide.store_id == store) & (wide.dept_id == dept)]
    days = [c for c in wide.columns if c.startswith("d_")][-HISTORY:]
    long = wide.melt(id_vars=["item_id"], value_vars=days, var_name="d", value_name="y")
    long["ds"] = long["d"].str[2:].astype(int)
    return long[["item_id", "ds", "y"]].sort_values(["item_id", "ds"]).reset_index(drop=True)


def features(df):
    g = df.groupby("item_id")["y"]
    out = df.copy()
    out["lag28"] = g.shift(H)
    out["lag35"] = g.shift(H + 7)
    out["rmean7"] = g.transform(lambda s: s.shift(H).rolling(7).mean())
    out["rmean28"] = g.transform(lambda s: s.shift(H).rolling(28).mean())
    out["dow"] = out["ds"] % 7
    out["item"] = out["item_id"].astype("category").cat.codes
    return out.dropna().reset_index(drop=True)


FEATS = ["item", "dow", "lag28", "lag35", "rmean7", "rmean28"]


def main(path, store="CA_1", dept="FOODS_3"):
    t0 = time.time()
    df = features(load(path, store, dept))
    print(f"M5 {store} {dept}: {df.item_id.nunique()} items, {df.ds.nunique()} days, {len(df):,} rows\n")
    X, y = df[FEATS].to_numpy(float), df["y"].to_numpy(float)
    model = HistGradientBoostingRegressor(max_iter=200, random_state=0)

    # 1. How much each split strategy flatters the model.
    opt = fl.split_optimism(model, X, y, groups=df["item"].to_numpy(), time=df["ds"].to_numpy(),
                            reference="time", n_splits=5, gap=0, random_state=0)
    print("1. split optimism (MAE; reference = time split, the way the model is used)")
    print(opt.round(4).to_string(), "\n")

    # Final holdout: the last 28 days.
    cut = df.ds.max() - H
    tr, te = df[df.ds <= cut], df[df.ds > cut].copy()

    # 2. Run-to-run noise floor: same data, same settings, different seeds.
    def run(seed):
        m = HistGradientBoostingRegressor(max_iter=200, random_state=seed, early_stopping=True)
        m.fit(tr[FEATS], tr["y"])
        return float(np.abs(m.predict(te[FEATS]) - te["y"]).mean())
    nf = fl.noise_floor(run, seeds=range(10))
    print("2.", nf)
    print(f"   run MAEs: {np.round(nf.scores, 4).tolist()}\n")

    # 3. Audit against two naive baselines.
    m = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(tr[FEATS], tr["y"])
    te["GBM"] = m.predict(te[FEATS])
    te["Naive28"] = te["lag28"]
    last = tr.groupby("item_id")["y"].apply(lambda s: s.iloc[-28:].mean())
    te["MA28"] = te["item_id"].map(last)
    hist = df[df.ds <= cut][["item_id", "ds", "y"]]
    rep = fl.audit(te, ["GBM", "Naive28"], baseline="MA28", y="y", id_col="item_id", time_col="ds",
                   train=hist, segments="demand_class")
    print("3.", rep, "\n")

    # 4. Is the GBM really better than the moving average?
    print("4. Diebold-Mariano, GBM vs MA28 (squared loss, h = 1 on the per-day mean):")
    print("  ", fl.dm_test(te, "GBM", "MA28", id_col="item_id"))
    losses = pd.DataFrame({c: ((te[c] - te["y"]) ** 2).groupby(te["ds"]).mean() for c in ["GBM", "Naive28", "MA28"]})
    print("   model confidence set (10%):")
    print(fl.model_confidence_set(losses, size=0.10, random_state=0).round(4).to_string())
    print(f"\nran in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(*sys.argv[1:])
