import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

import forelint as fl


# ── audit: hand-checked against the eight-day forecast-bias example ─────────
def _eight_days():
    demand = [12, 9, 15, 10, 14, 11, 13, 10]
    eh = [3, 1, 2, 3, 1, 2, 3, 1]
    en = [3, -2, -3, 2, 1, -1, -3, 3]
    return pd.DataFrame({
        "unique_id": "sku1", "ds": range(8), "y": demand,
        "H": np.add(demand, eh), "N": np.add(demand, en),
    })


def test_audit_bias_tracking_signal_and_fva():
    rep = fl.audit(_eight_days(), ["H"], baseline="N")
    s = rep.summary
    assert s.loc["H", "bias"] == pytest.approx(2.0)
    assert s.loc["H", "mae"] == pytest.approx(2.0)
    assert s.loc["H", "rmse"] == pytest.approx(np.sqrt(38 / 8))
    assert s.loc["H", "pct_bias"] == pytest.approx(16 / 94)
    assert s.loc["N", "bias"] == pytest.approx(0.0)
    assert s.loc["N", "mae"] == pytest.approx(2.25)
    # tracking signal = sum of errors / MAD = 16 / 2 = 8 for H, 0 for N
    pi = rep.per_item.set_index("model")
    assert pi.loc["H", "tracking_signal"] == pytest.approx(8.0)
    assert pi.loc["N", "tracking_signal"] == pytest.approx(0.0)
    assert s.loc["H", "n_biased_items"] == 1
    # forecast value added of H over N: 1 - 2.00 / 2.25
    assert s.loc["H", "fva"] == pytest.approx(1 - 2.0 / 2.25)
    assert "forecast value added" in str(rep)


def test_audit_error_concentration():
    rows = []
    for i in range(10):
        err = 10.0 if i == 0 else 1.0
        rows += [{"unique_id": i, "ds": t, "y": 0.0, "m": err} for t in range(4)]
    rep = fl.audit(pd.DataFrame(rows), ["m"], top_frac=0.1)
    # item 0: SSE 400; others: 4 each -> 400 / (400 + 36)
    assert rep.summary.loc["m", "error_concentration"] == pytest.approx(400 / 436)


def test_audit_missing_column():
    with pytest.raises(KeyError):
        fl.audit(_eight_days(), ["nope"])


# ── noise floor ─────────────────────────────────────────────────────────────
def test_noise_floor_values():
    nf = fl.noise_floor(lambda s: [1.0, 2.0, 3.0][s], seeds=[0, 1, 2])
    assert nf.mean == pytest.approx(2.0)
    assert nf.sd == pytest.approx(1.0)
    assert nf.mde == pytest.approx(stats.t.ppf(0.975, 2) * np.sqrt(2))
    assert not nf.is_meaningful(1.0)
    assert nf.is_meaningful(7.0)


def test_noise_floor_needs_three_runs():
    with pytest.raises(ValueError, match="at least 3"):
        fl.noise_floor(lambda s: s, seeds=[0, 1])


# ── Diebold-Mariano ─────────────────────────────────────────────────────────
def test_dm_hand_computed():
    # loss differences d = 1..5 with squared loss: e_a^2 - e_b^2
    ea = np.sqrt(np.array([1, 2, 3, 4, 5], dtype=float))
    eb = np.zeros(5)
    r = fl.diebold_mariano(ea, eb, h=1, loss="squared")
    raw = 3 / np.sqrt(2 / 5)
    expected = raw * np.sqrt((5 + 1 - 2) / 5)
    assert r.statistic == pytest.approx(expected)
    assert r.p_value == pytest.approx(2 * stats.t.sf(expected, df=4))
    assert r.mean_loss_diff == pytest.approx(3.0)


def test_dm_is_antisymmetric_and_detects_a_better_model():
    rng = np.random.default_rng(1)
    good = rng.normal(0, 1, 300)
    bad = rng.normal(0, 2, 300)
    ab = fl.diebold_mariano(bad, good)
    ba = fl.diebold_mariano(good, bad)
    assert ab.statistic == pytest.approx(-ba.statistic)
    assert ab.statistic > 0 and ab.p_value < 1e-6


def test_dm_panel_long_format():
    rng = np.random.default_rng(2)
    rows = []
    for i in range(20):
        for t in range(60):
            y = rng.normal(10, 1)
            rows.append({"unique_id": i, "ds": t, "y": y,
                         "good": y + rng.normal(0, 0.5), "bad": y + rng.normal(0, 1.5)})
    df = pd.DataFrame(rows)
    r = fl.dm_test(df, "bad", "good")
    assert r.n == 60 and r.statistic > 0 and r.p_value < 0.01


# ── model confidence set ────────────────────────────────────────────────────
def _losses(seed=3, n=400):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, n)
    return pd.DataFrame({
        "A": 1.00 + base + rng.normal(0, 0.3, n),
        "B": 1.02 + base + rng.normal(0, 0.3, n),
        "C": 1.60 + base + rng.normal(0, 0.3, n),
    })


def test_mcs_drops_the_clearly_worse_model():
    out = fl.model_confidence_set(_losses(), size=0.10, reps=2000, random_state=0)
    assert not out.loc["C", "in_set"]
    assert out.loc["A", "in_set"] and out.loc["B", "in_set"]
    assert out["pvalue"].is_monotonic_increasing


def test_mcs_agrees_with_arch():
    arch = pytest.importorskip("arch.bootstrap")
    L = _losses()
    ours = fl.model_confidence_set(L, size=0.10, reps=2000, block_size=8, random_state=0)
    ref = arch.MCS(L, size=0.10, reps=2000, block_size=8, method="max", bootstrap="mbb", seed=0)
    ref.compute()
    assert set(ours.index[ours["in_set"]]) == set(ref.included)


# ── splitter ────────────────────────────────────────────────────────────────
def _panel(n_groups=20, n_times=30):
    g = np.repeat(np.arange(n_groups), n_times)
    t = np.tile(np.arange(n_times), n_groups)
    return g, t


def test_group_time_split_holds_out_future_and_new_groups():
    g, t = _panel()
    cv = fl.GroupTimeSplit(t, n_splits=3, gap=2, test_group_frac=0.25, random_state=0)
    folds = list(cv.split(np.zeros(len(g)), groups=g))
    assert len(folds) == 3
    for train, test in folds:
        assert not set(g[train]) & set(g[test])
        assert t[train].max() + 2 < t[test].min()


def test_time_only_split_keeps_all_groups():
    g, t = _panel()
    cv = fl.GroupTimeSplit(t, n_splits=3, holdout="none")
    for train, test in cv.split(np.zeros(len(g))):
        assert set(g[test]) <= set(g[train])
        assert t[train].max() < t[test].min()


def test_group_time_split_validates_inputs():
    g, t = _panel()
    with pytest.raises(ValueError):
        fl.GroupTimeSplit(t, holdout="bogus")
    with pytest.raises(ValueError, match="groups"):
        next(fl.GroupTimeSplit(t).split(np.zeros(len(g))))


# ── split optimism: a leak through the entity shows up ──────────────────────
def test_split_optimism_exposes_entity_leak():
    rng = np.random.default_rng(4)
    g, t = _panel(n_groups=40, n_times=24)
    level = rng.normal(0, 5, 40)
    y = level[g] + 0.05 * t + rng.normal(0, 0.5, len(g))
    X = np.column_stack([g, t])  # the group code lets a model memorise each entity
    model = RandomForestRegressor(n_estimators=50, random_state=0)
    out = fl.split_optimism(model, X, y, groups=g, time=t, n_splits=3)
    assert out.loc["random", "vs_reference"] < 0.5   # random CV hides most of the error
    assert out.loc["group", "vs_reference"] > 0.7


# ── demand classes (Syntetos-Boylan cut-offs 1.32 and 0.49) ─────────────────
def test_classify_demand_four_classes():
    rows = []
    series = {
        "smooth": [5, 5, 5, 5, 5, 5],            # ADI 1, CV2 0
        "erratic": [1, 9, 1, 9, 1, 9],           # ADI 1, CV2 0.64
        "intermittent": [0, 0, 5, 0, 0, 5],      # ADI 3, CV2 0
        "lumpy": [0, 0, 1, 0, 0, 9],             # ADI 3, CV2 0.64
        "none": [0, 0, 0, 0, 0, 0],
    }
    for k, v in series.items():
        rows += [{"unique_id": k, "ds": t, "y": x} for t, x in enumerate(v)]
    out = fl.classify_demand(pd.DataFrame(rows))
    assert out.loc["smooth", "demand_class"] == "smooth"
    assert out.loc["erratic", "demand_class"] == "erratic"
    assert out.loc["intermittent", "demand_class"] == "intermittent"
    assert out.loc["lumpy", "demand_class"] == "lumpy"
    assert out.loc["none", "demand_class"] == "no demand"
    assert out.loc["lumpy", "adi"] == pytest.approx(3.0)
    assert out.loc["lumpy", "cv2"] == pytest.approx(0.64)


# ── MASE / RMSSE, hand-checked ──────────────────────────────────────────────
def test_audit_mase_rmsse():
    train = pd.DataFrame({"unique_id": "a", "ds": range(4), "y": [1.0, 2, 3, 4]})  # naive diffs all 1
    test = pd.DataFrame({"unique_id": "a", "ds": [4, 5], "y": [5.0, 6], "m": [7.0, 4]})  # errors +2, -2
    rep = fl.audit(test, ["m"], train=train)
    assert rep.summary.loc["m", "mase"] == pytest.approx(2.0)
    assert rep.summary.loc["m", "rmsse"] == pytest.approx(2.0)
    assert "RMSSE" in str(rep)


def test_audit_seasonal_scale():
    train = pd.DataFrame({"unique_id": "a", "ds": range(6), "y": [1.0, 5, 1, 5, 2, 6]})  # lag-2 diffs: 0,0,1,1
    test = pd.DataFrame({"unique_id": "a", "ds": [6], "y": [2.0], "m": [3.0]})
    rep = fl.audit(test, ["m"], train=train, season_length=2)
    assert rep.summary.loc["m", "mase"] == pytest.approx(1 / 0.5)


# ── stockout-aware scoring ──────────────────────────────────────────────────
def test_audit_excludes_flagged_rows():
    df = _eight_days()
    df["stockout"] = False
    df.loc[df.ds == 0, "stockout"] = True       # drop the +3 error of day 0
    rep = fl.audit(df, ["H"], exclude="stockout")
    assert rep.n_excluded == 1
    assert rep.summary.loc["H", "bias"] == pytest.approx(13 / 7)
    assert "excluded" in str(rep)


# ── per-segment breakdown ───────────────────────────────────────────────────
def test_audit_by_segment():
    rows = []
    for item, err in [("a", 1.0), ("b", 1.0), ("c", 5.0)]:
        rows += [{"unique_id": item, "ds": t, "y": 10.0, "m": 10.0 + err, "base": 10.0 + 2} for t in range(3)]
    rep = fl.audit(pd.DataFrame(rows), ["m"], baseline="base", segments={"a": "fast", "b": "fast", "c": "slow"})
    seg = rep.by_segment
    assert seg.loc[("m", "fast"), "mae"] == pytest.approx(1.0)
    assert seg.loc[("m", "slow"), "mae"] == pytest.approx(5.0)
    assert seg.loc[("m", "fast"), "fva"] == pytest.approx(0.5)
    assert seg.loc[("m", "slow"), "fva"] == pytest.approx(-1.5)
    assert "by segment" in str(rep)
