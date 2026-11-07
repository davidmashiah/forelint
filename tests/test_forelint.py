import warnings

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
    ref = arch.MCS(L, size=0.10, reps=2000, block_size=8, method="max", bootstrap="cbb", seed=0)
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


# ── split optimism: matched comparisons ─────────────────────────────────
def _shock_panel(seed=4, n_groups=40, n_times=60):
    rng = np.random.default_rng(seed)
    g, t = _panel(n_groups=n_groups, n_times=n_times)
    level = rng.normal(0, 1, n_groups)
    shock = rng.normal(0, 3, n_times)          # a daily shock every item shares
    y = level[g] + shock[t] + rng.normal(0, 0.5, len(g))
    return np.column_stack([g, t]).astype(float), y, g, t


def test_split_optimism_exposes_cross_series_leak():
    X, y, g, t = _shock_panel()
    out = fl.split_optimism(RandomForestRegressor(n_estimators=60, random_state=0), X, y,
                            groups=g, time=t, n_splits=4)
    # same-day rows of other items reveal the shock: both leaky splits look far better
    assert out.loc["random", "optimism"] > 0.3
    assert out.loc["group", "optimism"] > 0.2
    assert out.loc["random", "honest"] == "time"


def test_split_optimism_is_near_zero_without_a_leak():
    from sklearn.neighbors import KNeighborsRegressor
    rng = np.random.default_rng(5)
    g, t = _panel(n_groups=40, n_times=60)
    X = rng.normal(size=(len(g), 3))
    y = X[:, 0] + rng.normal(0, 1, len(g))     # no dependence on group or time
    out = fl.split_optimism(KNeighborsRegressor(5), X, y, groups=g, time=t, n_splits=4)
    assert abs(out.loc["random", "optimism"]) < 0.08
    assert abs(out.loc["group", "optimism"]) < 0.08


def test_split_optimism_rejects_score_metrics():
    X, y, g, t = _shock_panel()
    with pytest.raises(ValueError, match="neg_"):
        fl.split_optimism(RandomForestRegressor(n_estimators=5), X, y, groups=g, time=t, scoring="r2")


# ── demand classes (Syntetos-Boylan cut-offs 1.32 and 0.49) ─────────────────
def test_classify_demand_four_classes():
    rows = []
    series = {
        "smooth": [5, 5, 5, 5, 5, 5],            # ADI 1, CV2 0
        "erratic": [1, 9, 1, 9, 1, 9],           # ADI 1, CV2 0.64
        "intermittent": [5, 0, 0, 5, 0, 0],      # ADI 3, CV2 0
        "lumpy": [0, 0, 1, 0, 0, 9],             # ADI 2 from first demand, CV2 0.64
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
    assert out.loc["lumpy", "adi"] == pytest.approx(2.0)   # (6 periods − first demand at 2) / 2 demands
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


# ── review fixes ────────────────────────────────────────────────────────────
def test_bias_flag_has_the_right_null_rate():
    rng = np.random.default_rng(7)
    rows = []
    for i in range(500):
        e = rng.normal(0, 1, 28)
        rows += [{"unique_id": i, "ds": t, "y": 10.0, "m": 10.0 + e[t]} for t in range(28)]
    rep = fl.audit(pd.DataFrame(rows), ["m"])
    assert rep.summary.loc["m", "share_biased_items"] < 0.09   # about 5% expected under no bias


def _cv_panel(rng, n_series=30, n_cut=20, horizon=7, step=7, effect=0.0):
    rows = []
    for c in range(n_cut):
        for i in range(n_series):
            shock = rng.normal(0, 1)                         # shared by all steps of one origin
            for k in range(1, horizon + 1):
                ds = c * step + k
                y = 10.0
                rows.append({"unique_id": i, "ds": ds, "cutoff": c * step, "y": y,
                             "A": y + shock + rng.normal(0, 1), "B": y + shock + rng.normal(0, 1) + effect})
    return pd.DataFrame(rows)


def test_dm_test_uses_cutoffs_and_holds_size():
    rng = np.random.default_rng(8)
    rejections = 0
    for _ in range(60):
        r = fl.dm_test(_cv_panel(rng), "A", "B")
        rejections += r.p_value < 0.05
    assert rejections / 60 < 0.15                    # was 0.46 when steps counted as separate points
    r = fl.dm_test(_cv_panel(rng, effect=1.0), "A", "B")
    assert r.n == 20 and r.p_value < 0.01 and r.statistic < 0


def test_dm_test_thins_overlapping_windows():
    rng = np.random.default_rng(9)
    df = _cv_panel(rng, n_cut=12, horizon=14, step=7)
    with pytest.warns(UserWarning, match="non-overlapping"):
        r = fl.dm_test(df, "A", "B")
    assert r.n == 6 and r.h == 1


def _daily_origin_null(rng, n_series=20, n_dates=80, horizon=7, n_cut=40):
    """Equally good models; each model's error at a target date is shared by every
    origin that covers it, as in a real rolling backtest with step 1."""
    ea = rng.normal(0, 1, (n_series, n_dates))
    eb = rng.normal(0, 1, (n_series, n_dates))
    rows = []
    for c in range(n_cut):
        for i in range(n_series):
            for k in range(1, horizon + 1):
                t = c + k
                rows.append((i, t, c, 0.0, ea[i, t], eb[i, t]))
    return pd.DataFrame(rows, columns=["unique_id", "ds", "cutoff", "y", "A", "B"])


def test_dm_test_size_with_daily_origins():
    rng = np.random.default_rng(12)
    rej = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(150):
            rej += fl.dm_test(_daily_origin_null(rng), "A", "B").p_value < 0.05
    assert rej / 150 < 0.10                       # was 0.17 with a lag allowance instead of thinning


def test_dm_test_refuses_duplicates_without_cutoff():
    rng = np.random.default_rng(10)
    df = _cv_panel(rng, n_cut=5, horizon=14, step=7).drop(columns="cutoff")
    with pytest.raises(ValueError, match="duplicate"):
        fl.dm_test(df, "A", "B")
    with pytest.raises(KeyError):
        fl.dm_test(df.assign(cutoff=0), "A", "B", id_col="nope")


def test_dm_edge_cases():
    with pytest.raises(ValueError):
        fl.diebold_mariano([1, 2, 3, 4], [0, 0, 0, 0], h=0)
    with pytest.raises(ValueError, match="NaN"):
        fl.diebold_mariano([1, np.nan, 3, 4, 5], [0, 0, 0, 0, 0], h=2)
    assert "no difference" in str(fl.DMResult(0.0, 1.0, 0.0, 5))


def test_rmsse_follows_m5_and_ignores_leading_zeros():
    train = pd.DataFrame({"unique_id": "a", "ds": range(10), "y": [0, 0, 0, 0, 0, 0, 2, 4, 2, 4.0]})
    test = pd.DataFrame({"unique_id": "a", "ds": [10, 11], "y": [2.0, 4], "m": [3.0, 3]})
    assert fl.audit(test, ["m"], train=train).summary.loc["m", "rmsse"] == pytest.approx(0.5)


def test_scale_skips_gaps_in_time():
    train = pd.DataFrame({"unique_id": "a", "ds": [0, 1, 2, 10, 11, 12], "y": [1.0, 2, 3, 50, 51, 52]})
    test = pd.DataFrame({"unique_id": "a", "ds": [13], "y": [53.0], "m": [54.0]})
    assert fl.audit(test, ["m"], train=train).summary.loc["m", "mase"] == pytest.approx(1.0)


def test_adi_counts_from_first_demand():
    df = pd.DataFrame({"unique_id": "new", "ds": range(12), "y": [0] * 8 + [5, 5, 5, 5]})
    out = fl.classify_demand(df)
    assert out.loc["new", "adi"] == pytest.approx(1.0)
    assert out.loc["new", "demand_class"] == "smooth"


def test_audit_categorical_ids_have_no_phantom_items():
    df = _eight_days()
    df["unique_id"] = pd.Categorical(df["unique_id"], categories=["sku1", "unused"])
    rep = fl.audit(df, ["H"], baseline="N")
    assert rep.summary.loc["H", "n_items"] == 1


def test_audit_drops_missing_rows_for_every_model():
    df = pd.DataFrame({"unique_id": "a", "ds": range(4), "y": [10.0, 10, 10, 100],
                       "m": [11.0, 11, 11, np.nan], "base": [12.0, 12, 12, 100]})
    rep = fl.audit(df, ["m"], baseline="base")
    assert rep.n_missing == 1
    assert rep.summary.loc["m", "fva"] == pytest.approx(0.5)
    assert rep.summary.loc["m", "pct_bias"] == pytest.approx(0.1)


def test_exclude_column_types():
    df = _eight_days()
    df["flag"] = pd.array([None, None] + [False] * 6, dtype="boolean")
    assert fl.audit(df, ["H"], exclude="flag").n_excluded == 0
    df["flag"] = ["False"] * 8
    with pytest.raises(TypeError):
        fl.audit(df, ["H"], exclude="flag")


def test_zero_error_baseline_gives_no_fva():
    df = pd.DataFrame({"unique_id": ["a"] * 3 + ["b"] * 3, "ds": [0, 1, 2] * 2, "y": [1.0] * 6,
                       "m": [2.0, 2, 2, 1.5, 1.5, 1.5], "base": [1.0, 1, 1, 3, 3, 3]})
    rep = fl.audit(df, ["m"], baseline="base")
    pi = rep.per_item.set_index(["model", "unique_id"])
    assert np.isnan(pi.loc[("m", "a"), "fva"])
    assert rep.summary.loc["m", "share_items_worse"] == pytest.approx(0.0)


def test_mismatched_ids_are_reported():
    train = pd.DataFrame({"unique_id": 1, "ds": range(5), "y": [1.0, 2, 3, 4, 5]})
    test = pd.DataFrame({"unique_id": "1", "ds": [5], "y": [6.0], "m": [7.0]})
    with pytest.warns(UserWarning, match="same id"):
        rep = fl.audit(test, ["m"], train=train)
    assert any("no usable history" in n for n in rep.notes)


def test_concentration_label_uses_actual_count():
    rows = [{"unique_id": i, "ds": 0, "y": 0.0, "m": float(i + 1)} for i in range(2)]
    rep = fl.audit(pd.DataFrame(rows), ["m"])
    assert "top 1 of 2 items" in str(rep)


def test_noise_floor_validation():
    with pytest.raises(ValueError):
        fl.noise_floor(lambda s: float(s), seeds=[0, 1, 2], confidence=95)
    with pytest.raises(ValueError, match="NaN"):
        fl.noise_floor(lambda s: [1.0, np.nan, 2.0][s], seeds=[0, 1, 2])


def test_group_time_split_rejects_bad_input():
    g, t = _panel()
    t = t.astype(float)
    t[0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        next(fl.GroupTimeSplit(t).split(np.zeros(len(g)), groups=g))
    with pytest.raises(ValueError):
        fl.GroupTimeSplit(t, test_size=0)


def test_bias_flag_null_rate_with_overlapping_origins():
    rng = np.random.default_rng(11)
    df = _cv_panel(rng, n_series=300, n_cut=20, horizon=14, step=7)
    assert fl.audit(df, ["A"]).summary.loc["A", "share_biased_items"] < 0.12


def test_dm_falls_back_when_variance_is_negative():
    d = np.array([1.0, -1.0] * 6) + 0.5        # strong negative lag-1 autocorrelation
    r = fl.diebold_mariano(np.sqrt(np.clip(d, 0, None)), np.sqrt(np.clip(-d, 0, None)), h=2)
    assert np.isfinite(r.statistic) and np.isfinite(r.p_value)



def test_bias_flag_size_with_daily_origins():
    rng = np.random.default_rng(13)
    df = _daily_origin_null(rng, n_series=300, n_dates=200, horizon=7, n_cut=150)
    assert fl.audit(df, ["A"]).summary.loc["A", "share_biased_items"] < 0.10


def test_bias_not_tested_with_too_few_origins():
    rng = np.random.default_rng(14)
    df = _cv_panel(rng, n_cut=2, horizon=7, step=7)
    rep = fl.audit(df, ["A"])
    assert rep.summary.loc["A", "n_biased_items"] == 0
    assert rep.per_item["bias_pvalue"].isna().all()
    assert any("not tested" in n for n in rep.notes)


def test_bias_tolerance_is_relative():
    rows = []
    for c, e in enumerate([1e-9, -1e-9, 1e-9, -1e-9]):
        rows.append({"unique_id": "a", "ds": c * 7 + 1, "cutoff": c * 7, "y": 0.0, "m": e})
    rep = fl.audit(pd.DataFrame(rows), ["m"])
    assert rep.per_item.loc[0, "bias_pvalue"] > 0.05


def test_stockouts_do_not_drive_the_demand_class():
    train = pd.DataFrame({"unique_id": "a", "ds": range(8), "y": [5.0, 0, 0, 0, 5, 5, 5, 5],
                          "stockout": [False, True, True, True, False, False, False, False]})
    test = pd.DataFrame({"unique_id": "a", "ds": [8], "y": [5.0], "m": [5.5], "stockout": [False]})
    rep = fl.audit(test, ["m"], train=train, exclude="stockout", segments="demand_class")
    assert rep.per_item.loc[0, "segment"] == "smooth"


def test_exclude_accepts_zero_one_flags():
    df = _eight_days()
    df["flag"] = [1] + [0] * 7
    assert fl.audit(df, ["H"], exclude="flag").n_excluded == 1


def test_split_optimism_exposes_entity_leak():
    rng = np.random.default_rng(4)
    g, t = _panel(n_groups=40, n_times=24)
    level = rng.normal(0, 5, 40)
    y = level[g] + rng.normal(0, 0.5, len(g))
    X = np.column_stack([g, t]).astype(float)  # the entity code lets a model memorise each entity
    out = fl.split_optimism(RandomForestRegressor(n_estimators=50, random_state=0), X, y,
                            groups=g, time=t, n_splits=3)
    assert out.loc["kfold", "optimism"] > 0.6     # plain K-fold hides most of the error
    assert out.loc["kfold", "honest"] == "group"
    assert abs(out.loc["random", "optimism"]) < 3 * out.loc["random", "optimism_se"] + 0.05
