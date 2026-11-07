# forelint

[![PyPI](https://img.shields.io/pypi/v/forelint.svg)](https://pypi.org/project/forelint/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24044927.svg)](https://doi.org/10.5281/zenodo.24044927)

**Lint your forecast backtests before you trust a score.**

forelint checks the things a backtest score hides: leakage between related
series, the run-to-run noise floor, whether model A is really better than
model B, and which items carry the bias and the error. It works on the
long-format tables that statsforecast, mlforecast and most forecasting code
already produce, and on any scikit-learn estimator.

```bash
pip install forelint
```

## What it checks

| Question | Function |
|---|---|
| Does my validation leak through related stores, products or regions? | `split_optimism`, `GroupTimeSplit` |
| Is this improvement bigger than seed-to-seed noise? | `noise_floor` |
| Is model A really better than B across many series? | `dm_test`, `diebold_mariano`, `model_confidence_set` |
| Which items are always over or under forecast? Does the model beat a naive forecast? Where does the error live? | `audit` |
| Does the model win on steady items and lose on intermittent ones? | `classify_demand`, `audit(segments="demand_class")` |
| What is the scaled error (MASE, RMSSE) per item? | `audit(train=history)` |
| How do I score without counting stockout days as low demand? | `audit(exclude="stockout")` |

## Quickstart

```python
import forelint as fl

# df: one row per item and date, the actual in "y", one column per model,
# e.g. the output of statsforecast's cross_validation()
report = fl.audit(df, models=["LightGBM", "ETS"], baseline="SeasonalNaive",
                  train=history,              # adds MASE and RMSSE
                  segments="demand_class",    # smooth / erratic / intermittent / lumpy
                  exclude="stockout")         # optional: skip stockout periods
print(report)

# Is LightGBM really better than the seasonal naive, or is it noise?
print(fl.dm_test(df, "LightGBM", "SeasonalNaive"))
```

```python
# How much does random K-fold flatter the model, compared with holding out
# the future of entities the model never saw?
fl.split_optimism(model, X, y, groups=store_ids, time=dates)

# What difference between two runs is just noise?
nf = fl.noise_floor(lambda seed: train_and_score(seed), seeds=range(10))
nf.is_meaningful(new_mae - old_mae)
```

## With statsforecast

`examples/statsforecast_cv.py` runs `StatsForecast.cross_validation` with AutoETS,
Croston, seasonal naive and naive on 60 daily series, then audits the output
as it comes out. AutoETS wins overall and is the only model left in the
confidence set, yet on the intermittent series it is still worse than the
naive forecast on 59% of items. That is the kind of result an average hides.

## On real data: one Walmart store in M5

`benchmarks/m5_benchmark.py` runs every check on the public M5 data: store
CA_1, department FOODS_3, 823 items, the last 365 days, a direct 28-day-ahead
gradient-boosting model from scikit-learn, scored on the final 28 days.
Measured, not assumed:

- **The model added nothing over a 28-day moving average.** Forecast value
  added was **−0.1%** on MAE, the model was worse than the moving average on
  **49% of items**, and Diebold-Mariano gives **p = 0.31**. The model
  confidence set keeps both. An average-MAE table alone would not show this.
- **The average hid a split by demand class.** Against the moving average, the
  model added **+6.2%** on the 156 smooth items but **−3.0%** on the 491
  intermittent items and **−5.0%** on the 146 lumpy ones, losing on more than
  half of each. Intermittent items are 60% of this store. The overall −0.1% was
  the two cancelling out.
- **It under-forecasts by 9.3%** of actual sales.
- **RMSSE** (M5's official metric) is 0.911 for the model against 0.907 for the
  moving average.
- **10% of items hold 73% of the squared error.**
- **The noise floor is small here:** over 10 seeds the MAE sd is 0.0014, so
  any change under 0.0045 is noise.
- **Splitting barely mattered for this data:** random K-fold was 1% optimistic
  against a time split. Leakage depends on the features; forelint measures it
  rather than assuming it.

The model is deliberately plain (six features, no tuning). The point is the
audit, not the model.

## Why

In an international machine-learning competition on well-geology prediction,
validation ranked five builds against the hidden leaderboard with a
correlation of −0.745: the build it liked most scored worst, and identical code scored 6.241, 6.259 and
6.368 on three runs. Both failures were measurable before the deadline with
the right checks. forelint is those checks, packaged.

Background: [My validation ranked five models backwards](https://davidmashiah.com/notes/validation-ranked-my-models-backwards).

## Conventions

- Error is **forecast − actual**. Positive bias means over-forecasting.
- Tracking signal is the sum of errors divided by the mean absolute error;
  the default alarm level is ±4.
- `dm_test` averages the loss difference across series at each time stamp,
  then applies Diebold-Mariano with the Harvey-Leybourne-Newbold correction.
- `model_confidence_set` follows Hansen, Lunde and Nason (2011) with the
  T_max statistic and a moving-block bootstrap. Its kept set matches
  `arch.bootstrap.MCS` in the test suite.

## Cite

If forelint helps your work, please cite it:

> Mashiah, D. (2026). forelint: lint your forecast backtests before you trust a score. Zenodo. https://doi.org/10.5281/zenodo.24044927

## Status

Alpha. The API may change before 1.0. Issues and pull requests are welcome.

## Author

[David Mashiah](https://davidmashiah.com). MIT licensed.
