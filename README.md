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

# df: one row per item, date and forecast origin ("cutoff"), the actual in "y",
# one column per model, e.g. the output of statsforecast's cross_validation()
report = fl.audit(df, models=["LightGBM", "ETS"], baseline="SeasonalNaive",
                  train=history,              # adds MASE and RMSSE
                  segments="demand_class",    # smooth / erratic / intermittent / lumpy
                  exclude="stockout")         # optional: skip stockout periods
print(report)

# Is LightGBM really better than the seasonal naive, or is it noise?
# (uses the cutoff column, one point per forecast origin)
print(fl.dm_test(df, "LightGBM", "SeasonalNaive"))
```

```python
# How much does a leaky split flatter the model, against an honest split on
# the same test rows and training size?
fl.split_optimism(model, X, y, groups=store_ids, time=dates)

# What difference between two runs is just noise?
nf = fl.noise_floor(lambda seed: train_and_score(seed), seeds=range(10))
nf.is_meaningful(new_mae - old_mae)
```

## With statsforecast

`examples/statsforecast_cv.py` runs `StatsForecast.cross_validation` with AutoETS,
Croston, seasonal naive and naive on 60 daily series at 12 forecast origins
whose windows do not overlap, then audits the output as it comes out, using
the `cutoff` column so each origin counts once. AutoETS is the only model left
in the confidence set, yet on the intermittent series it is still worse than
the naive forecast on about half of them.

## On real data: one Walmart store in M5

`benchmarks/m5_benchmark.py` runs every check on the public M5 data: store
CA_1, department FOODS_3, 823 items, a direct 28-day-ahead gradient-boosting
model from scikit-learn, backtested at 10 forecast origins 28 days apart
(no overlapping windows) against a 28-day moving average. Measured, not
assumed:

- **On absolute error the model is worse.** Forecast value added is **−3.9%**
  on MAE, and the moving average is the better forecast for **58% of items**.
- **The loss is all in the slow sellers.** By demand class, the model adds
  **+5.0%** on the 152 smooth items and **+1.4%** on the 33 erratic ones, but
  loses **−10.5%** on the 463 intermittent items and **−9.2%** on the 171 lumpy
  ones, being worse on about two thirds of each.
- **On squared error the evidence is weak.** The model's RMSE is lower, 3.44
  against 3.51, but Diebold-Mariano across the 10 origins gives **p = 0.20**;
  the model confidence set at 10% keeps the model and drops the moving
  average only narrowly (p = 0.057).
- **Bias is item by item.** Overall bias is small (−0.6% of sales), but the
  model's mean error is significantly non-zero on 24% of items, against under
  1% for the moving average.
- **10% of items hold 74% of the squared error.**
- **The noise floor is small:** over 10 seeds the MAE sd is 0.0027, so a
  difference under 0.0087 between two runs is noise.
- **Leakage is small for this data:** plain K-fold is 3.3% optimistic against
  holding out the items (95% interval 0.8% to 5.7%); for random and group
  splits the interval includes zero. Leakage depends on the features; forelint measures it rather than
  assuming it.

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
- With a `cutoff` column, `dm_test` and the audit's bias test use one value
  per forecast origin, because the steps of one forecast share an origin and
  are correlated. When windows overlap (origins closer together than the
  horizon, such as statsforecast's default `step_size=1`), each series keeps
  only windows that do not overlap, counted back from its latest, and windows
  are lined up across series by that position, so series that end on
  different dates still count fully. At least 3 windows are needed. Backtest with
  `step_size` at least the horizon and ten or more windows. Without a cutoff
  column, every time stamp is treated as a separate one-step forecast.
- `dm_test` applies Diebold-Mariano with the Harvey-Leybourne-Newbold
  correction, falling back to Newey-West weights if the long-run variance
  comes out negative.
- `model_confidence_set` follows Hansen, Lunde and Nason (2011) with the
  T_max statistic and a circular block bootstrap. Its p-values match
  `arch.bootstrap.MCS` when given the same bootstrap draws.
- MASE and RMSSE are scaled by each item's in-sample naive error from its
  first positive value onward, as M5 does; RMSSE is the per-series part of
  M5's weighted WRMSSE. Gaps in the history are not counted as one-step
  changes.
- Demand classes use ADI from each item's first demand and CV² of the
  positive demand sizes, with the Syntetos-Boylan cut-offs 1.32 and 0.49.
- `split_optimism` compares each leaky split with its honest counterpart on the
  same test rows and the same number of training rows per group: `random`
  vs `time` (leak through time), `group` vs `group_time` (through other
  entities' future) and `kfold` vs `group` (through the entity itself, what
  plain K-fold does). Each optimism figure comes with a 95% interval over
  folds; a negative value means no detectable leak, not a pessimistic split.
- `noise_floor` reports the critical difference, t · √2 · sd: the smallest
  difference between two single runs that is significant.

## Cite

If forelint helps your work, please cite it:

> Mashiah, D. (2026). forelint: lint your forecast backtests before you trust a score. Zenodo. https://doi.org/10.5281/zenodo.24044927

## Status

Alpha. The API may change before 1.0. Issues and pull requests are welcome.

## Author

[David Mashiah](https://davidmashiah.com). MIT licensed.
