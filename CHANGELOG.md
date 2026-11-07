# Changelog

## 0.2.0 (2026-11-08)

Breaking changes against 0.1.0:

- `split_optimism`: `reference` removed; columns `vs_reference` replaced by
  `honest`, `optimism` and `optimism_se`; strategies are now matched pairs
  (`random`/`time`, `group`/`group_time`, `kfold`/`group`).
- `audit`: `ts_limit` removed and `AuditReport.ts_limit` replaced by
  `bias_alpha`; `n_biased_items` now counts items with significant bias, not
  a tracking signal beyond ±4.
- `NoiseFloor.mde` renamed `critical_difference` (alias kept).

Corrections, from two independent reviews before release:

- `dm_test` uses the `cutoff` column: one point per forecast origin, and when
  windows overlap only non-overlapping origins are used. 0.1.0 treated every
  step of a multi-step forecast as independent and rejected a true null far
  too often. Without a cutoff column it refuses duplicate (id, time) rows.
- `diebold_mariano` falls back to Newey-West weights when the long-run
  variance is negative, and rejects NaN with h > 1, inf and h < 1.
- `audit` flags biased items with a test of zero mean error (per
  non-overlapping forecast origin when there is a cutoff column, and not at
  all with fewer than 3), replacing the ±4 tracking-signal
  alarm, which flags most unbiased items over long backtests. The tracking
  signal is still reported, over each item's last 12 periods.
- `audit` scores every model on the same rows (rows with missing actuals or
  forecasts are dropped and counted), ignores unused categories in ids,
  checks the type of the `exclude` column, warns when ids do not match the
  training history or the segments, and gives no value added for an item
  whose baseline error is zero.
- MASE and RMSSE start each item's history at its first positive value, as
  M5 does, and skip gaps in time.
- `classify_demand` counts ADI from each item's first demand.
- `split_optimism` compares each leaky split with its honest counterpart on
  the same test rows and the same number of training rows per group, so it
  measures the leak and not the amount of data, and accepts only error
  scorers.
- Stockout rows flagged in `train` no longer drive the demand class; 0/1
  integer flags are accepted for `exclude`.
- `NoiseFloor.mde` is now `critical_difference` (`mde` still works) and
  validates its inputs.
- `model_confidence_set` is documented as a circular block bootstrap and no
  longer builds a reps × n × m array.
- Packaging needs hatchling 1.26 or later.

New:

- `classify_demand`: smooth / erratic / intermittent / lumpy classes from
  ADI and CV² (Syntetos, Boylan and Croston 2005).
- `audit(segments=...)`: the whole audit broken down by segment, including
  `segments="demand_class"`.
- `audit(train=..., season_length=...)`: MASE and RMSSE per item and overall.
- `audit(exclude=...)`: leave flagged rows, such as stockout periods, out of
  scoring.
- M5 benchmark at 10 non-overlapping origins: the model wins on smooth items
  and loses on intermittent and lumpy ones, and overall on absolute error.
- Example on statsforecast cross-validation output.

## 0.1.0 (2026-11-08)

First release: `GroupTimeSplit`, `split_optimism`, `noise_floor`,
`diebold_mariano`, `dm_test`, `model_confidence_set` and `audit`.
