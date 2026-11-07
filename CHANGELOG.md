# Changelog

## 0.2.0 (2026-11-08)

- `classify_demand`: smooth / erratic / intermittent / lumpy classes from
  ADI and CV² (Syntetos, Boylan and Croston 2005).
- `audit(segments=...)`: the whole audit broken down by segment, including
  `segments="demand_class"`.
- `audit(train=..., season_length=...)`: MASE and RMSSE per item and overall.
- `audit(exclude=...)`: leave flagged rows, such as stockout periods, out of
  scoring.
- M5 benchmark shows the model winning on smooth items and losing on
  intermittent and lumpy ones.
- Example on statsforecast cross-validation output.

## 0.1.0 (2026-11-08)

First release: `GroupTimeSplit`, `split_optimism`, `noise_floor`,
`diebold_mariano`, `dm_test`, `model_confidence_set` and `audit`.
