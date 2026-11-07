"""Audit statsforecast cross-validation output with forelint.

    pip install forelint statsforecast
    python examples/statsforecast_cv.py

statsforecast's `cross_validation` returns one row per series, date and
cutoff, with the actual in `y` and one column per model. forelint reads that
table as it is.
"""
import numpy as np

import forelint as fl
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, CrostonOptimized, Naive, SeasonalNaive
from statsforecast.utils import generate_series

# 60 daily series with weekly seasonality, some of them sparse.
df = generate_series(n_series=60, freq="D", min_length=240, max_length=240,
                     equal_ends=True, seed=0)
rng = np.random.default_rng(0)
df["y"] = (df["y"] + rng.normal(0, 3, len(df))).round().clip(lower=0)  # real demand is noisy
sparse = (df["unique_id"].astype(int) % 3 == 0) & (df.groupby("unique_id").cumcount() % 4 != 0)
df.loc[sparse, "y"] = 0  # a third of the series sell only one day in four

sf = StatsForecast(
    models=[AutoETS(season_length=7), CrostonOptimized(), SeasonalNaive(season_length=7), Naive()],
    freq="D",
)
h = 14
cv = sf.cross_validation(df=df, h=h, n_windows=3, step_size=h)
history = df[df["ds"] < cv["ds"].min()]

report = fl.audit(cv, ["AutoETS", "CrostonOptimized", "SeasonalNaive"], baseline="Naive",
                  train=history, season_length=7, segments="demand_class")
print(report)

print("\nIs AutoETS really better than the seasonal naive?")
print(fl.dm_test(cv, "AutoETS", "SeasonalNaive"))

losses = (cv.assign(**{m: (cv[m] - cv["y"]).abs() for m in ["AutoETS", "CrostonOptimized", "SeasonalNaive", "Naive"]})
            .groupby("ds")[["AutoETS", "CrostonOptimized", "SeasonalNaive", "Naive"]].mean())
print("\nModels that cannot be told apart from the best (10%):")
print(fl.model_confidence_set(losses, size=0.10, random_state=0))
