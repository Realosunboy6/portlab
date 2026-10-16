import numpy as np
import pandas as pd

from portlab import forecast as F


def _series(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = pd.Series(100 + np.cumsum(rng.normal(0.05, 1.0, n)),
                  index=pd.date_range("2023-01-01", periods=n, freq="B"))
    return y


def test_arima_shape_and_intervals():
    y = _series()
    fc = F.arima_forecast(y, order=(5, 1, 0), steps=30)
    assert len(fc) == 30
    assert ((fc["lower"] <= fc["forecast"]) & (fc["forecast"] <= fc["upper"])).all()
    assert np.isfinite(fc.attrs["aic"])
    assert fc.index[0] > y.index[-1]


def test_ets_shape_and_intervals():
    y = _series()
    fc = F.ets_forecast(y, steps=20, trend="add")
    assert len(fc) == 20
    assert ((fc["lower"] <= fc["forecast"]) & (fc["forecast"] <= fc["upper"])).all()


def test_arima_tracks_trend():
    y = _series()
    fc = F.arima_forecast(y, order=(5, 1, 0), steps=10)
    # upward drift: 10-step forecast should sit above the last observation
    assert fc["forecast"].iloc[-1] > y.iloc[-1] - 5.0
