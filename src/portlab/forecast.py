"""Time-series forecasting: ARIMA and ETS (Holt-Winters exponential
smoothing), each returning the point forecast with confidence intervals.

Conventions follow statsmodels: `order=(p, d, q)` for ARIMA; ETS takes
trend/seasonal components ('add', 'mul' or None) with a seasonal period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def arima_forecast(series: pd.Series, order: tuple[int, int, int] = (5, 1, 0),
                   steps: int = 30, alpha: float = 0.05) -> pd.DataFrame:
    """ARIMA forecast with (1-alpha) confidence intervals."""
    from statsmodels.tsa.arima.model import ARIMA
    y = series.dropna().astype(float)
    if len(y) < max(order[0], order[2]) + 10:
        raise ValueError("not enough observations for ARIMA{0}".format(order))
    fit = ARIMA(y, order=order).fit()
    fc = fit.get_forecast(steps=steps)
    idx = _future_index(y.index, steps)
    out = pd.DataFrame({"forecast": fc.predicted_mean.values}, index=idx)
    ci = fc.conf_int(alpha=alpha).values
    out["lower"] = ci[:, 0]
    out["upper"] = ci[:, 1]
    out.attrs["aic"] = float(fit.aic)
    return out


def ets_forecast(series: pd.Series, steps: int = 30, trend: str | None = "add",
                 seasonal: str | None = None, seasonal_periods: int | None = None,
                 alpha: float = 0.05) -> pd.DataFrame:
    """Holt-Winters ETS forecast.

    Intervals are the residual-based approximation forecast ± z·σ·√h
    (this statsmodels version exposes no analytic ETS prediction SEs).
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    y = series.dropna().astype(float)
    if len(y) < 2 * (seasonal_periods or 2) + 10:
        raise ValueError("not enough observations for ETS")
    fit = ExponentialSmoothing(y, trend=trend, seasonal=seasonal,
                               seasonal_periods=seasonal_periods).fit()
    idx = _future_index(y.index, steps)
    fc = np.asarray(fit.forecast(steps), dtype=float)
    sigma = float(np.std(fit.resid, ddof=1))
    z = float(norm.ppf(1 - alpha / 2))
    h = np.arange(1, steps + 1)
    out = pd.DataFrame({"forecast": fc,
                        "lower": fc - z * sigma * np.sqrt(h),
                        "upper": fc + z * sigma * np.sqrt(h)}, index=idx)
    out.attrs["aic"] = float(fit.aic)
    return out


def _future_index(index: pd.DatetimeIndex, steps: int) -> pd.DatetimeIndex:
    """Extend a regular datetime index by `steps` periods."""
    if len(index) < 2:
        raise ValueError("need at least 2 observations to infer frequency")
    step = index[-1] - index[-2]
    return pd.DatetimeIndex([index[-1] + (i + 1) * step for i in range(steps)],
                           name=index.name)
