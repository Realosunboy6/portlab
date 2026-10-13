"""Return computation and cleaning utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS


def simple_returns(prices: pd.DataFrame | pd.Series,
                   periods_lag: int = 1) -> pd.DataFrame | pd.Series:
    """Simple returns; periods_lag > 1 gives N-period (overlapping) returns."""
    if periods_lag < 1:
        raise ValueError("periods_lag must be >= 1")
    return prices.pct_change(periods_lag).iloc[periods_lag:]


def log_returns(prices: pd.DataFrame | pd.Series,
                periods_lag: int = 1) -> pd.DataFrame | pd.Series:
    if periods_lag < 1:
        raise ValueError("periods_lag must be >= 1")
    return np.log(prices / prices.shift(periods_lag)).iloc[periods_lag:]


def clean_returns(
    rets: pd.DataFrame,
    max_missing: float = 0.10,
    clip_abs: float | None = 1.0,
) -> pd.DataFrame:
    """Drop sparse columns, forward-fill small gaps, optionally clip outliers.

    max_missing: drop tickers missing more than this fraction of observations.
    clip_abs: winsorize single-period returns beyond +/- this value (bad ticks);
        pass None to disable.
    """
    keep = rets.columns[rets.isna().mean() <= max_missing]
    out = rets[keep].ffill(limit=5).fillna(0.0)
    if clip_abs is not None:
        out = out.clip(-clip_abs, clip_abs)
    return out


def annualize_return(rets: pd.Series | pd.DataFrame, periods: int = TRADING_DAYS):
    """Geometric annualized return from simple per-period returns."""
    n = len(rets)
    if n == 0:
        raise ValueError("empty return series")
    growth = (1 + rets).prod()
    return growth ** (periods / n) - 1


def annualize_vol(rets: pd.Series | pd.DataFrame, periods: int = TRADING_DAYS):
    return rets.std(ddof=1) * np.sqrt(periods)


def to_monthly(daily_rets: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Compound daily simple returns into calendar-month returns."""
    return (1 + daily_rets).resample("ME").prod() - 1


def growth_of(rets: pd.Series | pd.DataFrame, initial: float = 1.0,
              geometric: bool = True):
    """Cumulative growth of an initial investment, same index as rets.

    geometric=False uses additive (cumsum) wealth.
    """
    if geometric:
        return initial * (1 + rets).cumprod()
    return initial * (1 + rets.cumsum())
