"""Risk-free rate helpers.

PV computes Sharpe/Sortino against the U.S. 3-Month Treasury Bill rate.
:func:`get_tbill` fetches that series (FRED DGS3MO) so portlab can match
the convention exactly instead of assuming rf=0.
"""

from __future__ import annotations

import pandas as pd


def get_tbill(start: str, end: str | None = None) -> pd.Series:
    """Daily 3-month T-bill rate (annualized decimal) from FRED.

    Forward-filled over weekends/holidays. Raises on download failure —
    callers that need a hard guarantee should catch and fall back to 0.
    """
    from pandas_datareader import data as pdr

    s = pdr.DataReader("DGS3MO", "fred", start, end)["DGS3MO"]
    s = s.astype(float) / 100.0
    return s.ffill().bfill().rename("TBILL_3M")


def period_rf(tbill: pd.Series, index: pd.DatetimeIndex, periods: int) -> pd.Series:
    """Resample a daily annualized T-bill series to per-period rates.

    The rate in effect at each period end, divided into a per-period rate.
    """
    per = tbill.resample("ME" if periods == 12 else "D").last()
    per = per.reindex(index, method="ffill").fillna(0.0)
    return (per / periods).rename("rf")
