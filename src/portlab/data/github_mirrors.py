"""Long-history market data from freely licensed GitHub dataset mirrors
(the Datahub `datasets` org). Useful as a no-key fallback when Yahoo, FRED,
or the Ken French site are unreachable or rate-limited — GitHub raw content
is almost always accessible.

Series (all monthly):
  - Shiller S&P 500: price, dividends, earnings, CPI, long rate (1871+)
  - US 10-year Treasury constant-maturity yield (1953+)
  - Gold price (1833+)
"""

from __future__ import annotations

import io
import urllib.request

import numpy as np
import pandas as pd

from . import cache

MIRRORS = {
    "shiller": "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv",
    "us10y":   "https://raw.githubusercontent.com/datasets/bond-yields-us-10y/main/data/monthly.csv",
    "gold":    "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv",
}


def _fetch_csv(name: str, refresh: bool = False) -> pd.DataFrame:
    key = cache.cache_key("mirror", name)
    if not refresh:
        hit = cache.load(key)
        if hit is not None:
            return hit
    with urllib.request.urlopen(MIRRORS[name], timeout=60) as r:
        df = pd.read_csv(io.BytesIO(r.read()))
    cache.save(key, df)
    return df


def shiller_sp500(refresh: bool = False) -> pd.DataFrame:
    """Raw Shiller dataset: SP500, Dividend, Earnings, CPI, Long Interest
    Rate, Real Price/Dividend/Earnings, PE10 — monthly since 1871."""
    df = _fetch_csv("shiller", refresh)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")


def sp500_total_returns(refresh: bool = False) -> pd.Series:
    """Monthly S&P 500 total returns (price + dividends), 1871+."""
    sp = shiller_sp500(refresh)
    tr = (sp["SP500"] + sp["Dividend"] / 12) / sp["SP500"].shift(1) - 1
    return tr.dropna().rename("US Stocks (S&P 500 TR)")


def cpi_series(refresh: bool = False) -> pd.Series:
    """Monthly CPI index from the Shiller dataset (fallback for FRED CPIAUCSL).
    Zero placeholder rows in recent months are dropped."""
    cpi = shiller_sp500(refresh)["Consumer Price Index"]
    return cpi.replace(0, np.nan).dropna().rename("CPI")


def shiller_cape(refresh: bool = False) -> pd.Series:
    """Monthly CAPE / PE10 (fallback for the Yale xls download)."""
    sp = shiller_sp500(refresh)
    col = next((c for c in sp.columns if c.upper() in ("PE10", "CAPE")), None)
    if col is None:
        raise KeyError("PE10 column not found in Shiller mirror")
    s = pd.to_numeric(sp[col], errors="coerce").dropna()
    return s[s > 0].rename("CAPE")


def us10y_yield(refresh: bool = False) -> pd.Series:
    """10-year constant-maturity yield in percent, monthly since 1953."""
    df = _fetch_csv("us10y", refresh)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")["Rate"].rename("US10Y")


def treasury10y_returns(refresh: bool = False) -> pd.Series:
    """Synthetic 10-year Treasury monthly total returns from real yields."""
    from .asset_classes import synthetic_treasury_returns
    return synthetic_treasury_returns(us10y_yield(refresh),
                                      duration=8.5).rename("10Y Treasury")


def gold_returns(refresh: bool = False) -> pd.Series:
    """Monthly gold returns from the price fix series, 1833+."""
    df = _fetch_csv("gold", refresh)
    idx = pd.to_datetime(df["Date"], format="%Y-%m", errors="coerce")
    px = pd.Series(df["Price"].values, index=idx).dropna()
    return px.pct_change().dropna().rename("Gold")


def get_mirror_returns(start: str = "1972-01-01",
                       refresh: bool = False) -> pd.DataFrame:
    """Monthly returns for stocks / 10Y Treasuries / gold from the mirrors —
    a complete asset-class panel with zero API dependencies."""
    out = pd.DataFrame({
        s.name: s for s in (sp500_total_returns(refresh),
                            treasury10y_returns(refresh),
                            gold_returns(refresh))
    }).dropna()
    return out.loc[out.index >= pd.Timestamp(start)]
