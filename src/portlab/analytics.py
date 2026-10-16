"""Asset analytics: correlations (static + rolling), autocorrelation,
cointegration, and side-by-side performance comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from .config import DEFAULT_RF, TRADING_DAYS
from . import metrics as M


def correlation_matrix(rets: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    return rets.corr(method=method)


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 63) -> pd.Series:
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    return df.iloc[:, 0].rolling(window).corr(df.iloc[:, 1])


def autocorrelation(rets: pd.Series, lags: int = 12) -> pd.Series:
    """Autocorrelation of returns at lags 1..lags (with 95% CI in attrs)."""
    vals = [rets.autocorr(lag=k) for k in range(1, lags + 1)]
    out = pd.Series(vals, index=pd.Index(range(1, lags + 1), name="lag"))
    out.attrs["ci95"] = 1.96 / np.sqrt(len(rets))
    return out


def cointegration_test(price_a: pd.Series, price_b: pd.Series) -> pd.Series:
    """Engle-Granger cointegration between two (log) price series."""
    df = pd.concat([price_a, price_b], axis=1, join="inner").dropna()
    la, lb = np.log(df.iloc[:, 0]), np.log(df.iloc[:, 1])
    t_stat, p_value, crit = coint(la, lb)
    hedge = float(np.polyfit(lb, la, 1)[0])
    spread = la - hedge * lb
    adf_p = adfuller(spread)[1]
    return pd.Series({
        "EG t-stat": float(t_stat),
        "EG p-value": float(p_value),
        "crit 5%": float(crit[1]),
        "hedge ratio": hedge,
        "spread ADF p-value": float(adf_p),
        "cointegrated (5%)": bool(p_value < 0.05),
    })


def performance_table(rets: pd.DataFrame, rf: float = DEFAULT_RF,
                      periods: int = TRADING_DAYS,
                      bench: pd.Series | None = None) -> pd.DataFrame:
    """Side-by-side comparison of several assets/funds."""
    return pd.concat(
        [M.summary(rets[c].dropna(), rf=rf, periods=periods, bench=bench, name=c)
         for c in rets.columns], axis=1)


def pca(rets: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
    """Principal component analysis of asset returns (correlation-based).

    Returns loadings (assets x components). attrs carry explained variance
    ratios and the principal-portfolio weight matrix (eigenvectors scaled to
    sum to 1 in absolute terms).
    """
    corr = rets.corr().values
    vals, vecs = np.linalg.eigh(corr)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    k = min(n_components, len(vals))
    cols = [f"PC{i+1}" for i in range(k)]
    loadings = pd.DataFrame(vecs[:, :k], index=rets.columns, columns=cols)
    explained = pd.Series(vals[:k] / vals.sum(), index=cols)
    pp = loadings / loadings.abs().sum(axis=0)
    loadings.attrs["explained_variance"] = explained
    loadings.attrs["principal_portfolios"] = pp
    return loadings


def screener(
    rets: pd.DataFrame,
    filters: dict | None = None,
    sort_by: str = "Sharpe Ratio",
    ascending: bool = False,
    rf: float = DEFAULT_RF,
    periods: int = TRADING_DAYS,
    top: int | None = None,
) -> pd.DataFrame:
    """Return-based fund/stock screener over any return panel.

    filters: {metric_name: (min, max)} using metric names from
    portlab.metrics.summary, e.g. {"CAGR": (0.05, None),
    "Max Drawdown": (-0.35, None)}. None disables a bound.
    """
    table = performance_table(rets, rf=rf, periods=periods).T
    for metric, (lo, hi) in (filters or {}).items():
        if metric not in table.columns:
            raise ValueError(f"unknown metric {metric!r}; "
                             f"available: {list(table.columns)}")
        if lo is not None:
            table = table[table[metric] >= lo]
        if hi is not None:
            table = table[table[metric] <= hi]
    table = table.sort_values(sort_by, ascending=ascending)
    return table.head(top) if top else table


def income_history(tickers: list[str], start: str = "2015-01-01",
                   holdings: dict[str, float] | None = None) -> pd.DataFrame:
    """Annual dividend income per ticker from yfinance dividend records.

    holdings: optional {ticker: dollar value today} — adds an estimated
    annual income column using the trailing-12-month yield.
    """
    import yfinance as yf
    from .data.prices import get_prices

    prices = get_prices(tickers, start)
    rows = {}
    for t in tickers:
        try:
            div = yf.Ticker(t).dividends
            div.index = div.index.tz_localize(None)
            div = div.loc[div.index >= pd.Timestamp(start)]
        except Exception:
            continue
        if div.empty:
            continue
        annual = div.groupby(div.index.year).sum()
        last_price = float(prices[t].dropna().iloc[-1])
        ttm = float(div.iloc[-4:].sum()) if len(div) >= 4 else float(annual.iloc[-1])
        row = {f"{y}": float(v) for y, v in annual.items()}
        row["TTM yield"] = ttm / last_price
        if holdings and t in holdings:
            row["Est. annual income"] = holdings[t] * ttm / last_price
        rows[t] = row
    return pd.DataFrame(rows).T
