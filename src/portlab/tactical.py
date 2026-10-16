"""Tactical / timing models:
moving-average timing (price-vs-MA and MA crossover, multi-period weighted
signals), relative strength, dual momentum, target volatility, seasonal, and
CAPE-based valuation switching.

Every model returns a weights DataFrame (dates x assets). `evaluate` applies
weights with a one-period execution lag (no look-ahead) and transaction costs,
producing a strategy return series that plugs into portlab.metrics/plots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS


# ------------------------------------------------------------------ helpers

def _ma(prices: pd.DataFrame, window: int, kind: str = "sma") -> pd.DataFrame:
    if kind == "sma":
        return prices.rolling(window).mean()
    if kind == "ema":
        return prices.ewm(span=window, adjust=False).mean()
    raise ValueError("kind must be 'sma' or 'ema'")


def evaluate(weights: pd.DataFrame, returns: pd.DataFrame,
             tc_bps: float = 0.0, lag: int = 1) -> pd.Series:
    """Strategy returns from a weight schedule.

    Weights are shifted `lag` periods (signals trade at the next period's
    price — no look-ahead) and transaction costs are charged on turnover.
    """
    w = weights.shift(lag).reindex(returns.index).fillna(0.0)
    cols = [c for c in w.columns if c in returns.columns]
    strat = (w[cols] * returns[cols]).sum(axis=1)
    if tc_bps > 0:
        turnover = w[cols].diff().abs().sum(axis=1).fillna(0.0)
        strat = strat - turnover * tc_bps / 1e4
    strat.name = "strategy"
    return strat


def _route(signal: pd.DataFrame, base_weights: pd.Series,
           out_asset: str | None) -> pd.DataFrame:
    """Turn a boolean in/out signal per asset into weights; failed signals
    route to `out_asset` (cash if None)."""
    w = signal.astype(float).mul(base_weights, axis=1)
    if out_asset is not None:
        w[out_asset] = w.get(out_asset, 0.0) + (1.0 - w.sum(axis=1)).clip(lower=0.0)
    return w


# ------------------------------------------------------------------ models

def ma_timing(
    prices: pd.DataFrame,
    windows: dict[int, float] | int = 200,
    ma_kind: str = "sma",
    crossover_fast: int | None = None,
    weights: dict[str, float] | None = None,
    out_asset: str | None = None,
    signal_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Moving-average timing.

    windows: single window (price vs MA) or {window: weight} for
        multiple weighted timing periods (fraction invested = weighted share
        of bullish signals).
    crossover_fast: if set, signal is fast-MA > slow-MA instead of price > MA.
    signal_prices: optional separate signal asset prices; defaults to
        traded prices.
    """
    sp = signal_prices if signal_prices is not None else prices
    if isinstance(windows, int):
        windows = {windows: 1.0}
    total = sum(windows.values())
    frac = None
    for win, wgt in windows.items():
        slow = _ma(sp, win, ma_kind)
        bull = (_ma(sp, crossover_fast, ma_kind) > slow) if crossover_fast \
            else (sp > slow)
        part = bull.astype(float) * (wgt / total)
        frac = part if frac is None else frac + part

    base = pd.Series(weights) if weights else \
        pd.Series(1.0 / prices.shape[1], index=prices.columns)
    base = base / base.sum()
    w = frac.mul(base, axis=1)
    if out_asset is not None:
        w[out_asset] = w.get(out_asset, 0.0) + (1.0 - w.sum(axis=1))
    return w.dropna(how="all").fillna(0.0)


def relative_strength(
    prices: pd.DataFrame,
    lookbacks: dict[int, float] | int = 126,
    top_n: int = 1,
    out_asset: str | None = None,
    ma_risk_control: int | None = None,
) -> pd.DataFrame:
    """Hold the top-N assets by (weighted multi-period) trailing return.

    ma_risk_control: a selected asset below its own N-period MA is replaced
    by the out-of-market asset.
    """
    if isinstance(lookbacks, int):
        lookbacks = {lookbacks: 1.0}
    total = sum(lookbacks.values())
    score = None
    for lb, wgt in lookbacks.items():
        mom = prices.pct_change(lb) * (wgt / total)
        score = mom if score is None else score + mom
    ranks = score.rank(axis=1, ascending=False)
    selected = ranks <= top_n

    if ma_risk_control:
        above_ma = prices > _ma(prices, ma_risk_control)
        selected = selected & above_ma

    base = pd.Series(1.0 / top_n, index=prices.columns)
    w = _route(selected, base, out_asset)
    return w.dropna(how="all").fillna(0.0)


def dual_momentum(
    prices: pd.DataFrame,
    cash_prices: pd.Series,
    lookback: int = 252,
    top_n: int = 1,
    out_asset: str | None = None,
) -> pd.DataFrame:
    """Antonacci dual momentum: relative momentum picks the leaders, absolute
    momentum (excess return vs cash over the lookback) gates them into cash."""
    mom = prices.pct_change(lookback)
    cash_mom = cash_prices.pct_change(lookback)
    ranks = mom.rank(axis=1, ascending=False)
    selected = (ranks <= top_n) & mom.gt(cash_mom, axis=0)
    base = pd.Series(1.0 / top_n, index=prices.columns)
    w = _route(selected, base, out_asset)
    return w.dropna(how="all").fillna(0.0)


def target_volatility(
    returns: pd.DataFrame,
    weights: dict[str, float],
    target_annual: float = 0.10,
    window: int = 21,
    max_exposure: float = 1.0,
    out_asset: str | None = None,
    periods: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Scale portfolio exposure so realized vol matches the target."""
    base = pd.Series(weights)
    base = base / base.sum()
    port = (returns[list(base.index)] * base.values).sum(axis=1)
    realized = port.rolling(window).std() * np.sqrt(periods)
    exposure = (target_annual / realized).clip(upper=max_exposure)
    w = pd.DataFrame(np.outer(exposure, base.values),
                     index=returns.index, columns=base.index)
    if out_asset is not None:
        w[out_asset] = w.get(out_asset, 0.0) + (1.0 - exposure).clip(lower=0.0)
    return w.dropna(how="all").fillna(0.0)


def seasonal(
    index: pd.DatetimeIndex,
    assets: list[str],
    out_asset: str | None = None,
    in_months=(11, 12, 1, 2, 3, 4),
) -> pd.DataFrame:
    """'Sell in May': invested during in_months, out-of-market otherwise."""
    invested = pd.Series(index.month, index=index).isin(in_months)
    base = pd.Series(1.0 / len(assets), index=assets)
    sig = pd.DataFrame(np.tile(invested.values[:, None], len(assets)),
                       index=index, columns=assets)
    return _route(sig, base, out_asset)


def cape_valuation(
    cape: pd.Series,
    index: pd.DatetimeIndex,
    stock_asset: str,
    bond_asset: str,
    low_pct: float = 0.25,
    high_pct: float = 0.75,
    lookback_years: int = 30,
    freq_per_year: int = 12,
) -> pd.DataFrame:
    """Shiller-CAPE valuation switch: cheap market -> more stocks.

    Stock weight is 100% below the rolling low percentile, 40% above the high
    percentile, linear in between (percentiles computed on a trailing window
    so there is no full-sample look-ahead).
    """
    window = lookback_years * freq_per_year
    lo = cape.rolling(window, min_periods=window // 2).quantile(low_pct)
    hi = cape.rolling(window, min_periods=window // 2).quantile(high_pct)
    frac = 1.0 - 0.6 * ((cape - lo) / (hi - lo)).clip(0, 1)
    frac = frac.reindex(index, method="ffill").fillna(0.6)
    return pd.DataFrame({stock_asset: frac, bond_asset: 1.0 - frac}, index=index)


def adaptive_allocation(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    lookback: int = 126,
    top_n: int = 4,
    vol_window: int = 63,
    out_asset: str | None = None,
    rebalance_every: int = 21,
) -> pd.DataFrame:
    """Adaptive allocation: momentum selects the top-N assets, risk
    (inverse volatility) sets their weights. Re-evaluated every
    `rebalance_every` periods; weights held constant in between."""
    mom = prices.pct_change(lookback)
    # ISSUE-003: returns is typically one row shorter than prices (pct_change
    # drops the first row). Reindex vol onto the price calendar so positional
    # indexing below stays in bounds at the final row.
    vol = returns.rolling(vol_window).std().reindex(prices.index)
    rows, idx = [], []
    for i in range(max(lookback, vol_window), len(prices), rebalance_every):
        date = prices.index[i]
        m = mom.iloc[i].dropna()
        if m.empty:
            continue
        picks = m.nlargest(top_n).index
        iv = 1.0 / vol.iloc[i][picks].replace(0, np.nan).dropna()
        if iv.empty:
            continue
        w = iv / iv.sum()
        row = pd.Series(0.0, index=prices.columns)
        row[w.index] = w.values
        rows.append(row)
        idx.append(date)
    w = pd.DataFrame(rows, index=idx).reindex(prices.index).ffill().fillna(0.0)
    if out_asset is not None:
        w[out_asset] = (1.0 - w.sum(axis=1)).clip(lower=0.0)
    return w
