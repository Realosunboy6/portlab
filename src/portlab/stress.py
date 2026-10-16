"""Historical and hypothetical stress testing.

A stress scenario is a vector of asset shocks (simple returns) applied to
current positions::

    shocked_value_i = qty_i * price_i * (1 + shock_i)
    scenario_pnl    = sum_i qty_i * price_i * shock_i

Historical scenarios are extracted from real price history: the shock for
each asset is its cumulative simple return over the window. Canned crisis
windows cover the modern era's big drawdowns; any custom window or explicit
shock vector works too.

Reverse stress testing answers "what breaks us": :func:`worst_window` finds
the worst historical ``window_days``-long stretch for a given portfolio, and
:func:`uniform_shock_for_loss` gives the flat market shock that would inflict
a target loss.
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "CRISES",
    "scenario_shocks",
    "apply_scenario",
    "stress_report",
    "worst_window",
    "uniform_shock_for_loss",
]

#: Canned crisis windows as (start, end) peak-to-trough dates for the S&P 500.
#: Shocks are always extracted per-asset from the caller's own price history,
#: so these windows work for any asset universe with data that far back.
CRISES = {
    "GFC 2008": ("2007-10-09", "2009-03-09"),
    "COVID crash 2020": ("2020-02-19", "2020-03-23"),
    "Rate shock 2022": ("2022-01-03", "2022-10-12"),
}


def _as_wide(prices) -> pd.DataFrame:
    if isinstance(prices, pd.DataFrame):
        wide = prices.copy()
    else:  # long format with date/symbol/price columns
        df = pd.read_csv(prices) if isinstance(prices, str) else prices.copy()
        df["date"] = pd.to_datetime(df["date"])
        wide = df.pivot_table(index="date", columns="symbol",
                              values="price", aggfunc="last")
    wide.index = pd.to_datetime(wide.index)
    if wide.index.tz is not None:
        # yfinance returns tz-aware indexes; comparisons against tz-naive
        # bounds would raise, so normalize to tz-naive wall time.
        wide.index = wide.index.tz_localize(None)
    return wide.sort_index()


def scenario_shocks(prices, start, end) -> pd.Series:
    """Per-asset cumulative simple return over ``[start, end]``.

    shock_i = price_i(end) / price_i(start) - 1, using the last available
    price on or before each bound (trading-calendar safe). Assets missing
    either bound are dropped.
    """
    wide = _as_wide(prices)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end <= start:
        raise ValueError(f"end ({end.date()}) must be after start ({start.date()})")
    px0 = wide[wide.index <= start].iloc[-1] if (wide.index <= start).any() else None
    px1 = wide[wide.index <= end].iloc[-1] if (wide.index <= end).any() else None
    if px0 is None or px1 is None:
        raise ValueError("price history does not cover the scenario window")
    shocks = px1 / px0 - 1.0
    return shocks.dropna().rename("shock")


def apply_scenario(holdings: dict, prices_last: dict, shocks) -> pd.DataFrame:
    """Apply shocks to positions; returns per-symbol P&L attribution.

    ``holdings``: symbol -> shares. ``prices_last``: symbol -> current price.
    ``shocks``: symbol -> simple-return shock (Series or dict).

    Columns: shares, price, market_value, shock, pnl, pnl_pct_of_portfolio.
    Symbols without a shock are assumed unshocked (0.0).
    """
    if isinstance(shocks, pd.Series):
        shocks = shocks.to_dict()
    rows = []
    total_mv = 0.0
    for sym, qty in holdings.items():
        sym = str(sym).strip().upper()
        px = float(prices_last[sym])
        mv = float(qty) * px
        total_mv += mv
        rows.append({"symbol": sym, "shares": float(qty), "price": px,
                     "market_value": mv,
                     "shock": float(shocks.get(sym, 0.0))})
    df = pd.DataFrame(rows)
    df["pnl"] = df["market_value"] * df["shock"]
    df["pnl_pct_of_portfolio"] = df["pnl"] / total_mv if total_mv else 0.0
    return df.set_index("symbol")


def stress_report(holdings: dict, prices_last: dict, scenarios: dict) -> dict:
    """Run several named scenarios; returns {name: {pnl, return, table}}.

    ``scenarios``: name -> shock Series/dict (e.g. from :func:`scenario_shocks`
    or :data:`CRISES` windows).
    """
    out = {}
    for name, shocks in scenarios.items():
        table = apply_scenario(holdings, prices_last, shocks)
        port_mv = float(table["market_value"].sum())
        pnl = float(table["pnl"].sum())
        out[name] = {"pnl": pnl,
                     "portfolio_return": pnl / port_mv if port_mv else float("nan"),
                     "portfolio_value": port_mv,
                     "shocked_value": port_mv + pnl,
                     "table": table}
    return out


def worst_window(prices, weights: dict, window_days: int = 60) -> dict:
    """Find the worst ``window_days``-trading-day stretch in history.

    Builds the portfolio's daily return series from ``weights`` (symbol ->
    weight, need not sum to 1 — it is the *shape* that matters), then scans
    every ``window_days``-long window for the minimum cumulative simple
    return. Returns start/end dates, the window return, and the per-asset
    shocks over that window (usable directly as a scenario).
    """
    wide = _as_wide(prices).pct_change().dropna()
    w = pd.Series({str(k).strip().upper(): float(v)
                   for k, v in weights.items()})
    w = w[w.index.isin(wide.columns)]
    if w.empty:
        raise ValueError("no weight symbols found in price history")
    port_ret = (wide[w.index] * (w / w.sum()).to_numpy()).sum(axis=1)
    if len(port_ret) < window_days:
        raise ValueError(f"only {len(port_ret)} returns, need {window_days}")
    # cumulative simple return over each window: prod(1+r) - 1
    cum = (1.0 + port_ret).rolling(window_days).apply(lambda x: x.prod() - 1.0,
                                                     raw=True)
    end = cum.idxmin()
    loc = port_ret.index.get_loc(end)
    start = port_ret.index[loc - window_days + 1]
    # shocks are price ratios: the window's first return runs from the
    # previous trading day's close, so anchor the shock window there.
    wide_px = _as_wide(prices)
    px_start = wide_px.index[wide_px.index.get_loc(start) - 1]
    shocks = scenario_shocks(wide_px, px_start.date().isoformat(),
                             end.date().isoformat())
    return {"start": start.date().isoformat(), "end": end.date().isoformat(),
            "window_return": float(cum.min()), "window_days": window_days,
            "shocks": shocks}


def uniform_shock_for_loss(target_loss: float) -> float:
    """Flat per-asset shock that inflicts ``target_loss`` (e.g. -0.20).

    Under a uniform shock every asset falls by the same fraction, so the
    portfolio loses exactly that fraction regardless of weights. Trivial
    arithmetically — but it is the honest answer to "how bad a day breaks us",
    and the starting point before running non-uniform scenarios.
    """
    if not -1.0 <= target_loss < 0:
        raise ValueError("target_loss must be in [-1, 0)")
    return float(target_loss)
