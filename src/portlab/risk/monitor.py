"""RiskLens daily watch: from fills to a risk report.

Pipeline::

    fills (broker reality) + price history (Yahoo)
        -> portfolio_returns_from_fills()   # value + time-weighted returns
        -> daily_risk_report()              # VaR, exceptions, backtests, tails
        -> write_markdown_report()          # human-readable output

Read-only: nothing here trades, edits the ledger, or touches cost basis.
Cost basis stays broker-side only; prices are used for *valuation* only.
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

from . import var as _var
from . import backtest as _bt
from . import tail as _tail
from . import rebalance as _rb

__all__ = [
    "portfolio_returns_from_fills",
    "daily_risk_report",
    "write_markdown_report",
]

#: Minimum observations before a rolling VaR is quoted (avoids degenerate
#: quantiles on a handful of points).
MIN_VAR_OBS = 20


def _read_fills(fills) -> pd.DataFrame:
    """Load fills from a DataFrame, a CSV file, or a directory of CSVs."""
    if isinstance(fills, pd.DataFrame):
        df = fills.copy()
    else:
        if os.path.isdir(fills):
            paths = sorted(os.path.join(fills, f) for f in os.listdir(fills)
                           if f.endswith(".csv"))
            if not paths:
                raise ValueError(f"no CSV files in {fills}")
            df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        else:
            df = pd.read_csv(fills)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["side"] = df["side"].astype(str).str.strip().str.upper()
    df["qty"] = pd.to_numeric(df["qty"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)
    bad = set(df["side"]) - {"BUY", "SELL"}
    if bad:
        raise ValueError(f"unknown fill sides: {bad}")
    return df


def _read_prices(prices) -> pd.DataFrame:
    """Load prices as a wide DataFrame (dates x symbols).

    Accepts long format (DataFrame or CSV path with columns
    date,symbol,price) or an already-wide DataFrame with a datetime index.
    """
    if isinstance(prices, pd.DataFrame) and "date" not in prices.columns:
        wide = prices.copy()
        wide.index = pd.to_datetime(wide.index)
        return wide.sort_index()
    df = pd.read_csv(prices) if not isinstance(prices, pd.DataFrame) else prices.copy()
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["price"] = pd.to_numeric(df["price"], errors="raise")
    wide = df.pivot_table(index="date", columns="symbol", values="price",
                          aggfunc="last").sort_index()
    return wide


def portfolio_returns_from_fills(fills, prices) -> dict:
    """Rebuild the portfolio from fills and price history.

    ``fills``: DataFrame / CSV file / directory of CSVs with columns
    date,symbol,side,qty,price,fee,order_id (the fill *price* column is
    ignored here — valuation uses the ``prices`` history).
    ``prices``: DataFrame or CSV path, long format date,symbol,price.

    Returns dict with:
    - ``value``: daily portfolio market-value Series (holdings_t x prices_t)
    - ``returns``: daily *time-weighted* return Series. Computed as the
      previous-day-weighted average of symbol returns, so trades (external
      flows) do not create spurious return spikes.
    - ``holdings``: latest symbol -> qty dict (for the drift table)
    - ``prices_last``: latest symbol -> price dict
    - ``dates``: the price-history index used
    """
    fdf = _read_fills(fills)
    px = _read_prices(prices)

    signed = fdf["qty"].where(fdf["side"] == "BUY", -fdf["qty"])
    fdf = fdf.assign(signed_qty=signed)
    # Cumulative holdings per symbol as of each fill date, then forward-fill
    # onto the full price calendar (holdings apply from the fill date onward).
    cum = (fdf.pivot_table(index="date", columns="symbol", values="signed_qty",
                           aggfunc="sum").cumsum())
    holdings = cum.reindex(px.index).ffill().fillna(0.0)
    holdings = holdings.reindex(columns=px.columns, fill_value=0.0)

    value = (holdings * px).sum(axis=1)
    # Time-weighted returns: weights from t-1, symbol returns to t.
    sym_ret = px.pct_change()
    w_prev = holdings.shift(1) * px.shift(1)
    row_sum = w_prev.sum(axis=1)
    w_prev = w_prev.div(row_sum.replace(0, np.nan), axis=0)
    port_ret = (w_prev * sym_ret).sum(axis=1)
    port_ret[row_sum == 0] = np.nan  # no capital invested yet

    last = holdings.iloc[-1]
    return {
        "value": value.rename("portfolio_value"),
        "returns": port_ret.rename("daily_return"),
        "holdings": {s: float(last[s]) for s in last.index if last[s] != 0},
        "prices_last": {s: float(px[s].iloc[-1]) for s in px.columns},
        "dates": px.index,
    }


def _rolling_var(returns: pd.Series, alpha: float, window: int) -> pd.Series:
    """Rolling historical VaR with expanding warmup.

    Uses up to ``window`` past observations, but quotes a value once
    ``MIN_VAR_OBS`` have accumulated (expanding warmup, then rolling).
    No look-ahead: VaR at t uses returns strictly before t.
    """
    r = pd.Series(returns, dtype=float)
    out = pd.Series(np.nan, index=r.index, dtype=float)
    vals = r.to_numpy()
    for i in range(len(vals)):
        lo = max(0, i - window)
        hist = vals[lo:i]              # strictly before t=i, at most `window` obs
        hist = hist[~np.isnan(hist)]   # warmup NaNs must not poison the quantile
        if len(hist) >= MIN_VAR_OBS:
            # VaR at confidence (1-alpha): loss exceeded with prob alpha,
            # i.e. -(alpha-quantile of returns), a positive number.
            out.iloc[i] = float(-np.quantile(hist, alpha))
    return out


def daily_risk_report(value_series: pd.Series, alpha: float = 0.05,
                      var_window: int = 250,
                      returns: pd.Series | None = None,
                      target_weights: dict[str, float] | None = None,
                      holdings: dict[str, float] | None = None,
                      last_prices: dict[str, float] | None = None,
                      drift_threshold: float = _rb.DEFAULT_ABS_THRESHOLD,
                      ) -> tuple[pd.DataFrame, dict]:
    """Build the daily risk report.

    ``value_series``: daily portfolio market values (datetime index).
    ``returns``: optional daily return Series; defaults to ``value.pct_change()``.
        Pass the time-weighted series from ``portfolio_returns_from_fills``
        when fills exist, so trades don't fake returns.
    ``alpha``: tail probability for the 95% column (var_95); the 99% column
        always uses alpha=0.01.
    ``target_weights`` / ``holdings`` / ``last_prices``: when all three are
        given, a REBALANCE section (drift table, signal, trades) is added.

    Returns (report_df, summary):
    - report_df columns: date, portfolio_value, daily_return, var_95, var_99,
      exception_95, exception_99, cum_exceptions_99, traffic_light.
    - summary dict: n_obs, current_var_95/99, current_es_95, tail (from
      ``tail.tail_summary``), backtests (kupiec/christoffersen/joint at both
      levels when n>=250, else a note), basel_99 (traffic light on full
      sample), and optionally rebalance.
    """
    v = pd.Series(value_series, dtype=float).sort_index()
    r = (pd.Series(returns, dtype=float).reindex(v.index) if returns is not None
         else v.pct_change())

    var95 = _rolling_var(r, alpha, var_window)
    var99 = _rolling_var(r, 0.01, var_window)
    # exceptions() marks unscored days (warmup, NaN gaps) as <NA>; the
    # backtests consume that series directly (Kupiec counts scored days
    # only, Christoffersen skips pairs straddling a hole). The report
    # table gets a False-filled copy purely for display alignment.
    exc95 = _bt.exceptions(r, var95)
    exc99 = _bt.exceptions(r, var99)
    exc95_full = exc95.fillna(False)
    exc99_full = exc99.fillna(False)

    # Basel light on the trailing 250 observations of 99% exceptions.
    exc99_int = exc99_full.astype(int)
    trail = exc99_int.rolling(250, min_periods=1).sum()
    light = trail.map(lambda c: _bt.basel_traffic_light(int(c))["zone"])

    report_df = pd.DataFrame({
        "date": v.index,
        "portfolio_value": v.to_numpy(),
        "daily_return": r.to_numpy(),
        "var_95": var95.to_numpy(),
        "var_99": var99.to_numpy(),
        "exception_95": exc95_full.to_numpy(),
        "exception_99": exc99_full.to_numpy(),
        "cum_exceptions_99": exc99_int.cumsum().to_numpy(),
        "traffic_light": light.to_numpy(),
    })

    r_clean = r.dropna()
    n = len(r_clean)
    tail = _tail.tail_summary(r_clean)
    # Total return from the first *funded* day — the value series starts at
    # $0 before the first fill, and dividing by that gives inf, not a return.
    vnz = v[v != 0].dropna()
    # Current risk numbers on the full available history.
    cur_var95 = float(_var.historical_var(r_clean, alpha).iloc[-1]) \
        if n >= MIN_VAR_OBS else float("nan")
    cur_var99 = float(_var.historical_var(r_clean, 0.01).iloc[-1]) \
        if n >= MIN_VAR_OBS else float("nan")
    cur_es95 = float(_var.expected_shortfall(r_clean, alpha).iloc[-1]) \
        if n >= MIN_VAR_OBS else float("nan")

    if n >= _bt.MIN_BACKTEST_OBS:
        backtests = {
            "kupiec_95": _bt.kupiec_pof(exc95, alpha),
            "kupiec_99": _bt.kupiec_pof(exc99, 0.01),
            "christoffersen_99": _bt.christoffersen_ind(exc99),
            "joint_99": _bt.joint_test(exc99, 0.01),
        }
    else:
        backtests = {
            "note": (f"n={n} < {_bt.MIN_BACKTEST_OBS}: formal backtests "
                     "withheld (asymptotic tests need 250+ observations). "
                     "Exception counts and the traffic light are shown anyway."),
            "kupiec_95": None, "kupiec_99": None,
            "christoffersen_99": None, "joint_99": None,
        }

    summary: dict = {
        "n_obs": n,
        "start": str(v.index[0].date()) if len(v) else None,
        "end": str(v.index[-1].date()) if len(v) else None,
        "current_value": float(v.iloc[-1]) if len(v) else float("nan"),
        "current_var_95": cur_var95,
        "current_var_99": cur_var99,
        "current_es_95": cur_es95,
        "tail": tail,
        "backtests": backtests,
        "basel_99_full_sample": _bt.basel_traffic_light(int(exc99.sum()),
                                                       n_obs=n, alpha=0.01),
        "total_return": float(v.iloc[-1] / vnz.iloc[0] - 1)
        if len(vnz) > 1 else float("nan"),
    }

    if target_weights is not None:
        if holdings is None or last_prices is None:
            raise ValueError("rebalance section needs holdings and last_prices")
        drift_df = _rb.drift_table(holdings, target_weights, last_prices)
        sig = _rb.rebalance_signal(drift_df, abs_threshold=drift_threshold)
        trades = _rb.drift_trades(drift_df)
        summary["rebalance"] = {
            "signal": sig["signal"],
            "reason": sig["reason"],
            "worst_symbol": sig["worst_symbol"],
            "worst_drift": sig["worst_drift"],
            "drift_table": drift_df,
            "trades": trades,
        }

    return report_df, summary


def write_markdown_report(report_df: pd.DataFrame, summary: dict,
                          path: str, tail_rows: int = 30) -> str:
    """Write the human-readable markdown report; returns the path written.

    Shows the summary (risk numbers, tail diagnostics, backtests, rebalance)
    plus the last ``tail_rows`` of the daily table (the full series lives in
    the CSV the CLI writes alongside).
    """
    s = summary
    t = s["tail"]
    L = [f"# RiskLens risk report — {s['end']}",
         "",
         f"Sample: {s['n_obs']} daily observations ({s['start']} → {s['end']}).",
         "",
         "## Current risk",
         "",
         f"- Portfolio value: ${s['current_value']:,.2f}",
         f"- Total return: {s['total_return']:.2%}" if np.isfinite(s['total_return']) else "- Total return: n/a (no funded history yet)",
         f"- 95% VaR (hist): {s['current_var_95']:.2%} of portfolio",
         f"- 99% VaR (hist): {s['current_var_99']:.2%} of portfolio",
         f"- 95% ES (hist): {s['current_es_95']:.2%} of portfolio",
         "",
         "## Tail diagnostics",
         "",
    ]
    if t.get("note"):
        # Young sample: tail_summary returned NaNs with an explanation.
        L += [f"- {t['note']}", ""]
    else:
        L += [
         f"- Skew: {t['skew']:.3f}, kurtosis: {t['kurtosis']:.3f}",
         f"- Jarque–Bera: {t['jb_stat']:.1f} (p={t['jb_p']:.3g}) — "
         f"{'REJECTS normality' if t['jb_reject'] else 'normality not rejected'}",
         f"- Hill tail index: {t['hill_xi']:.3f} ± {t['hill_se']:.3f} "
         f"(k={t['hill_k']})" +
         (" — ⚠ infinite variance regime" if t["hill_xi"] >= 0.5 else ""),
         "",
        ]
    L += ["## Backtests", ""]
    bt = s["backtests"]
    if bt.get("kupiec_95") is None:
        L += [f"- {bt['note']}", ""]
    else:
        for name in ("kupiec_95", "kupiec_99", "christoffersen_99", "joint_99"):
            d = bt[name]
            L += [f"- {name}: LR={d['lr_stat']:.3f}, p={d['p_value']:.3g} — "
                  f"{'REJECT' if d['reject'] else 'no reject'}"]
        L += [""]
    bl = s["basel_99_full_sample"]
    L += [f"Basel 99% traffic light (full sample): **{bl['zone'].upper()}** "
          f"({bl['n_exceptions']} exceptions / {bl['n_obs']} obs).", ""]

    if "rebalance" in s:
        rb = s["rebalance"]
        L += ["## Rebalance",
              "",
              f"- Signal: **{'REBALANCE' if rb['signal'] else 'hold'}** — {rb['reason']}",
              ""]
        trades = rb["trades"]
        L += ["| symbol | qty | price | mkt value | actual wt | target wt | "
              "drift | trade qty | trade value | side |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for sym, row in trades.iterrows():
            L += [f"| {sym} | {row['qty']:.4f} | {row['price']:.2f} | "
                  f"${row['market_value']:.2f} | {row['actual_weight']:.2%} | "
                  f"{row['target_weight']:.2%} | {row['drift']:+.2%} | "
                  f"{row['trade_qty']:+.4f} | ${row['trade_value']:+.2f} | "
                  f"{row['side']} |"]
        L += [""]

    L += [f"## Daily series (last {tail_rows})", "",
          "| date | value | return | VaR 95% | VaR 99% | exc 95 | exc 99 | light |",
          "|---|---|---|---|---|---|---|---|"]
    tail_df = report_df.tail(tail_rows)
    for _, row in tail_df.iterrows():
        d = pd.Timestamp(row["date"]).date()
        L += [f"| {d} | ${row['portfolio_value']:,.2f} | "
              f"{row['daily_return']:.2%} | {row['var_95']:.2%} | "
              f"{row['var_99']:.2%} | {int(row['exception_95'])} | "
              f"{int(row['exception_99'])} | {row['traffic_light']} |"]
    L += ["", "_RiskLens: daily market-risk watch._"]
    text = "\n".join(L) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return path
