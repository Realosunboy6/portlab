"""Rebalance signals: WHEN to trade, not just what to hold.

The optimizer outputs *target weights*; fills are *reality*; prices move.
This module measures the drift between the two and answers the only question
that matters day to day: is it time to rebalance?

Two trigger philosophies (both standard practitioner practice):

- *Threshold*: trade when any position has drifted too far from target.
  Default bands — 5 percentage points absolute drift, or 25% relative drift —
  are the common practitioner bands (e.g. the bands Vanguard/DFA-style
  rebalancing literature uses as a starting point; tune to turnover costs).
- *Calendar*: trade on a schedule (e.g. quarterly), regardless of drift.

The two compose: calendar says "look", threshold says "act".
"""

from __future__ import annotations

import datetime as dt
import functools
import warnings

import numpy as np
import pandas as pd

__all__ = ["drift_table", "rebalance_signal", "drift_trades"]

#: Default absolute drift band: |actual - target| in weight points.
DEFAULT_ABS_THRESHOLD = 0.05
#: Default relative drift band: |actual - target| / target.
DEFAULT_REL_THRESHOLD = 0.25


def drift_table(holdings: dict[str, float],
                target_weights: dict[str, float],
                prices: dict[str, float]) -> pd.DataFrame:
    """Target-vs-actual drift table.

    ``holdings``: symbol -> share quantity (from fills; fractional OK).
    ``target_weights``: symbol -> target portfolio weight (should sum to ~1).
    ``prices``: symbol -> current price per share (Yahoo is fine here — this
    is valuation, not cost-basis accounting; the ledger stays broker-side only).

    Symbols present in only one of holdings/targets appear with zeros on the
    other side (a held-but-untargeted symbol shows target 0 and full drift).

    Returns DataFrame indexed by symbol with columns: qty, price,
    market_value, actual_weight, target_weight, drift (= actual - target).
    """
    symbols = sorted(set(holdings) | set(target_weights))
    rows = []
    for s in symbols:
        qty = float(holdings.get(s, 0.0))
        price = prices.get(s)
        if price is None or not np.isfinite(price) or price <= 0:
            raise ValueError(f"missing or invalid price for symbol {s!r}")
        mv = qty * price
        rows.append({"symbol": s, "qty": qty, "price": float(price),
                     "market_value": mv,
                     "target_weight": float(target_weights.get(s, 0.0))})
    df = pd.DataFrame(rows).set_index("symbol")
    total = df["market_value"].sum()
    df["actual_weight"] = df["market_value"] / total if total > 0 else 0.0
    df["drift"] = df["actual_weight"] - df["target_weight"]
    return df[["qty", "price", "market_value", "actual_weight",
               "target_weight", "drift"]]


def rebalance_signal(drift_df: pd.DataFrame, method: str = "threshold",
                     abs_threshold: float = DEFAULT_ABS_THRESHOLD,
                     rel_threshold: float = DEFAULT_REL_THRESHOLD,
                     last_rebalance: dt.date | None = None,
                     freq_days: int = 90,
                     today: dt.date | None = None) -> dict:
    """Should the portfolio be rebalanced now?

    method='threshold': signal True if any |drift| > abs_threshold (weight
        points) OR any |drift|/target_weight > rel_threshold. Symbols with
        target_weight == 0 are judged on the absolute band only (no
        divide-by-zero); a held-but-untargeted position always trips the
        absolute band once it exceeds it.
    method='calendar': signal True if today >= last_rebalance + freq_days.
        ``last_rebalance`` is required; ``today`` defaults to date.today().

    Returns dict: signal (bool), reason (str), worst_symbol, worst_drift
    (largest |drift|), abs_threshold, rel_threshold, method.
    """
    if method not in ("threshold", "calendar"):
        raise ValueError("method must be 'threshold' or 'calendar'")
    worst_symbol = drift_df["drift"].abs().idxmax() if len(drift_df) else None
    worst_drift = (float(drift_df.loc[worst_symbol, "drift"])
                   if worst_symbol is not None else 0.0)

    if method == "calendar":
        if last_rebalance is None:
            raise ValueError("calendar method requires last_rebalance")
        today = today or dt.date.today()
        due = today >= last_rebalance + dt.timedelta(days=freq_days)
        return {
            "signal": bool(due),
            "reason": (f"calendar: {today} >= {last_rebalance} + {freq_days}d"
                       if due else
                       f"calendar: next rebalance "
                       f"{last_rebalance + dt.timedelta(days=freq_days)}"),
            "worst_symbol": worst_symbol,
            "worst_drift": worst_drift,
            "abs_threshold": abs_threshold,
            "rel_threshold": rel_threshold,
            "method": method,
        }

    # Threshold mode: check absolute band, then relative band.
    abs_hit = drift_df["drift"].abs() > abs_threshold
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = drift_df["drift"].abs() / drift_df["target_weight"].replace(0, np.nan)
    rel_hit = rel > rel_threshold
    hit = abs_hit | rel_hit.fillna(False)
    if hit.any():
        sym = drift_df.loc[hit, "drift"].abs().idxmax()
        d = float(drift_df.loc[sym, "drift"])
        t = float(drift_df.loc[sym, "target_weight"])
        which = ("absolute" if abs(d) > abs_threshold else "relative")
        return {
            "signal": True,
            "reason": (f"threshold ({which}): {sym} drift {d:+.2%} vs target "
                       f"{t:.2%} exceeds band"),
            "worst_symbol": worst_symbol,
            "worst_drift": worst_drift,
            "abs_threshold": abs_threshold,
            "rel_threshold": rel_threshold,
            "method": method,
        }
    return {
        "signal": False,
        "reason": (f"threshold: all |drift| <= {abs_threshold:.0%} and all "
                   f"relative drift <= {rel_threshold:.0%}"),
        "worst_symbol": worst_symbol,
        "worst_drift": worst_drift,
        "abs_threshold": abs_threshold,
        "rel_threshold": rel_threshold,
        "method": method,
    }


def drift_trades(drift_df: pd.DataFrame,
                 portfolio_value: float | None = None) -> pd.DataFrame:
    """Shares to trade per symbol to restore target weights.

    trade_qty = (target_weight * portfolio_value - market_value) / price.
    Positive = BUY, negative = SELL. ``portfolio_value`` defaults to the
    drift table's total market value. Includes trade_value (= qty * price)
    and side columns. Symbols already at target produce ~zero trades.

    (Renamed from ``rebalance_trades`` in 2026-10-09 so the name no longer
    collides with the dollar-based ``portlab.rebalance.rebalance_trades``;
    the old name still works with a DeprecationWarning.)
    """
    df = drift_df.copy()
    total = portfolio_value if portfolio_value is not None else float(
        df["market_value"].sum())
    if total <= 0:
        raise ValueError("portfolio value must be positive")
    df["target_value"] = df["target_weight"] * total
    df["trade_qty"] = (df["target_value"] - df["market_value"]) / df["price"]
    df["trade_value"] = df["trade_qty"] * df["price"]
    df["side"] = np.where(df["trade_qty"] > 1e-12, "BUY",
                          np.where(df["trade_qty"] < -1e-12, "SELL", "HOLD"))
    return df[["qty", "price", "market_value", "actual_weight", "target_weight",
               "drift", "trade_qty", "trade_value", "side"]]


@functools.wraps(drift_trades)
def rebalance_trades(*args, **kwargs):
    """Deprecated alias of :func:`drift_trades` (name collided with
    ``portlab.rebalance.rebalance_trades``)."""
    warnings.warn(
        "portlab.risk.rebalance.rebalance_trades is deprecated; "
        "use drift_trades instead.",
        DeprecationWarning, stacklevel=2,
    )
    return drift_trades(*args, **kwargs)
