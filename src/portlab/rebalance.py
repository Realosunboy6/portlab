"""Individual-investor utilities: 'what do I actually do today?'"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rebalance_trades(
    current_holdings: dict[str, float],
    target_weights: dict[str, float] | pd.Series,
    cash_to_add: float = 0.0,
    min_trade: float = 0.0,
    basis: dict[str, float] | None = None,
    tax_aware: bool = False,
) -> pd.DataFrame:
    """Exact buy/sell dollars to move current holdings to target weights.

    current_holdings: {ticker: current dollar value} (use 0 for new positions).
    target_weights: desired allocation (normalized automatically).
    cash_to_add: new money to invest (+) or cash to raise (-).
    min_trade: suppress trades smaller than this many dollars.
    basis: optional {ticker: cost basis dollars} — adds an estimated
        realized-gain column for sells.
    tax_aware: sort sells so realized losses come first (harvest losses).
    """
    w = pd.Series(target_weights, dtype=float)
    w = w / w.sum()
    cur = pd.Series(current_holdings, dtype=float)
    all_assets = w.index.union(cur.index)
    cur = cur.reindex(all_assets).fillna(0.0)
    w = w.reindex(all_assets).fillna(0.0)

    total = cur.sum() + cash_to_add
    if total <= 0:
        raise ValueError("nothing to allocate: holdings + cash_to_add <= 0")
    target = total * w
    trade = target - cur
    trade[trade.abs() < min_trade] = 0.0

    out = pd.DataFrame({
        "current": cur, "target": target.round(2), "trade": trade.round(2),
        "action": np.where(trade > 0, "BUY", np.where(trade < 0, "SELL", "HOLD")),
    })
    if basis is not None:
        b = pd.Series(basis, dtype=float).reindex(all_assets).fillna(np.nan)
        frac_sold = np.where((trade < 0) & (cur > 0), -trade / cur, 0.0)
        out["est_realized_gain"] = ((cur - b) * frac_sold).round(2)
    if tax_aware and "est_realized_gain" in out:
        sells = out[out["action"] == "SELL"].sort_values("est_realized_gain")
        rest = out[out["action"] != "SELL"]
        out = pd.concat([sells, rest])
    out.attrs["unallocated_cash"] = round(float(cash_to_add - out["trade"].sum()), 2)
    return out
