"""Transaction-cost modeling for trades and rebalances.

A backtest or optimizer that ignores trading costs lies about returns. This
module implements the standard practitioner cost stack (cf. cvxportfolio,
Almgren-Chriss):

    fixed         flat $ per non-zero trade (ticket charges)
    proportional  bps x |trade value| (commissions, fees, taxes)
    spread        half-spread crossing: (spread_bps / 2) x |trade value|
    market impact square-root model: coef x sigma x (|shares| / ADV)^power
                  x |trade value|  (power = 0.5 is the industry default)

Models compose via :class:`CombinedCost`. :func:`portfolio_costs` prices a
whole rebalance ticket from a per-symbol market-data map, and
:func:`turnover_penalty` gives the L1 penalty term optimizers add to the
objective to discourage churn.

Conventions: ``trade_value`` is signed notional (+ = buy); costs are always
non-negative. ``sigma`` is daily volatility (decimal), ``adv`` average daily
volume in *shares*, ``spread_bps`` the full bid-ask spread in basis points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "FixedCost",
    "ProportionalCost",
    "SpreadCost",
    "MarketImpact",
    "CombinedCost",
    "portfolio_costs",
    "turnover",
    "turnover_penalty",
    "net_of_costs",
]

_BPS = 1e-4


@dataclass(frozen=True)
class FixedCost:
    """Flat ``per_trade`` dollars charged on every non-zero trade."""
    per_trade: float

    def cost(self, trade_value: float, **_) -> float:
        return float(self.per_trade) if trade_value != 0 else 0.0


@dataclass(frozen=True)
class ProportionalCost:
    """``bps`` basis points of traded notional (commissions, fees)."""
    bps: float

    def cost(self, trade_value: float, **_) -> float:
        return abs(float(trade_value)) * float(self.bps) * _BPS


@dataclass(frozen=True)
class SpreadCost:
    """Half-spread crossing cost from the full ``spread_bps`` bid-ask spread."""
    spread_bps: float

    def cost(self, trade_value: float, **_) -> float:
        return abs(float(trade_value)) * float(self.spread_bps) * _BPS / 2.0


@dataclass(frozen=True)
class MarketImpact:
    """Square-root (power-law) market impact.

    cost = coef x sigma x (|shares| / adv)^power x |trade value|.

    ``coef`` ~ 1 is the usual calibration; ``power`` = 0.5 is the
    square-root model. Needs per-symbol ``sigma`` (daily vol) and ``adv``
    (average daily volume in shares); symbols missing either contribute 0
    impact rather than raising.
    """
    coef: float = 1.0
    power: float = 0.5

    def cost(self, trade_value: float, *, shares: float = 0.0,
             sigma: float | None = None, adv: float | None = None,
             **_) -> float:
        tv = abs(float(trade_value))
        if tv == 0 or not sigma or not adv or sigma <= 0 or adv <= 0:
            return 0.0
        participation = abs(float(shares)) / float(adv)
        return float(self.coef) * float(sigma) * participation ** float(self.power) * tv


@dataclass(frozen=True)
class CombinedCost:
    """Sum of several cost models, evaluated with the same trade context."""
    models: tuple = field(default_factory=tuple)

    def __init__(self, models):
        object.__setattr__(self, "models", tuple(models))

    def cost(self, trade_value: float, **ctx) -> float:
        return float(sum(m.cost(trade_value, **ctx) for m in self.models))


def turnover(trade_values, portfolio_value: float) -> float:
    """Two-sided turnover: sum |trade value| / portfolio value."""
    tv = np.asarray(list(trade_values), dtype=float)
    if not portfolio_value or portfolio_value <= 0:
        return float("nan")
    return float(np.abs(tv).sum() / portfolio_value)


def turnover_penalty(trade_values, gamma: float) -> float:
    """L1 turnover penalty ``gamma x sum|trade value|`` for optimizer use.

    Subtract this from the objective (or add to a minimization objective) to
    discourage churn; ``gamma`` in the same units as returns per dollar.
    """
    tv = np.asarray(list(trade_values), dtype=float)
    return float(gamma) * float(np.abs(tv).sum())


def portfolio_costs(trades: dict, market: dict, model) -> pd.DataFrame:
    """Price a whole rebalance ticket.

    ``trades``: symbol -> signed *shares* (+ = buy).
    ``market``: symbol -> dict with ``price`` and optionally ``sigma``
    (daily vol), ``adv`` (avg daily volume, shares), ``spread_bps``.
    ``model``: a cost model (or CombinedCost) with ``.cost(trade_value, **ctx)``.

    Returns a DataFrame indexed by symbol with columns shares, trade_value,
    cost, cost_bps (cost per unit traded, in bps). Symbols missing from
    ``market`` are skipped.
    """
    rows = []
    for sym, shares in trades.items():
        m = market.get(str(sym).strip().upper())
        if m is None or shares == 0:
            continue
        price = float(m["price"])
        tv = float(shares) * price
        ctx = {"shares": float(shares),
               "price": price,
               "sigma": m.get("sigma"),
               "adv": m.get("adv"),
               "spread_bps": m.get("spread_bps")}
        c = float(model.cost(tv, **ctx))
        rows.append({"symbol": str(sym).strip().upper(),
                     "shares": float(shares),
                     "trade_value": tv,
                     "cost": c,
                     "cost_bps": (c / abs(tv) / _BPS) if tv else 0.0})
    df = pd.DataFrame(rows, columns=["symbol", "shares", "trade_value",
                                     "cost", "cost_bps"])
    if not df.empty:
        df = df.set_index("symbol")
    return df


def net_of_costs(gross_pnl: float, cost: float) -> float:
    """Net P&L after subtracting transaction costs."""
    return float(gross_pnl) - float(cost)
