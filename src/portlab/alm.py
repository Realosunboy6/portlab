"""Asset-liability modeling (ALM): liability present value, funded status,
and surplus optimization in the Sharpe–Tint (1990) tradition.

A pension or endowment holds assets A against a schedule of future payouts L.
The funded ratio A/PV(L) and the surplus S = A − L are the objects of
interest. The liability-aware objective is the surplus Sharpe,
E[R_A − R_L]/Vol(R_A − R_L); the liability-hedging portfolio minimizes
surplus volatility. Together they give the classic LDI two-fund
decomposition: hedge the liabilities, then seek surplus return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .optimize.core import solve_weights


def liability_schedule(annual_payment: float, years: int, growth: float = 0.0,
                       start: str | None = None, freq: str = "ME") -> pd.Series:
    """Future benefit payouts, one per period, growing at `growth` per year."""
    ppy = {"ME": 12, "QE": 4, "YE": 1}[freq]
    n = years * ppy
    idx = (pd.date_range(start, periods=n, freq=freq) if start is not None
           else pd.RangeIndex(1, n + 1, name="period"))
    t = np.arange(1, n + 1) / ppy
    payments = annual_payment / ppy * (1 + growth) ** t
    return pd.Series(payments, index=idx, name="benefit_payment")


def present_value(schedule: pd.Series, discount_rate: float,
                  periods_per_year: int = 12) -> float:
    """Present value of the payout schedule at a flat annual discount rate."""
    t = np.arange(1, len(schedule) + 1) / periods_per_year
    return float((schedule.values / (1 + discount_rate) ** t).sum())


def funded_status(assets: float, schedule: pd.Series, discount_rate: float,
                  periods_per_year: int = 12) -> dict:
    """Funded ratio, surplus/deficit and PV of liabilities."""
    pv = present_value(schedule, discount_rate, periods_per_year)
    return {"pv_liabilities": pv,
            "funded_ratio": assets / pv if pv > 0 else float("nan"),
            "surplus": assets - pv}


def surplus_optimize(mu: pd.Series, cov: pd.DataFrame, liab_mu: float,
                     liab_var: float, cross_cov,
                     bounds=(0.0, 1.0), groups=None) -> pd.Series:
    """Maximize the surplus Sharpe (w'μ − μ_L)/√(w'Σw − 2w'c + σ_L²).

    Sharpe & Tint (1990): the liability-aware objective, where c is the
    asset-liability covariance vector. Falls back to the hedging portfolio
    when no asset beats the liability growth rate.
    """
    m, C, c = mu.values, cov.values, np.asarray(cross_cov, float).reshape(-1)
    assert c.shape == m.shape, "cross_cov must have one entry per asset"

    def neg_surplus_sharpe(w):
        s_var = max(float(w @ C @ w - 2 * w @ c + liab_var), 1e-16)
        return -((w @ m - liab_mu) / np.sqrt(s_var))

    if np.all(m <= liab_mu):
        return hedge_portfolio(cov, cross_cov, bounds, groups)
    return solve_weights(neg_surplus_sharpe, len(mu), bounds, groups,
                         index=mu.index)


def hedge_portfolio(cov: pd.DataFrame, cross_cov,
                    bounds=(0.0, 1.0), groups=None) -> pd.Series:
    """Liability-hedging portfolio: min_w w'Σw − 2w'c (minimum surplus
    variance, ignoring expected returns)."""
    C, c = cov.values, np.asarray(cross_cov, float).reshape(-1)
    assert c.shape == (C.shape[0],), "cross_cov must have one entry per asset"
    return solve_weights(lambda w: float(w @ C @ w - 2 * w @ c),
                         C.shape[0], bounds, groups, index=cov.index)


def funded_projection(assets: float, schedule: pd.Series,
                      port_return_annual: float, discount_rate: float,
                      periods_per_year: int = 12) -> pd.DataFrame:
    """Deterministic funded-ratio path: assets earn `port_return_annual` and
    pay the scheduled benefits; PV(liabilities) rolls at the discount rate
    as each payment goes out."""
    n = len(schedule)
    per_ret = (1 + port_return_annual) ** (1 / periods_per_year) - 1
    per_disc = (1 + discount_rate) ** (1 / periods_per_year) - 1
    pays = schedule.values
    A = np.zeros(n)
    PV = np.zeros(n)
    a = assets
    for i, pay in enumerate(pays):
        a = (a - pay) * (1 + per_ret)
        pv_rem = (pays[i + 1:] / (1 + per_disc) ** np.arange(1, n - i)).sum() \
            if i + 1 < n else 0.0
        A[i], PV[i] = a, pv_rem
    with np.errstate(divide="ignore", invalid="ignore"):
        fr = np.where(PV > 0, A / PV, np.nan)
    return pd.DataFrame({"assets": A, "pv_liabilities": PV,
                         "funded_ratio": fr}, index=schedule.index)
