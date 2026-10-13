"""Optimization objectives beyond plain mean-variance:
max Sortino, Kelly criterion, Omega ratio, minimum maximum-drawdown,
minimum tracking error, and maximum information ratio."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from ..config import DEFAULT_RF, TRADING_DAYS
from .core import solve_weights


def max_sortino(rets: pd.DataFrame, rf: float = DEFAULT_RF,
                periods: int = TRADING_DAYS, bounds=(0.0, 1.0), groups=None) -> pd.Series:
    R = rets.values
    thr = rf / periods

    def neg_sortino(w):
        pr = R @ w - thr
        downside = np.sqrt(np.mean(np.minimum(pr, 0.0) ** 2))
        if downside <= 1e-12:
            return -1e6 if pr.mean() > 0 else 0.0
        return -pr.mean() / downside

    return solve_weights(neg_sortino, R.shape[1], bounds, groups, index=rets.columns)


def kelly(rets: pd.DataFrame, bounds=(0.0, 1.0), groups=None) -> pd.Series:
    """Maximize expected log growth E[log(1 + w·r)] over historical returns."""
    R = rets.values

    def neg_log_growth(w):
        g = 1.0 + R @ w
        if (g <= 1e-9).any():
            return 1e6
        return -np.mean(np.log(g))

    return solve_weights(neg_log_growth, R.shape[1], bounds, groups, index=rets.columns)


def max_omega(rets: pd.DataFrame, threshold_annual: float = 0.0,
              periods: int = TRADING_DAYS, bounds=(0.0, 1.0), groups=None) -> pd.Series:
    R = rets.values
    thr = threshold_annual / periods

    def neg_omega(w):
        d = R @ w - thr
        losses = -d[d < 0].sum()
        gains = d[d > 0].sum()
        if losses <= 1e-12:
            return -1e6 if gains > 0 else 0.0
        return -gains / losses

    return solve_weights(neg_omega, R.shape[1], bounds, groups, index=rets.columns)


def min_max_drawdown(rets: pd.DataFrame, bounds=(0.0, 1.0)) -> pd.Series:
    """Minimize historical maximum drawdown via linear programming.

    Uses the additive-wealth approximation W_t = w · cumsum(r): variables
    [w (n), m (T) running max, u (1) max drawdown].
    """
    Ccum = rets.cumsum().values
    T, n = Ccum.shape
    nv = n + T + 1
    rows, cols, vals, b = [], [], [], []
    r = 0
    # m_t >= w·C_t   ->  w·C_t - m_t <= 0
    for t in range(T):
        rows.extend([r] * n); cols.extend(range(n)); vals.extend(Ccum[t])
        rows.append(r); cols.append(n + t); vals.append(-1.0)
        b.append(0.0); r += 1
    # m_t >= m_{t-1} ->  m_{t-1} - m_t <= 0
    for t in range(1, T):
        rows.extend([r, r]); cols.extend([n + t - 1, n + t]); vals.extend([1.0, -1.0])
        b.append(0.0); r += 1
    # m_t - w·C_t <= u
    for t in range(T):
        rows.append(r); cols.append(n + t); vals.append(1.0)
        rows.extend([r] * n); cols.extend(range(n)); vals.extend(-Ccum[t])
        rows.append(r); cols.append(n + T); vals.append(-1.0)
        b.append(0.0); r += 1
    A_ub = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()

    A_eq = np.zeros((1, nv)); A_eq[0, :n] = 1.0
    lo, hi = bounds
    var_bounds = [(lo, hi)] * n + [(None, None)] * T + [(0, None)]
    c = np.zeros(nv); c[-1] = 1.0

    res = linprog(c, A_ub=A_ub, b_ub=np.array(b), A_eq=A_eq, b_eq=[1.0],
                  bounds=var_bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"max-drawdown LP failed: {res.message}")
    w = res.x[:n]
    return pd.Series(w / w.sum(), index=rets.columns, name="weight")


def min_tracking_error(rets: pd.DataFrame, bench: pd.Series,
                       bounds=(0.0, 1.0), groups=None) -> pd.Series:
    df = rets.join(bench.rename("_bench"), how="inner").dropna()
    R = df[rets.columns].values
    bvec = df["_bench"].values

    def te2(w):
        active = R @ w - bvec
        return float(active.var())

    return solve_weights(te2, R.shape[1], bounds, groups, index=rets.columns)


def max_information_ratio(rets: pd.DataFrame, bench: pd.Series,
                          bounds=(0.0, 1.0), groups=None) -> pd.Series:
    df = rets.join(bench.rename("_bench"), how="inner").dropna()
    R = df[rets.columns].values
    bvec = df["_bench"].values

    def neg_ir(w):
        active = R @ w - bvec
        sd = active.std(ddof=1)
        if sd <= 1e-12:
            return 0.0
        return -active.mean() / sd

    return solve_weights(neg_ir, R.shape[1], bounds, groups, index=rets.columns)


def geometric_frontier(rets: pd.DataFrame, n_points: int = 12,
                       periods: int = 252, bounds=(0.0, 1.0)) -> pd.DataFrame:
    """Geometric mean frontier (Bernstein & Wilkinson): for a sweep of
    volatility caps, maximize expected log growth instead of arithmetic mean.
    Rebalancing bonus/volatility drag is captured automatically."""
    R = rets.values
    vols = rets.std().values * np.sqrt(periods)

    def neg_log_growth(w):
        g = 1.0 + R @ w
        if (g <= 1e-9).any():
            return 1e6
        return -np.mean(np.log(g))

    rows = []
    C = rets.cov().values * periods
    for cap in np.linspace(vols.min() * 0.7, vols.max(), n_points):
        extra = [{"type": "ineq",
                  "fun": lambda w, c=cap, Q=C: c ** 2 - w @ Q @ w}]
        try:
            w = solve_weights(neg_log_growth, R.shape[1], bounds,
                              extra_constraints=extra, index=rets.columns)
        except Exception:
            continue
        pr = rets @ w
        geo = (1 + pr).prod() ** (periods / len(pr)) - 1
        rows.append({"vol_cap": cap,
                     "realized_vol": float(pr.std() * np.sqrt(periods)),
                     "geometric_mean": float(geo),
                     **{f"w_{c}": w[c] for c in rets.columns}})
    return pd.DataFrame(rows)
