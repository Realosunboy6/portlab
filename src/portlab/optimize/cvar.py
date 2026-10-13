"""CVaR (expected shortfall) portfolio optimization via the
Rockafellar-Uryasev linear program — scipy.linprog, no cvxpy required."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from ..config import DEFAULT_ALPHA


def min_cvar(
    rets: pd.DataFrame,
    alpha: float = DEFAULT_ALPHA,
    min_return: float | None = None,
    bounds=(0.0, 1.0),
) -> pd.Series:
    """Minimize historical CVaR_alpha of per-period portfolio returns.

    min_return: optional lower bound on mean per-period return.
    Decision variables: [w (n), zeta (1), u (T)] minimizing
    zeta + 1/((1-alpha)T) * sum(u),  u_t >= -r_t·w - zeta,  u_t >= 0.
    """
    R = rets.values
    T, n = R.shape
    inv = 1.0 / ((1 - alpha) * T)

    c = np.concatenate([np.zeros(n), [1.0], np.full(T, inv)])

    # -R w - zeta - u <= 0  ->  rows: t
    rows, cols, vals = [], [], []
    for t in range(T):
        rows.extend([t] * n); cols.extend(range(n)); vals.extend(-R[t])
        rows.append(t); cols.append(n); vals.append(-1.0)
        rows.append(t); cols.append(n + 1 + t); vals.append(-1.0)
    A_ub = coo_matrix((vals, (rows, cols)), shape=(T, n + 1 + T)).tocsr()
    b_ub = np.zeros(T)

    if min_return is not None:
        mean_row = np.concatenate([-R.mean(axis=0), [0.0], np.zeros(T)])
        A_ub = coo_matrix(np.vstack([A_ub.toarray(), mean_row]))
        b_ub = np.append(b_ub, -min_return)

    A_eq = np.concatenate([np.ones(n), [0.0], np.zeros(T)]).reshape(1, -1)
    b_eq = [1.0]

    lo, hi = bounds
    var_bounds = [(lo, hi)] * n + [(None, None)] + [(0, None)] * T

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=var_bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"CVaR LP failed: {res.message}")
    w = res.x[:n]
    return pd.Series(w / w.sum(), index=rets.columns, name="weight")


def cvar_frontier(rets: pd.DataFrame, alphas=(0.90, 0.95, 0.975, 0.99),
                  bounds=(0.0, 1.0)) -> pd.DataFrame:
    """Min-CVaR portfolios across confidence levels (original Phase 6 sweep)."""
    from ..metrics import cvar_historical
    rows = []
    for a in alphas:
        w = min_cvar(rets, alpha=a, bounds=bounds)
        pr = rets @ w
        rows.append({"alpha": a, "mean_return": pr.mean(),
                     "cvar": cvar_historical(pr, a),
                     **{f"w_{c}": w[c] for c in rets.columns}})
    return pd.DataFrame(rows)


def min_cdar(
    rets: pd.DataFrame,
    alpha: float = DEFAULT_ALPHA,
    bounds=(0.0, 1.0),
) -> pd.Series:
    """Minimize Conditional Drawdown at Risk (Chekhlov-Uryasev-Zabarankin LP)
    on additive cumulative returns.

    Variables: [w (n), m (T) running max, zeta (1), u (T)] minimizing
    zeta + 1/((1-alpha)T) * sum(u),  u_t >= (m_t - C_t) - zeta,  u_t >= 0,
    m_t >= C_t,  m_t >= m_{t-1}.
    """
    Ccum = rets.cumsum().values
    T, n = Ccum.shape
    nv = n + T + 1 + T
    inv = 1.0 / ((1 - alpha) * T)
    rows, cols, vals, b = [], [], [], []
    r = 0
    # C_t - m_t <= 0
    for t in range(T):
        rows.extend([r] * n); cols.extend(range(n)); vals.extend(Ccum[t])
        rows.append(r); cols.append(n + t); vals.append(-1.0)
        b.append(0.0); r += 1
    # m_{t-1} - m_t <= 0
    for t in range(1, T):
        rows.extend([r, r]); cols.extend([n + t - 1, n + t]); vals.extend([1.0, -1.0])
        b.append(0.0); r += 1
    # (m_t - C_t) - zeta - u_t <= 0
    for t in range(T):
        rows.append(r); cols.append(n + t); vals.append(1.0)
        rows.extend([r] * n); cols.extend(range(n)); vals.extend(-Ccum[t])
        rows.append(r); cols.append(n + T); vals.append(-1.0)
        rows.append(r); cols.append(n + T + 1 + t); vals.append(-1.0)
        b.append(0.0); r += 1
    A_ub = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    A_eq = np.zeros((1, nv)); A_eq[0, :n] = 1.0
    lo, hi = bounds
    # m_t >= 0 encodes C_0 = 0: the starting wealth is itself a peak, so a
    # loss taken before any new high still counts as drawdown.
    var_bounds = ([(lo, hi)] * n + [(0, None)] * T + [(None, None)]
                  + [(0, None)] * T)
    c = np.zeros(nv); c[n + T] = 1.0; c[n + T + 1:] = inv
    res = linprog(c, A_ub=A_ub, b_ub=np.array(b), A_eq=A_eq, b_eq=[1.0],
                  bounds=var_bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"CDaR LP failed: {res.message}")
    w = res.x[:n]
    return pd.Series(w / w.sum(), index=rets.columns, name="weight")


def max_return_at_cvar(
    rets: pd.DataFrame,
    alpha: float = DEFAULT_ALPHA,
    max_cvar: float | None = None,
    bounds=(0.0, 1.0),
) -> pd.Series:
    """Maximize mean return subject to a CVaR budget (positive number).

    The FactSet-style direction of scenario-based tail optimization:
    "how much return can I get while keeping expected tail loss under X".
    ``alpha`` is the CONFIDENCE level (0.95, matching ``min_cvar`` — note
    this differs from ``portlab.risk``, where alpha is the tail probability).
    Rockafellar-Uryasev LP: max mu'w s.t.
    zeta + 1/((1-alpha)T) * sum(u) <= max_cvar, u_t >= -r_t.w - zeta, u >= 0.
    """
    R = rets.values
    T, n = R.shape
    if max_cvar is None or max_cvar <= 0:
        raise ValueError("max_cvar must be a positive CVaR budget")
    inv = 1.0 / ((1 - alpha) * T)
    mu = R.mean(axis=0)

    # variables: [w (n), zeta (1), u (T)]; maximize mu'w  <=>  minimize -mu'w
    c = np.concatenate([-mu, [0.0], np.zeros(T)])
    rows, cols, vals = [], [], []
    # -R w - zeta - u <= 0
    for t in range(T):
        rows.extend([t] * n); cols.extend(range(n)); vals.extend(-R[t])
        rows.append(t); cols.append(n); vals.append(-1.0)
        rows.append(t); cols.append(n + 1 + t); vals.append(-1.0)
    # zeta + inv*sum(u) <= max_cvar
    rows.extend([T] * (1 + T)); cols.extend([n] + list(range(n + 1, n + 1 + T)))
    vals.extend([1.0] + [inv] * T)
    A_ub = coo_matrix((vals, (rows, cols)), shape=(T + 1, n + 1 + T)).tocsr()
    b_ub = np.concatenate([np.zeros(T), [max_cvar]])

    A_eq = np.concatenate([np.ones(n), [0.0], np.zeros(T)]).reshape(1, -1)
    lo, hi = bounds
    var_bounds = [(lo, hi)] * n + [(None, None)] + [(0, None)] * T
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=var_bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"max_return_at_cvar LP failed: {res.message}")
    w = res.x[:n]
    return pd.Series(w / w.sum(), index=rets.columns, name="weight")


def cvar_efficient_frontier(
    rets: pd.DataFrame,
    alpha: float = DEFAULT_ALPHA,
    n_points: int = 20,
    bounds=(0.0, 1.0),
) -> pd.DataFrame:
    """Mean vs CVaR efficient frontier at fixed confidence ``alpha``.

    Traces min-CVaR portfolios over a grid of target returns between the
    min-CVaR portfolio's return and the max-return asset's mean. Each row:
    target_return, mean_return, cvar (positive loss number), weights.
    """
    from ..metrics import cvar_historical
    w0 = min_cvar(rets, alpha=alpha, bounds=bounds)
    r_min = float((rets @ w0).mean())
    r_max = float(rets.mean().max())
    rows = []
    for target in np.linspace(r_min, r_max, n_points):
        w = min_cvar(rets, alpha=alpha, min_return=target, bounds=bounds)
        pr = rets @ w
        rows.append({"target_return": target,
                     "mean_return": float(pr.mean()),
                     "cvar": float(cvar_historical(pr, alpha)),
                     **{f"w_{c}": float(w[c]) for c in rets.columns}})
    return pd.DataFrame(rows)
