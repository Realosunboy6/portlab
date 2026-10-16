"""Multi-period, turnover-aware portfolio optimization.

Plans trades over a horizon of T periods in one joint optimization, instead
of re-solving a myopic single-period problem every rebalance date. The
formulation follows Boyd et al., "Multi-Period Trading via Convex
Optimization" (Foundations and Trends in Optimization, 2017):

    min  sum_t [ (gamma/2) w_t' Sigma_t w_t - mu_t' w_t
                 + kappa * ||w_t - w_{t-1}||_1 ]
    s.t. sum(w_t) = 1, w_t >= 0,  w_0 given

The L1 turnover penalty (kappa = proportional one-way cost) is the
turnover-aware part: it prices the round-trip cost of getting into and out
of positions, so the planner prefers slow ramps over churn. With kappa = 0
each period decouples into an independent single-period mean-variance
problem; with kappa -> infinity the plan holds w_0 forever. Both limits are
tested.

The L1 term is split into positive/negative trade variables
(w_t - w_{t-1} = z+_t - z-_t, z+- >= 0), turning the problem into a QP with
nonnegativity constraints solved by the house augmented-Lagrangian solver
(:func:`portlab.optimize.auglag.auglag_qp`). Long-only, fully invested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .optimize.auglag import auglag_qp

__all__ = ["multiperiod_weights"]


def _as_frame(mus) -> pd.DataFrame:
    m = pd.DataFrame(mus, dtype=float)
    if m.empty:
        raise ValueError("mus must have at least one period")
    return m


def multiperiod_weights(w0, mus, covs,
                        risk_aversion: float = 5.0,
                        tc_bps: float = 10.0) -> dict:
    """Optimal weight trajectory over T periods with turnover penalty.

    ``w0``: current weights (Series, sums to 1).
    ``mus``: DataFrame (T rows x n assets) of per-period expected returns.
    ``covs``: list of T (n x n) covariance DataFrames (or 3D array).
    ``risk_aversion``: gamma in the objective.
    ``tc_bps``: one-way proportional transaction cost in basis points —
        the L1 turnover penalty kappa. Consistent with
        ``portlab.costs.ProportionalCost``.

    Returns dict with ``weights`` (T x n planned weights), ``trades``
    (w_t - w_{t-1}), ``turnover`` (per-period one-way turnover),
    ``total_turnover``, and ``objective`` (the minimized value).
    """
    w0 = pd.Series(w0, dtype=float)
    mu = _as_frame(mus)
    T, n = mu.shape
    assets = mu.columns
    w0 = w0.reindex(assets).fillna(0.0)
    if abs(w0.sum() - 1.0) > 1e-8:
        raise ValueError(f"w0 must sum to 1, got {w0.sum()}")
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")
    if tc_bps < 0:
        raise ValueError("tc_bps must be non-negative")
    covs = np.asarray(covs, dtype=float)
    if covs.shape != (T, n, n):
        raise ValueError(f"covs must have shape ({T}, {n}, {n})")
    kappa = tc_bps / 1e4

    # variable order: [w_1..w_T | zp_1..zp_T | zm_1..zm_T], each block n*T
    N = 3 * n * T

    C = np.zeros((N, N))
    for t in range(T):
        C[t * n:(t + 1) * n, t * n:(t + 1) * n] = 0.5 * risk_aversion * covs[t]
    c = np.concatenate([-mu.to_numpy().ravel(),
                        np.full(n * T, kappa), np.full(n * T, kappa)])

    # equalities: (a) budget sum(w_t) = 1  [T rows]
    #             (b) w_t - w_{t-1} - zp_t + zm_t = 0  [n*T rows], w_0 given
    n_eq = T + n * T
    A = np.zeros((n_eq, N))
    b = np.zeros(n_eq)
    for t in range(T):
        A[t, t * n:(t + 1) * n] = 1.0
        b[t] = 1.0
    for t in range(T):
        rows = slice(T + t * n, T + (t + 1) * n)
        A[rows, t * n:(t + 1) * n] = np.eye(n)                    # w_t
        if t == 0:
            b[rows] = w0.to_numpy()                              # - w_0 on RHS
        else:
            A[rows, (t - 1) * n:t * n] = -np.eye(n)               # - w_{t-1}
        A[rows, n * T + t * n:n * T + (t + 1) * n] = -np.eye(n)   # - zp_t
        A[rows, 2 * n * T + t * n:2 * n * T + (t + 1) * n] = np.eye(n)  # + zm_t

    y0 = np.concatenate([np.tile(w0.to_numpy(), T),
                         np.zeros(2 * n * T)])
    y = auglag_qp(C, A, b, y0=y0, c=c)

    wmat = y[:n * T].reshape(T, n)
    weights = pd.DataFrame(wmat, index=mu.index, columns=assets)
    trades = weights.diff()
    trades.iloc[0] = weights.iloc[0] - w0.values
    turnover = trades.abs().sum(axis=1) / 2.0  # one-way
    return {"weights": weights,
            "trades": trades,
            "turnover": turnover.rename("turnover"),
            "total_turnover": float(turnover.sum()),
            "objective": float(y @ C @ y + c @ y)}
