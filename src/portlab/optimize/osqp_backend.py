"""OSQP backend for mean-variance QPs (free, no license, no size limit).

ADMM-based; slower than FISTA/Gurobi on dense large QPs but a robust
independent cross-check and fallback. Every entry raises _OSQPUnavailable
when osqp is missing or the solve fails, so callers fall through.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class _OSQPUnavailable(Exception):
    """Raised when the OSQP path cannot be used."""


def _import_osqp():
    try:
        import osqp
        return osqp
    except ImportError as e:
        raise _OSQPUnavailable("osqp is not installed") from e


def _as_bounds_list(n: int, bounds) -> list[tuple[float, float]]:
    if isinstance(bounds, tuple) and len(bounds) == 2 and np.isscalar(bounds[0]):
        return [bounds] * n
    return list(bounds)


def _solve_qp_osqp(P: np.ndarray, q: np.ndarray, n: int, bounds, groups,
                   A_extra=None, l_extra=None, u_extra=None,
                   name: str = "qp", fully_invested: bool = True) -> np.ndarray:
    """min 1/2 x'Px + q'x s.t. [sum x = 1,] box/groups, optional extra rows."""
    import scipy.sparse as sp

    osqp = _import_osqp()
    P = sp.csc_matrix(np.asarray(P, dtype=float))
    q = np.asarray(q, dtype=float).ravel()
    blist = _as_bounds_list(n, bounds)
    rows, l, u = [], [], []
    if fully_invested:
        rows.append(np.ones(n)); l.append(1.0); u.append(1.0)  # fully invested
    for i, (lo, hi) in enumerate(blist):  # box as rows (OSQP has no var bounds)
        e = np.zeros(n)
        e[i] = 1.0
        rows.append(e)
        l.append(-np.inf if lo is None else lo)
        u.append(np.inf if hi is None else hi)
    if groups:
        for idx, lo, hi in groups.values():
            idx = np.asarray(idx, dtype=int)
            e = np.zeros(n)
            e[idx] = 1.0
            rows.append(e)
            l.append(-np.inf if lo is None else lo)
            u.append(np.inf if hi is None else hi)
    if A_extra is not None:
        A_extra = np.asarray(A_extra, dtype=float)
        if A_extra.ndim == 1:
            A_extra = A_extra.reshape(1, -1)
        for r, lo, hi in zip(A_extra, l_extra, u_extra):
            rows.append(r)
            l.append(lo)
            u.append(hi)
    A = sp.csc_matrix(np.vstack(rows))
    l = np.asarray(l, dtype=float)
    u = np.asarray(u, dtype=float)
    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, verbose=False, eps_abs=1e-8, eps_rel=1e-8)
    res = prob.solve()
    if res.info.status != "solved":
        raise _OSQPUnavailable(f"OSQP {name}: {res.info.status}")
    return np.asarray(res.x).ravel()


def gmv_osqp(mu: pd.Series, cov: pd.DataFrame, bounds=(0.0, 1.0), groups=None,
             gamma: float = 0.0) -> np.ndarray:
    n = len(mu)
    C = np.asarray(cov.values, dtype=float) + gamma * np.eye(n)
    return _solve_qp_osqp(2.0 * C, np.zeros(n), n, bounds, groups, name="gmv")


def min_vol_at_return_osqp(mu: pd.Series, cov: pd.DataFrame, target: float,
                           bounds=(0.0, 1.0), groups=None) -> np.ndarray:
    n = len(mu)
    C = np.asarray(cov.values, dtype=float)
    m = np.asarray(mu.values, dtype=float)
    return _solve_qp_osqp(2.0 * C, np.zeros(n), n, bounds, groups,
                          A_extra=m, l_extra=[target], u_extra=[target],
                          name="min_vol_at_return")


def max_sharpe_osqp(mu: pd.Series, cov: pd.DataFrame, rf: float,
                    bounds=(0.0, 1.0)) -> np.ndarray:
    """Max Sharpe via min y'Cy s.t. (mu-rf)'y = 1, y >= 0; w = y/sum(y).

    Long-only only (same exactness conditions as the Gurobi reformulation).
    """
    blist = _as_bounds_list(len(mu), bounds)
    if any(lo != 0 for lo, _ in blist) or any(
            hi is not None and hi < 1.0 for _, hi in blist):
        raise _OSQPUnavailable("max-Sharpe OSQP path needs long-only bounds")
    n = len(mu)
    C = np.asarray(cov.values, dtype=float)
    e = np.asarray(mu.values, dtype=float) - rf
    y = _solve_qp_osqp(2.0 * C, np.zeros(n), n, (0.0, None), None,
                       A_extra=e, l_extra=[1.0], u_extra=[1.0],
                       name="max_sharpe", fully_invested=False)
    s = y.sum()
    if s <= 0:
        raise _OSQPUnavailable("degenerate max-Sharpe solution")
    return y / s
