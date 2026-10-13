"""Gurobi backend for mean-variance optimization.

Same convex problems the SLSQP path solves, via gurobipy — exact, not a
heuristic, and much faster once n passes a few hundred (SLSQP uses numerical
gradients). Every function here falls back to SLSQP when Gurobi is missing,
unlicensed, or refuses the model size (the pip-bundled license is
size-limited; a full license such as the user's course license removes the
cap). Nothing changes mathematically: Gurobi certifies the global optimum of
the same problem.

Conventions (match core.build_constraints):
  bounds: (lo, hi) applied to every asset, or a per-asset sequence.
  groups: {name: (indices, min_weight, max_weight)}.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


class _GurobiUnavailable(Exception):
    """Raised when the Gurobi path cannot be used; caller falls back to SLSQP."""


def _import_gurobi():
    try:
        import gurobipy as gp
        return gp
    except ImportError as e:
        raise _GurobiUnavailable("gurobipy is not installed") from e


def gurobi_available() -> bool:
    """True if gurobipy imports (license/size checked at solve time)."""
    return _import_gurobi() is not None


def _as_bounds_list(n: int, bounds) -> list[tuple[float, float]]:
    if isinstance(bounds, tuple) and len(bounds) == 2 and np.isscalar(bounds[0]):
        return [bounds] * n
    return list(bounds)


def _new_model(gp, name: str):
    m = gp.Model(name)
    m.setParam("OutputFlag", 0)
    return m


def _apply_box_and_groups(m, w, n: int, bounds, groups) -> None:
    blist = _as_bounds_list(n, bounds)
    for i, (lo, hi) in enumerate(blist):
        if lo is not None:
            m.addConstr(w[i] >= lo)
        if hi is not None:
            m.addConstr(w[i] <= hi)
    if groups:
        for idx, lo, hi in groups.values():
            idx = np.asarray(idx, dtype=int)
            s = w[idx].sum()
            if lo is not None:
                m.addConstr(s >= lo)
            if hi is not None:
                m.addConstr(s <= hi)


def _check_optimal(gp, m, what: str):
    if m.status != gp.GRB.OPTIMAL:
        raise _GurobiUnavailable(f"Gurobi did not certify optimality for {what} "
                                 f"(status={m.status})")


def _solve_qp(Q: np.ndarray, c: np.ndarray, n: int, bounds, groups,
              A_eq: np.ndarray | None = None, b_eq: np.ndarray | None = None,
              quad_constr: tuple[np.ndarray, float] | None = None,
              name: str = "qp") -> np.ndarray:
    """min w'Qw + c'w  s.t. sum w = 1, box/groups, optional A_eq w = b_eq,
    optional convex quadratic constraint w'Qq w <= rhs."""
    gp = _import_gurobi()
    m = _new_model(gp, name)
    try:
        w = m.addMVar(n, lb=-gp.GRB.INFINITY, name="w")
        Q = np.asarray(Q, dtype=float)
        m.setObjective(w @ Q @ w + c @ w, gp.GRB.MINIMIZE)
        m.addConstr(w.sum() == 1.0)
        _apply_box_and_groups(m, w, n, bounds, groups)
        if A_eq is not None:
            A_eq = np.asarray(A_eq, dtype=float)
            b_eq = np.asarray(b_eq, dtype=float).ravel()
            m.addConstr(A_eq @ w == b_eq)
        if quad_constr is not None:
            Qq, rhs = quad_constr
            # NB: addConstr (not addQConstr) dispatches MQuadExpr comparisons
            # correctly in gurobipy 13.
            m.addConstr(w @ np.asarray(Qq, dtype=float) @ w <= rhs)
        m.optimize()
        _check_optimal(gp, m, name)
        return np.asarray(w.X).ravel()
    except gp.GurobiError as e:
        raise _GurobiUnavailable(str(e)) from e
    finally:
        m.dispose()


def gmv_qp(mu: pd.Series, cov: pd.DataFrame, bounds=(0.0, 1.0), groups=None,
           gamma: float = 0.0) -> np.ndarray:
    """Global minimum variance via QP: min w'(C + gamma I)w, sum w = 1."""
    n = len(mu)
    C = np.asarray(cov.values, dtype=float) + gamma * np.eye(n)
    return _solve_qp(C, np.zeros(n), n, bounds, groups, name="gmv")


def min_vol_at_return_qp(mu: pd.Series, cov: pd.DataFrame, target_return: float,
                         bounds=(0.0, 1.0), groups=None) -> np.ndarray:
    """min w'Cw s.t. w'mu = target, sum w = 1."""
    n = len(mu)
    C = np.asarray(cov.values, dtype=float)
    return _solve_qp(C, np.zeros(n), n, bounds, groups,
                     A_eq=np.asarray(mu.values, dtype=float).reshape(1, -1),
                     b_eq=np.array([target_return]), name="min_vol_at_return")


def max_return_at_vol_qp(mu: pd.Series, cov: pd.DataFrame, target_vol: float,
                         bounds=(0.0, 1.0), groups=None) -> np.ndarray:
    """max w'mu s.t. w'Cw <= target_vol^2, sum w = 1 (convex QCQP)."""
    n = len(mu)
    C = np.asarray(cov.values, dtype=float)
    return _solve_qp(np.zeros((n, n)), -np.asarray(mu.values, dtype=float),
                     n, bounds, groups,
                     quad_constr=(C, target_vol ** 2), name="max_return_at_vol")


def max_sharpe_qp(mu: pd.Series, cov: pd.DataFrame, rf: float,
                  bounds=(0.0, 1.0)) -> np.ndarray:
    """Max Sharpe via the Cornuejols-Tütüncü reformulation.

    min y'Cy s.t. (mu - rf)'y = 1, y >= 0  →  w = y / sum(y).

    Exact only for the long-only case (lb == 0, ub >= 1 or None, no groups):
    a per-asset cap ub < 1 becomes y_i <= ub * sum(y), which is bilinear.
    Raises _GurobiUnavailable for anything else so the caller falls back.
    """
    blist = _as_bounds_list(len(mu), bounds)
    if any(lo != 0 for lo, _ in blist):
        raise _GurobiUnavailable("max-Sharpe Gurobi path needs lb == 0")
    if any(hi is not None and hi < 1.0 for _, hi in blist):
        raise _GurobiUnavailable("max-Sharpe Gurobi path needs ub >= 1 "
                                 "(cap is redundant under sum w = 1, w >= 0)")
    gp = _import_gurobi()
    m = _new_model(gp, "max_sharpe")
    try:
        n = len(mu)
        y = m.addMVar(n, lb=0.0, name="y")
        C = np.asarray(cov.values, dtype=float)
        excess = np.asarray(mu.values, dtype=float) - rf
        m.setObjective(y @ C @ y, gp.GRB.MINIMIZE)
        m.addConstr(excess @ y == 1.0)
        m.optimize()
        _check_optimal(gp, m, "max_sharpe")
        yv = np.asarray(y.X).ravel()
        s = yv.sum()
        if s <= 0:
            raise _GurobiUnavailable("degenerate max-Sharpe solution")
        return yv / s
    except gp.GurobiError as e:
        raise _GurobiUnavailable(str(e)) from e
    finally:
        m.dispose()


def try_gurobi(label: str, fn, *args, **kwargs):
    """Run fn(*args, **kwargs); on _GurobiUnavailable warn once and return None
    so the caller can fall back to SLSQP. Import errors are silent (no Gurobi
    installed is a normal configuration); solve-time failures warn."""
    try:
        return fn(*args, **kwargs)
    except _GurobiUnavailable as e:
        msg = str(e)
        if "not installed" not in msg:
            warnings.warn(f"Gurobi {label} unavailable ({msg}); "
                          f"falling back to SLSQP.", RuntimeWarning, stacklevel=3)
        return None
