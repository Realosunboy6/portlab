"""Mean-variance optimization: GMV, max Sharpe, targets, efficient frontier.

solver="auto" (default) tries backends in order — Gurobi, auglag, FISTA, OSQP —
falling back to SLSQP when none is usable. solver="gurobi"/"auglag"/"fista"/
"osqp" force one backend (raising if unavailable); solver="slsqp" forces the
legacy path (reproducibility). solver="alm" is a deprecated alias of "auglag".
All backends solve the same convex problems; they differ in speed and
licensing, not in the answer.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..config import DEFAULT_RF
from .auglag import (_AugLagError, gmv_auglag, max_sharpe_auglag,
                     min_vol_at_return_auglag)
from .core import portfolio_stats, solve_weights
from .fista import _FistaError, gmv_fista, max_sharpe_fista, min_vol_at_return_fista
from .gurobi_backend import (
    _GurobiUnavailable,
    gmv_qp,
    max_return_at_vol_qp,
    max_sharpe_qp,
    min_vol_at_return_qp,
    try_gurobi,
)
from .osqp_backend import (
    _OSQPUnavailable,
    gmv_osqp,
    max_sharpe_osqp,
    min_vol_at_return_osqp,
)

_BACKENDS = ("gurobi", "auglag", "fista", "osqp", "slsqp")

#: Deprecated solver-name aliases, resolved with a DeprecationWarning.
_DEPRECATED_SOLVER_ALIASES = {"alm": "auglag"}


def _as_series(w: np.ndarray, index: pd.Index) -> pd.Series:
    w = np.asarray(w, dtype=float).ravel()
    if abs(w.sum()) > 1e-12:
        w = w / w.sum()
    return pd.Series(w, index=index, name="weight")


def _try_backend(name, label, fn):
    """Run one backend; return None (with a warning unless merely missing)
    when it cannot be used."""
    try:
        return fn()
    except (_GurobiUnavailable, _FistaError, _OSQPUnavailable, _AugLagError) as e:
        msg = str(e)
        if "not installed" not in msg:
            warnings.warn(f"{name} {label} unavailable ({msg}); trying next.",
                          RuntimeWarning, stacklevel=4)
        return None


def _solve_with_fallback(label, backend_fns: dict, slsqp_fn, solver, index):
    """backend_fns: {backend_name: callable} tried in _BACKENDS order."""
    if solver in _DEPRECATED_SOLVER_ALIASES:
        new = _DEPRECATED_SOLVER_ALIASES[solver]
        warnings.warn(f"solver={solver!r} is deprecated; use solver={new!r} instead.",
                      DeprecationWarning, stacklevel=3)
        solver = new
    if solver == "slsqp":
        return slsqp_fn()
    if solver != "auto" and solver not in _BACKENDS:
        raise ValueError(f"solver must be one of {_BACKENDS} or 'auto'")
    order = _BACKENDS if solver == "auto" else [solver]
    for name in order:
        if name == "slsqp":
            return slsqp_fn()
        w = _try_backend(name, label, backend_fns[name])
        if w is not None:
            return _as_series(w, index)
        if solver != "auto":
            raise RuntimeError(
                f"solver='{solver}' requested but the {solver} backend "
                f"could not solve {label}.")
    return slsqp_fn()  # unreachable, kept for clarity


def _is_long_only(n: int, bounds) -> bool:
    blist = ([bounds] * n if isinstance(bounds, tuple)
             and len(bounds) == 2 and np.isscalar(bounds[0])
             else list(bounds))
    return all(lo == 0 and (hi is None or hi >= 1.0) for lo, hi in blist)


def gmv(mu: pd.Series, cov: pd.DataFrame, bounds=(0.0, 1.0), groups=None,
        gamma: float = 0.0, solver: str = "auto") -> pd.Series:
    """Global minimum-variance portfolio. gamma adds L2 regularization
    (gamma * ||w||^2) to spread weights across more assets."""
    n = len(mu)

    def slsqp():
        C = cov.values

        def obj(w):
            return w @ C @ w + gamma * (w @ w)

        def jac(w):
            return 2.0 * (C @ w) + 2.0 * gamma * w

        return solve_weights(obj, n, bounds, groups, index=mu.index, jac=jac)

    def auglag():
        if groups is not None or not _is_long_only(n, bounds):
            raise _AugLagError("auglag GMV needs long-only bounds, no groups")
        return gmv_auglag(cov.values, gamma)

    def fista():
        if groups is not None or not _is_long_only(n, bounds):
            raise _FistaError("FISTA GMV needs long-only bounds, no groups")
        return gmv_fista(cov.values, gamma)

    return _solve_with_fallback(
        "gmv",
        {"gurobi": lambda: gmv_qp(mu, cov, bounds, groups, gamma),
         "auglag": auglag,
         "fista": fista,
         "osqp": lambda: gmv_osqp(mu, cov, bounds, groups, gamma)},
        slsqp, solver, mu.index)


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, rf: float = DEFAULT_RF,
               bounds=(0.0, 1.0), groups=None, gamma: float = 0.0,
               solver: str = "auto") -> pd.Series:
    m, C = mu.values, cov.values
    e = m - rf
    n = len(mu)

    def slsqp():
        def neg_sharpe(w):
            v = w @ C @ w
            base = -(w @ e) / np.sqrt(v) if v > 0 else 0.0
            return base + gamma * (w @ w)

        def jac(w):
            # d/dw [ -(e'w)/sqrt(w'Cw) ] = -(e*v - (e'w)*Cw) / v^1.5
            v = w @ C @ w
            if v <= 0:
                g = np.zeros_like(w)
            else:
                s = w @ e
                g = -(e * v - s * (C @ w)) / (v ** 1.5)
            return g + 2.0 * gamma * w

        return solve_weights(neg_sharpe, n, bounds, groups,
                             index=mu.index, jac=jac)

    def _long_only():
        return _is_long_only(n, bounds)

    def gurobi():
        if gamma != 0 or groups is not None:
            raise _GurobiUnavailable(
                "max-Sharpe Gurobi path needs gamma=0 and no groups")
        return max_sharpe_qp(mu, cov, rf, bounds)

    def auglag():
        if gamma != 0 or groups is not None or not _long_only():
            raise _AugLagError("auglag max-Sharpe needs long-only, gamma=0, no groups")
        return max_sharpe_auglag(m, C, rf)

    def fista():
        if gamma != 0 or groups is not None or not _long_only():
            raise _FistaError("FISTA max-Sharpe needs long-only, gamma=0")
        return max_sharpe_fista(m, C, rf)

    def osqp():
        if gamma != 0 or groups is not None:
            raise _OSQPUnavailable(
                "max-Sharpe OSQP path needs gamma=0 and no groups")
        return max_sharpe_osqp(mu, cov, rf, bounds)

    return _solve_with_fallback(
        "max_sharpe",
        {"gurobi": gurobi, "auglag": auglag, "fista": fista, "osqp": osqp},
        slsqp, solver, mu.index)


def min_vol_at_return(mu: pd.Series, cov: pd.DataFrame, target_return: float,
                      bounds=(0.0, 1.0), groups=None,
                      solver: str = "auto") -> pd.Series:
    n = len(mu)

    def slsqp():
        C = cov.values
        extra = [{"type": "eq", "fun": lambda w: w @ mu.values - target_return}]
        return solve_weights(lambda w: w @ C @ w, n, bounds, groups,
                             extra_constraints=extra, index=mu.index,
                             jac=lambda w: 2.0 * (C @ w))

    def fista():
        if groups is not None or not _is_long_only(n, bounds):
            raise _FistaError("FISTA min-vol needs long-only bounds, no groups")
        return min_vol_at_return_fista(mu.values, cov.values, target_return)

    def auglag():
        if groups is not None or not _is_long_only(n, bounds):
            raise _AugLagError("auglag min-vol needs long-only bounds, no groups")
        return min_vol_at_return_auglag(mu.values, cov.values, target_return)

    return _solve_with_fallback(
        "min_vol_at_return",
        {"gurobi": lambda: min_vol_at_return_qp(mu, cov, target_return, bounds, groups),
         "auglag": auglag,
         "fista": fista,
         "osqp": lambda: min_vol_at_return_osqp(mu, cov, target_return, bounds, groups)},
        slsqp, solver, mu.index)


def max_return_at_vol(mu: pd.Series, cov: pd.DataFrame, target_vol: float,
                      bounds=(0.0, 1.0), groups=None,
                      solver: str = "auto") -> pd.Series:
    n = len(mu)

    def slsqp():
        C = cov.values
        mv = mu.values
        extra = [{"type": "ineq",
                  "fun": lambda w: target_vol ** 2 - w @ C @ w}]
        return solve_weights(lambda w: -(w @ mv), n, bounds, groups,
                             extra_constraints=extra, index=mu.index,
                             jac=lambda w: -mv)

    def via_bisection(qp_solver):
        # max return at vol t  <=>  frontier point with vol == t.
        # Bisect the target return: vol(min_vol(tr)) <= t  <=>  tr <= tr*.
        w_g = gmv(mu, cov, bounds, groups, solver=qp_solver)
        r_lo = float(w_g @ mu.values)
        if float(np.sqrt(w_g.values @ cov.values @ w_g.values)) > target_vol:
            raise _FistaError("target vol below GMV vol: infeasible")
        lo, hi, w_best = r_lo, float(mu.max()), w_g
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            w = min_vol_at_return(mu, cov, mid, bounds, groups, solver=qp_solver)
            if float(np.sqrt(w.values @ cov.values @ w.values)) <= target_vol:
                lo, w_best = mid, w
            else:
                hi = mid
        return w_best.values

    def gurobi():
        return max_return_at_vol_qp(mu, cov, target_vol, bounds, groups)

    def auglag():
        return via_bisection("auglag")

    def fista():
        return via_bisection("fista")

    def osqp():
        return via_bisection("osqp")

    return _solve_with_fallback(
        "max_return_at_vol",
        {"gurobi": gurobi, "auglag": auglag, "fista": fista, "osqp": osqp},
        slsqp, solver, mu.index)


def frontier(mu: pd.Series, cov: pd.DataFrame, n_points: int = 30,
             rf: float = DEFAULT_RF, bounds=(0.0, 1.0), groups=None,
             solver: str = "auto") -> pd.DataFrame:
    """Efficient frontier: one row per target return with vol/sharpe/weights."""
    w_gmv = gmv(mu, cov, bounds, groups, solver=solver)
    r_lo = float(w_gmv @ mu)
    r_hi = float(mu.max())
    rows = []
    for tr in np.linspace(r_lo, r_hi, n_points):
        try:
            w = min_vol_at_return(mu, cov, tr, bounds, groups, solver=solver)
        except Exception:
            continue
        stats = portfolio_stats(w.values, mu, cov, rf)
        rows.append({"target_return": tr, **stats,
                     **{f"w_{a}": w[a] for a in mu.index}})
    return pd.DataFrame(rows)
