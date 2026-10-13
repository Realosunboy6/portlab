"""Shared plumbing for the optimizers: constraint building, generic solver, stats."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..config import DEFAULT_RF, TRADING_DAYS


def mean_cov(rets: pd.DataFrame, periods: int = TRADING_DAYS,
             cov_method: str = "ledoit_wolf") -> tuple[pd.Series, pd.DataFrame]:
    """Annualized mean vector (arithmetic) and covariance matrix from returns."""
    from ..covariance import get_cov
    mu = rets.mean() * periods
    cov = get_cov(rets, method=cov_method, annualize=periods)
    return mu, cov


def portfolio_stats(w: np.ndarray, mu: pd.Series, cov: pd.DataFrame,
                    rf: float = DEFAULT_RF) -> dict[str, float]:
    ret = float(w @ mu.values)
    vol = float(np.sqrt(w @ cov.values @ w))
    return {"return": ret, "volatility": vol,
            "sharpe": (ret - rf) / vol if vol > 0 else 0.0}


def build_constraints(
    n: int,
    bounds: tuple[float, float] | Sequence[tuple[float, float]] = (0.0, 1.0),
    groups: Mapping[str, tuple[Sequence[int], float, float]] | None = None,
    extra: list[dict] | None = None,
):
    """Standard constraint set: full investment, per-asset bounds, group bounds.

    groups: {name: (indices, min_weight, max_weight)} e.g. sector caps.
    Returns (bounds_list, constraints_list) for scipy SLSQP.
    """
    if isinstance(bounds, tuple) and len(bounds) == 2 and np.isscalar(bounds[0]):
        bounds_list = [bounds] * n
    else:
        bounds_list = list(bounds)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    if groups:
        for idx, lo, hi in groups.values():
            idx = np.asarray(idx)
            cons.append({"type": "ineq", "fun": lambda w, i=idx, m=lo: w[i].sum() - m})
            cons.append({"type": "ineq", "fun": lambda w, i=idx, h=hi: h - w[i].sum()})
    if extra:
        cons.extend(extra)
    return bounds_list, cons


def solve_weights(
    objective: Callable[[np.ndarray], float],
    n: int,
    bounds=(0.0, 1.0),
    groups=None,
    extra_constraints=None,
    x0: np.ndarray | None = None,
    index: pd.Index | None = None,
    jac: Callable[[np.ndarray], np.ndarray] | None = None,
) -> pd.Series:
    """Minimize `objective` over the simplex-with-bounds. Returns weights.

    jac: analytic gradient of `objective`. Strongly recommended for n > ~200:
    SLSQP's numerical gradient needs n+1 function evaluations per iteration.
    """
    bounds_list, cons = build_constraints(n, bounds, groups, extra_constraints)
    if x0 is None:
        x0 = np.full(n, 1.0 / n)
    res = minimize(objective, x0, method="SLSQP", jac=jac, bounds=bounds_list,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
    if not res.success:
        # Retry from a random feasible start before giving up.
        rng = np.random.default_rng(0)
        x0b = rng.dirichlet(np.ones(n))
        res2 = minimize(objective, x0b, method="SLSQP", jac=jac, bounds=bounds_list,
                        constraints=cons, options={"maxiter": 800, "ftol": 1e-10})
        if res2.success or res2.fun < res.fun:
            res = res2
    w = res.x
    if abs(w.sum()) > 1e-12:
        w = w / w.sum()
    return pd.Series(w, index=index if index is not None else range(n), name="weight")
