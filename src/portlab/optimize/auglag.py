"""Augmented Lagrangian method (auglag) for large-scale mean-variance QPs.

Renamed from ``portlab.optimize.alm`` (2026-10-09) so the name no longer
collides with ``portlab.alm`` (asset-liability modeling). The old module path
and ``solver="alm"`` still work via deprecated aliases.

Solves  min y'Cy  s.t.  A_eq @ y = b_eq, y >= 0
via the augmented Lagrangian
    L(y; lam, rho) = y'Cy + lam'(A_eq@y - b_eq) + (rho/2)*||A_eq@y - b_eq||^2,
    y >= 0,
with bound-constrained L-BFGS-B inner solves (warm-started) and the
multiplier update lam <- lam + rho*(A_eq@y - b_eq).

Why this is fast: the inner problems are bounds-only, which L-BFGS-B
(a mature Fortran implementation) solves far more efficiently than a
hand-rolled first-order method handles equality constraints. The outer
loop typically needs 3-6 iterations; warm starts make later inners cheap.
No license, no size limit. Used for max Sharpe (via the y-reformulation),
GMV, and min-vol-at-return.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np


class _AugLagError(Exception):
    """Raised when auglag cannot be used."""


def _cg_solve(matvec, b: np.ndarray, rtol: float = 1e-8,
              maxiter: int = 1000) -> np.ndarray:
    """Conjugate gradient for PD systems (matrix-free). Returns x ~= A^{-1}b."""
    from scipy.sparse.linalg import LinearOperator, cg

    n = b.size
    A = LinearOperator((n, n), matvec=matvec, dtype=float)
    x, info = cg(A, b, rtol=rtol, maxiter=maxiter)
    if info != 0:
        raise _AugLagError(f"CG did not converge (info={info})")
    return np.asarray(x).ravel()


def auglag_qp(C: np.ndarray, A_eq: np.ndarray, b_eq: np.ndarray,
           y0: np.ndarray | None = None, tol: float = 1e-9,
           max_outer: int = 30, c: np.ndarray | None = None) -> np.ndarray:
    """min y'Cy + c'y s.t. A_eq @ y = b_eq, y >= 0. Returns y (numpy).

    ``c`` is an optional linear term (default zero — previous behavior).
    """
    from scipy.optimize import minimize

    C = np.asarray(C, dtype=float)
    A_eq = np.asarray(A_eq, dtype=float)
    if A_eq.ndim == 1:
        A_eq = A_eq.reshape(1, -1)
    b_eq = np.asarray(b_eq, dtype=float).ravel()
    n = C.shape[0]
    m = A_eq.shape[0]
    cv = np.zeros(n) if c is None else np.asarray(c, dtype=float).ravel()
    if cv.shape != (n,):
        raise ValueError(f"c must have shape ({n},), got {cv.shape}")

    y = np.full(n, 1.0 / n) if y0 is None else np.asarray(y0, dtype=float).copy()
    y = np.maximum(y, 0.0)
    lam = np.zeros(m)
    rho = 1.0
    for _ in range(max_outer):
        def fg(yy, lam=lam, rho=rho):
            Cy = C @ yy
            r = A_eq @ yy - b_eq
            f = float(yy @ Cy) + float(cv @ yy) + float(lam @ r) \
                + 0.5 * rho * float(r @ r)
            g = 2.0 * Cy + cv + A_eq.T @ lam + rho * (A_eq.T @ r)
            return f, g

        res = minimize(lambda yy: fg(yy)[0], y, jac=lambda yy: fg(yy)[1],
                       method="L-BFGS-B", bounds=[(0.0, None)] * n,
                       options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-9})
        y = np.asarray(res.x).ravel()
        viol = np.max(np.abs(A_eq @ y - b_eq))
        if viol < tol:
            break
        lam = lam + rho * (A_eq @ y - b_eq)
        rho = min(rho * 5.0, 1e8)
    else:
        raise _AugLagError("auglag did not converge")
    return y


def max_sharpe_auglag(mu: np.ndarray, C: np.ndarray, rf: float,
                   tol: float = 1e-9) -> np.ndarray:
    """Max Sharpe via auglag on min y'Cy s.t. (mu-rf)'y = 1, y >= 0; w = y/sum(y).

    Long-only only. Uses a CG warm start from the unconstrained problem.
    """
    mu = np.asarray(mu, dtype=float)
    C = np.asarray(C, dtype=float)
    e = mu - rf
    if e.max() <= 0:
        raise _AugLagError("max Sharpe infeasible: no asset beats rf")
    n = mu.size

    def matvec(v):
        return C @ v

    # Warm start: unconstrained min y'Cy s.t. e'y = 1 -> y = C^{-1}e/(e'C^{-1}e)
    try:
        v = _cg_solve(matvec, e)
        denom = float(e @ v)
        if denom <= 0:
            raise _AugLagError("CG warm start degenerate")
        y0 = np.maximum(v / denom, 0.0)
        # Project onto {e'y = 1, y >= 0}
        from .fista import _project_weighted_simplex
        y0 = _project_weighted_simplex(y0, e, 1.0)
    except _AugLagError:
        y0 = np.full(n, 1.0 / n)

    y = auglag_qp(C, e.reshape(1, -1), np.array([1.0]), y0=y0, tol=tol)
    s = y.sum()
    if s <= 0:
        raise _AugLagError("degenerate max-Sharpe solution")
    return y / s


def gmv_auglag(C: np.ndarray, gamma: float = 0.0, tol: float = 1e-9) -> np.ndarray:
    """min w'(C + gamma I)w s.t. sum w = 1, w >= 0."""
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    Q = C + gamma * np.eye(n)
    y = auglag_qp(Q, np.ones((1, n)), np.array([1.0]), tol=tol)
    s = y.sum()
    return y / s if s > 0 else y


def min_vol_at_return_auglag(mu: np.ndarray, C: np.ndarray, target: float,
                          tol: float = 1e-9) -> np.ndarray:
    """min w'Cw s.t. w'mu = target, sum w = 1, w >= 0."""
    mu = np.asarray(mu, dtype=float)
    C = np.asarray(C, dtype=float)
    n = mu.size
    A_eq = np.vstack([np.ones(n), mu])
    y = auglag_qp(C, A_eq, np.array([1.0, target]), tol=tol)
    s = y.sum()
    return y / s if s > 0 else y


def _deprecated_alias(new_func, old_name: str):
    """Wrap ``new_func`` so calls via the old name warn (once per call site)."""
    @functools.wraps(new_func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{old_name} is deprecated; use {new_func.__name__} instead.",
            DeprecationWarning, stacklevel=2,
        )
        return new_func(*args, **kwargs)
    return wrapper


# Deprecated aliases for the pre-rename (optimize.alm) API.
alm_qp = _deprecated_alias(auglag_qp, "alm_qp")
max_sharpe_alm = _deprecated_alias(max_sharpe_auglag, "max_sharpe_alm")
gmv_alm = _deprecated_alias(gmv_auglag, "gmv_alm")
min_vol_at_return_alm = _deprecated_alias(min_vol_at_return_auglag,
                                         "min_vol_at_return_alm")
_ALMError = _AugLagError  # private name: plain alias, no warning needed
