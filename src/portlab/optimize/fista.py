"""FISTA (accelerated projected gradient) for large-scale mean-variance QPs.

First-order method: O(n^2) per iteration (one covariance matvec), no matrix
factorization, no license, no size limit. Strongly convex here (Ledoit-Wolf
covariance is PD), and adaptive restart keeps it linearly convergent in
practice — seconds at n=5000 where SLSQP never finishes.

Solves  min  y'Qy + c'y   s.t.  y in C
with C one of:
  - weighted simplex {e'y = target, y >= 0}          (max Sharpe, y-form)
  - simplex          {sum y = 1, y >= 0}             (GMV)
  - {mu'y = target, sum y = 1, y >= 0}               (min vol at return)
"""

from __future__ import annotations

import numpy as np


class _FistaError(Exception):
    """Raised when FISTA cannot be used (infeasible / no convergence)."""


# --------------------------------------------------------------------------
# projections
# --------------------------------------------------------------------------

def _project_weighted_simplex(z: np.ndarray, e: np.ndarray, target: float = 1.0,
                              tol: float = 1e-13) -> np.ndarray:
    """Projection onto {y : e'y = target, y >= 0}.

    KKT: y_i = max(z_i - lam*e_i, 0); h(lam) = e'max(z - lam*e, 0) - target
    is continuous and strictly decreasing, so one scalar bisection finds lam.
    """
    z = np.asarray(z, dtype=float)
    e = np.asarray(e, dtype=float)
    if e.max() <= 0 and target > 0:
        raise _FistaError("weighted-simplex projection infeasible: e <= 0")

    def h(lam: float) -> float:
        return float(e @ np.maximum(z - lam * e, 0.0)) - target

    lo, hi = -1.0, 1.0
    for _ in range(300):
        if h(lo) > 0:
            break
        lo *= 2.0
    else:
        raise _FistaError("weighted-simplex projection infeasible (lo)")
    for _ in range(300):
        if h(hi) < 0:
            break
        hi *= 2.0
    else:
        raise _FistaError("weighted-simplex projection infeasible (hi)")
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if h(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, abs(lo) + abs(hi)):
            break
    return np.maximum(z - 0.5 * (lo + hi) * e, 0.0)


def _project_simplex(z: np.ndarray) -> np.ndarray:
    """Projection onto {y : sum y = 1, y >= 0} (Duchi et al., O(n log n))."""
    z = np.asarray(z, dtype=float)
    u = np.sort(z)[::-1]
    css = np.cumsum(u) - 1.0
    rho = np.nonzero(u > css / (np.arange(z.size) + 1))[0][-1]
    theta = css[rho] / (rho + 1)
    return np.maximum(z - theta, 0.0)


def _project_ret_simplex(z: np.ndarray, mu: np.ndarray, target: float) -> np.ndarray:
    """Projection onto {mu'y = target, sum y = 1, y >= 0}.

    y_i(lam) = simplex-project(z - lam*mu); G(lam) = mu'y(lam) - target
    is decreasing in lam, so an outer bisection on lam works.
    """
    z = np.asarray(z, dtype=float)
    mu = np.asarray(mu, dtype=float)
    mu_min, mu_max = float(mu.min()), float(mu.max())
    if not (mu_min - 1e-12 <= target <= mu_max + 1e-12):
        raise _FistaError("return-simplex projection infeasible: "
                          f"target {target} outside [{mu_min}, {mu_max}]")

    def y_of(lam: float) -> np.ndarray:
        return _project_simplex(z - lam * mu)

    def G(lam: float) -> float:
        return float(mu @ y_of(lam)) - target

    lo, hi = -1.0, 1.0
    for _ in range(300):
        if G(lo) > 0:
            break
        lo *= 2.0
    else:
        raise _FistaError("return-simplex projection infeasible (lo)")
    for _ in range(300):
        if G(hi) < 0:
            break
        hi *= 2.0
    else:
        raise _FistaError("return-simplex projection infeasible (hi)")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if G(mid) > 0:
            lo = mid
        else:
            hi = mid
    return y_of(0.5 * (lo + hi))


# --------------------------------------------------------------------------
# FISTA core
# --------------------------------------------------------------------------

def _lipschitz(matvec, n: int, iters: int = 40) -> float:
    """Largest eigenvalue of the symmetric PSD operator via power iteration."""
    rng = np.random.default_rng(0)
    b = rng.standard_normal(n)
    b /= np.linalg.norm(b)
    for _ in range(iters):
        b = matvec(b)
        nb = np.linalg.norm(b)
        if nb == 0:
            return 0.0
        b /= nb
    return float(b @ matvec(b))


def fista(grad, L: float, proj, y0: np.ndarray, tol: float = 1e-7,
          max_iter: int = 20000) -> np.ndarray:
    """Minimize smooth convex f (grad L-Lipschitz) over a convex set.

    FISTA with adaptive restart (O'Donoghue & Candes): the gradient-mapping
    restart kills momentum oscillations, giving linear convergence on the
    strongly convex QPs here. grad(y) -> gradient; proj(z) -> projection.
    """
    y_prev = np.asarray(y0, dtype=float).copy()
    z = y_prev.copy()
    t = 1.0
    for _ in range(max_iter):
        y = proj(z - grad(z) / L)
        # restart when the momentum points uphill (gradient at new point)
        if np.dot(grad(y), y - y_prev) > 0:
            t = 1.0
            z = y.copy()
        else:
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            z = y + ((t - 1.0) / t_next) * (y - y_prev)
            t = t_next
        if np.max(np.abs(y - y_prev)) <= tol * (1.0 + np.max(np.abs(y))):
            return y
        y_prev = y
    raise _FistaError(f"FISTA did not converge in {max_iter} iterations")


def _solve_qp_fista(Q: np.ndarray, c: np.ndarray, proj, y0: np.ndarray,
                    tol: float = 1e-7) -> np.ndarray:
    """min y'Qy + c'y over the projected set. Q must be symmetric PSD."""
    Q = np.asarray(Q, dtype=float)
    c = np.asarray(c, dtype=float)
    n = Q.shape[0]

    def matvec(v):
        return Q @ v

    L = 2.0 * _lipschitz(matvec, n) * 1.05 + 1e-12

    def grad(y):
        return 2.0 * (Q @ y) + c

    return fista(grad, L, proj, y0, tol=tol)


# --------------------------------------------------------------------------
# public entry points (return weight vectors, numpy)
# --------------------------------------------------------------------------

def max_sharpe_fista(mu: np.ndarray, C: np.ndarray, rf: float,
                     tol: float = 1e-7) -> np.ndarray:
    """Max Sharpe via FISTA on min y'Cy s.t. (mu-rf)'y = 1, y >= 0; w = y/sum(y).

    Long-only only (matches the Gurobi reformulation's exactness conditions).
    """
    mu = np.asarray(mu, dtype=float)
    C = np.asarray(C, dtype=float)
    e = mu - rf
    if e.max() <= 0:
        raise _FistaError("max Sharpe infeasible: no asset beats rf")
    n = mu.size
    y0 = np.full(n, 1.0 / n)
    y = _solve_qp_fista(C, np.zeros(n),
                        lambda z: _project_weighted_simplex(z, e, 1.0), y0, tol)
    s = y.sum()
    if s <= 0:
        raise _FistaError("degenerate max-Sharpe solution")
    return y / s


def gmv_fista(C: np.ndarray, gamma: float = 0.0, tol: float = 1e-7) -> np.ndarray:
    """min w'(C + gamma I)w s.t. sum w = 1, w >= 0."""
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    Q = C + gamma * np.eye(n)
    return _solve_qp_fista(Q, np.zeros(n), _project_simplex,
                           np.full(n, 1.0 / n), tol)


def min_vol_at_return_fista(mu: np.ndarray, C: np.ndarray, target: float,
                            tol: float = 1e-7) -> np.ndarray:
    """min w'Cw s.t. w'mu = target, sum w = 1, w >= 0."""
    mu = np.asarray(mu, dtype=float)
    C = np.asarray(C, dtype=float)
    n = mu.size
    return _solve_qp_fista(C, np.zeros(n),
                           lambda z: _project_ret_simplex(z, mu, target),
                           np.full(n, 1.0 / n), tol)
