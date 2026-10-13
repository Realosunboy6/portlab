"""Robust mean-variance optimization with explicit uncertainty sets.

Classical MVO treats estimated expected returns as known; robust MVO
optimizes against the worst case over an uncertainty set around the
estimates (Bertsimas & Sim 2004; Goldfarb & Iyengar 2003).

Box uncertainty: ``|mu_i - mu_hat_i| <= delta_i``. The robust problem
``max_w min_{mu in U} mu'w - (g/2) w'Sw`` reduces *exactly* to standard MVO
with shrunk means ``mu_tilde = mu_hat - delta`` — but only when weights are
nonnegative (long-only), because the worst case puts ``d_i = -delta_i``
against ``w_i >= 0``. This module enforces long-only and documents the
condition; it is the retail-relevant case.

Ellipsoidal uncertainty: ``(mu - mu_hat)' Omega^{-1} (mu - mu_hat) <= k^2``,
where Omega is the estimation-error covariance of mu_hat (default:
``cov / n_obs``, the sample-mean error). The worst case gives the penalty
``k * sqrt(w' Omega w)`` — a concave maximization solved here with SLSQP
(analytic gradient).

``delta_from_se`` calibrates box half-widths from standard errors:
``delta_i = z * SE_i``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .constrained import opt_constrained

__all__ = ["delta_from_se", "robust_box_weights", "robust_ellipsoid_weights"]


def delta_from_se(standard_errors, z: float = 1.96) -> pd.Series:
    """Box half-widths from estimation standard errors: delta = z * SE."""
    se = pd.Series(standard_errors, dtype=float)
    return (z * se).rename("delta")


def robust_box_weights(mu, cov, delta, w_prev=None, risk_aversion: float = 5.0,
                       tc_bps: float = 10.0, bounds=(0.0, 1.0)) -> pd.Series:
    """Robust MVO under box uncertainty ``|mu_i - mu_hat_i| <= delta_i``.

    Exact for long-only portfolios (Bertsimas & Sim 2004): the worst-case
    mean is ``mu_hat - delta`` elementwise, so this is standard MVO on the
    shrunk means. ``bounds`` must stay nonnegative; a negative lower bound
    raises because the reduction would be inexact.
    """
    mu = pd.Series(mu, dtype=float)
    if np.isscalar(delta):
        delta = pd.Series(float(delta), index=mu.index)
    else:
        delta = pd.Series(delta, dtype=float).reindex(mu.index).fillna(0.0)
    lo, _ = bounds
    if lo < 0:
        raise ValueError("box reduction is exact only for long-only "
                         f"bounds; got lower bound {lo}")
    if (delta < 0).any():
        raise ValueError("delta must be nonnegative")
    return opt_constrained(mu - delta, cov, w_prev=w_prev,
                           risk_aversion=risk_aversion, tc_bps=tc_bps,
                           bounds=bounds)


def robust_ellipsoid_weights(mu, cov, kappa: float, omega=None,
                             n_obs: int | None = None,
                             risk_aversion: float = 5.0,
                             bounds=(0.0, 1.0)) -> pd.Series:
    """Robust MVO under ellipsoidal uncertainty on expected returns.

    Solves ``max_w mu_hat'w - k*sqrt(w'Om w) - (g/2) w'Sw``
    s.t. ``sum(w) = 1``, ``bounds``. ``omega`` is the estimation-error
    covariance of mu_hat; when omitted it defaults to ``cov / n_obs``
    (the sample-mean error, so ``n_obs`` is required then).
    ``kappa`` is the robustness budget (e.g. ``sqrt(chi2_n(0.95))``).
    """
    mu = pd.Series(mu, dtype=float)
    Sigma = pd.DataFrame(cov, dtype=float)
    n = len(mu)
    if kappa < 0:
        raise ValueError("kappa must be nonnegative")
    if omega is None:
        if n_obs is None or n_obs <= 0:
            raise ValueError("pass omega or n_obs")
        Om = Sigma.to_numpy() / n_obs
    else:
        Om = pd.DataFrame(omega, dtype=float).to_numpy()
    m = mu.to_numpy()
    S = Sigma.to_numpy()
    lo, hi = bounds

    def neg_obj(w):
        pen = kappa * np.sqrt(max(w @ Om @ w, 0.0))
        return -(m @ w - pen - 0.5 * risk_aversion * w @ S @ w)

    def neg_jac(w):
        denom = np.sqrt(max(w @ Om @ w, 1e-16))
        return -(m - kappa * (Om @ w) / denom - risk_aversion * S @ w)

    x0 = np.full(n, 1.0 / n)
    res = minimize(neg_obj, x0, jac=neg_jac, method="SLSQP",
                   bounds=[(lo, hi)] * n,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0,
                                "jac": lambda w: np.ones(n)},
                   options={"maxiter": 1000, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(f"robust ellipsoid solve failed: {res.message}")
    w = np.clip(res.x, lo, hi)
    return pd.Series(w / w.sum(), index=mu.index, name="weight")
