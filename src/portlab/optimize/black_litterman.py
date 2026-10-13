"""Black-Litterman expected returns: market-implied prior blended with views."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DEFAULT_RF


def implied_returns(cov: pd.DataFrame, market_weights: pd.Series,
                    risk_aversion: float = 2.5, rf: float = DEFAULT_RF) -> pd.Series:
    """Reverse-optimized equilibrium excess returns Pi = delta * Sigma * w_mkt."""
    w = market_weights.reindex(cov.index).fillna(0.0)
    w = w / w.sum()
    pi = risk_aversion * cov.values @ w.values
    return pd.Series(pi + rf, index=cov.index, name="implied_return")


def black_litterman(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    P: pd.DataFrame,
    Q: pd.Series,
    view_confidence: pd.Series | None = None,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    rf: float = DEFAULT_RF,
) -> pd.Series:
    """Posterior expected returns given views.

    P: views x assets pick matrix (columns must match cov's assets).
    Q: expected annual return of each view (absolute for single-asset rows,
       spread for long/short rows).
    view_confidence: per-view confidence in (0, 1]; defaults to proportional
       Omega = tau * P Sigma P' (He-Litterman).
    """
    pi = implied_returns(cov, market_weights, risk_aversion, rf) - rf
    Pm = P.reindex(columns=cov.index).fillna(0.0).values
    Qv = Q.values.astype(float) - rf * (np.abs(Pm).sum(axis=1) == Pm.sum(axis=1))
    S = cov.values
    tauS = tau * S
    omega_diag = np.diag(Pm @ tauS @ Pm.T).copy()
    if view_confidence is not None:
        conf = view_confidence.values.astype(float)
        omega_diag = omega_diag * (1 - conf) / np.maximum(conf, 1e-6)
    Omega = np.diag(np.maximum(omega_diag, 1e-12))

    middle = np.linalg.inv(np.linalg.inv(tauS) + Pm.T @ np.linalg.inv(Omega) @ Pm)
    posterior = middle @ (np.linalg.inv(tauS) @ pi.values
                          + Pm.T @ np.linalg.inv(Omega) @ Qv)
    return pd.Series(posterior + rf, index=cov.index, name="bl_return")
