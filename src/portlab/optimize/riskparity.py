"""Risk parity: equal risk contribution and inverse-volatility weighting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import solve_weights


def inverse_vol(cov: pd.DataFrame) -> pd.Series:
    iv = 1.0 / np.sqrt(np.diag(cov.values))
    return pd.Series(iv / iv.sum(), index=cov.index, name="weight")


def risk_contributions(w: pd.Series, cov: pd.DataFrame) -> pd.Series:
    wv = w.values
    total = np.sqrt(wv @ cov.values @ wv)
    mrc = cov.values @ wv / total
    return pd.Series(wv * mrc / total, index=cov.index, name="risk_contribution")


def equal_risk_contribution(cov: pd.DataFrame, bounds=(0.0, 1.0)) -> pd.Series:
    """Weights where every asset contributes equally to portfolio risk."""
    C = cov.values
    n = len(C)
    target = 1.0 / n

    def objective(w):
        vol = np.sqrt(w @ C @ w)
        if vol <= 0:
            return 1e6
        rc = w * (C @ w) / vol / vol   # fractional risk contributions
        return float(((rc - target) ** 2).sum())

    return solve_weights(objective, n, bounds,
                         x0=inverse_vol(cov).values, index=cov.index)
