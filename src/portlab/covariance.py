"""Covariance estimators: sample, Ledoit-Wolf shrinkage, EWMA, PSD repair."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .config import TRADING_DAYS


def cov_sample(rets: pd.DataFrame, annualize: int | None = None) -> pd.DataFrame:
    cov = rets.cov()
    return cov * annualize if annualize else cov


def cov_ledoit_wolf(rets: pd.DataFrame, annualize: int | None = None) -> pd.DataFrame:
    lw = LedoitWolf().fit(rets.values)
    cov = pd.DataFrame(lw.covariance_, index=rets.columns, columns=rets.columns)
    return cov * annualize if annualize else cov


def cov_semi(rets: pd.DataFrame, threshold: float = 0.0,
             annualize: int | None = None) -> pd.DataFrame:
    """Semicovariance: co-movement of below-threshold returns only
    (downside-risk-aware input for optimization)."""
    d = np.minimum(rets - threshold, 0.0)
    cov = pd.DataFrame(d.values.T @ d.values / max(len(rets) - 1, 1),
                       index=rets.columns, columns=rets.columns)
    return cov * annualize if annualize else cov


def cov_ewma(rets: pd.DataFrame, halflife: int = 60,
             annualize: int | None = None) -> pd.DataFrame:
    """Exponentially weighted covariance using the most recent estimate."""
    ew = rets.ewm(halflife=halflife).cov()
    cov = ew.loc[ew.index.get_level_values(0)[-1]]
    cov = cov.loc[rets.columns, rets.columns]
    return cov * annualize if annualize else cov


def get_cov(rets: pd.DataFrame, method: str = "ledoit_wolf",
            annualize: int | None = TRADING_DAYS, **kwargs) -> pd.DataFrame:
    methods = {"sample": cov_sample, "ledoit_wolf": cov_ledoit_wolf,
               "ewma": cov_ewma, "semi": cov_semi}
    if method not in methods:
        raise ValueError(f"method must be one of {sorted(methods)}")
    return psd_fix(methods[method](rets, annualize=annualize, **kwargs))


def psd_fix(cov: pd.DataFrame, min_eig: float = 1e-10) -> pd.DataFrame:
    """Clip negative eigenvalues so downstream solvers get a valid PSD matrix."""
    vals, vecs = np.linalg.eigh(cov.values)
    if vals.min() >= min_eig:
        return cov
    vals = np.clip(vals, min_eig, None)
    fixed = vecs @ np.diag(vals) @ vecs.T
    fixed = (fixed + fixed.T) / 2
    return pd.DataFrame(fixed, index=cov.index, columns=cov.columns)


def corr_from_cov(cov: pd.DataFrame) -> pd.DataFrame:
    sd = np.sqrt(np.diag(cov.values))
    corr = cov.values / np.outer(sd, sd)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)
