"""Factor regression analysis: CAPM, Fama-French 3/5, Carhart 4, FF5+Momentum.

Factor data comes free from the Ken French data library via
`portlab.data.factors.get_ff_factors`. Regressions use statsmodels OLS with
optional Newey-West (HAC) standard errors, plus rolling factor exposures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

MODEL_FACTORS = {
    "capm":    ["Mkt-RF"],
    "ff3":     ["Mkt-RF", "SMB", "HML"],
    "carhart": ["Mkt-RF", "SMB", "HML", "Mom"],
    "ff5":     ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    "ff5_mom": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
}


def factor_regression(
    rets: pd.Series,
    factors: pd.DataFrame,
    model: str = "ff3",
    rf_col: str = "RF",
    newey_west: bool = True,
    periods: int = 12,
) -> pd.DataFrame:
    """Regress excess returns on a factor model.

    rets: simple returns at the same frequency as `factors` (usually monthly).
    factors: DataFrame containing the model's factor columns plus `rf_col`,
        in decimal units (get_ff_factors handles the percent conversion).
    Returns a table with loadings, t-stats, p-values, and an annualized alpha
    row, with R² in `.attrs`.
    """
    if model not in MODEL_FACTORS:
        raise ValueError(f"model must be one of {sorted(MODEL_FACTORS)}")
    cols = MODEL_FACTORS[model]
    df = pd.concat([rets.rename("_ret"), factors], axis=1, join="inner").dropna()
    if len(df) < len(cols) + 10:
        raise ValueError("not enough overlapping observations for regression")
    y = df["_ret"] - df[rf_col]
    X = sm.add_constant(df[cols])
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6}) if newey_west \
        else sm.OLS(y, X).fit()

    out = pd.DataFrame({
        "loading": fit.params,
        "t_stat": fit.tvalues,
        "p_value": fit.pvalues,
    })
    out = out.rename(index={"const": "alpha (per period)"})
    out.loc["alpha (annualized)"] = [
        (1 + fit.params["const"]) ** periods - 1, np.nan, np.nan]
    out.attrs["r_squared"] = float(fit.rsquared)
    out.attrs["r_squared_adj"] = float(fit.rsquared_adj)
    out.attrs["n_obs"] = int(fit.nobs)
    return out


def rolling_exposures(
    rets: pd.Series,
    factors: pd.DataFrame,
    model: str = "ff3",
    window: int = 36,
    rf_col: str = "RF",
) -> pd.DataFrame:
    """Rolling factor loadings (default 36-period window)."""
    cols = MODEL_FACTORS[model]
    df = pd.concat([rets.rename("_ret"), factors], axis=1, join="inner").dropna()
    y = df["_ret"] - df[rf_col]
    X = sm.add_constant(df[cols])
    rows = {}
    for end in range(window, len(df) + 1):
        sl = slice(end - window, end)
        fit = sm.OLS(y.iloc[sl], X.iloc[sl]).fit()
        rows[df.index[end - 1]] = fit.params
    out = pd.DataFrame(rows).T
    return out.rename(columns={"const": "alpha"})


def compare_funds(
    rets: pd.DataFrame,
    factors: pd.DataFrame,
    model: str = "ff3",
    **kwargs,
) -> pd.DataFrame:
    """One regression per column; wide table of loadings/alpha/R² per fund."""
    rows = {}
    for col in rets.columns:
        try:
            tbl = factor_regression(rets[col].dropna(), factors, model, **kwargs)
        except ValueError:
            continue
        row = tbl["loading"].to_dict()
        row["R²"] = tbl.attrs["r_squared"]
        rows[col] = row
    return pd.DataFrame(rows).T


def attribution(
    rets: pd.Series,
    factors: pd.DataFrame,
    model: str = "ff3",
    rf_col: str = "RF",
) -> pd.DataFrame:
    """Factor performance attribution: cumulative return contribution of each
    factor exposure plus alpha and residual (selection) effects."""
    cols = MODEL_FACTORS[model]
    df = pd.concat([rets.rename("_ret"), factors], axis=1, join="inner").dropna()
    y = df["_ret"] - df[rf_col]
    X = sm.add_constant(df[cols])
    fit = sm.OLS(y, X).fit()
    parts = {c: fit.params[c] * df[c] for c in cols}
    parts["alpha"] = pd.Series(fit.params["const"], index=df.index)
    parts["residual"] = fit.resid
    parts["risk-free"] = df[rf_col]
    contrib = pd.DataFrame(parts)
    contrib.attrs["total"] = contrib.sum().rename("cumulative contribution")
    contrib.attrs["loadings"] = fit.params.rename(index={"const": "alpha"})
    return contrib


def match_exposure(
    target_rets: pd.Series,
    candidate_rets: pd.DataFrame,
    factors: pd.DataFrame | None = None,
    model: str = "ff3",
    bounds=(0.0, 1.0),
) -> dict:
    """Replicate a target fund with a portfolio of candidates: minimize
    tracking error vs the target, then compare the factor loadings of
    target and replica."""
    from .optimize import min_tracking_error
    from .metrics import tracking_error as te_metric

    w = min_tracking_error(candidate_rets, target_rets, bounds=bounds)
    replica = (candidate_rets[w.index] * w.values).sum(axis=1)
    out = {"weights": w,
           "tracking_error": te_metric(replica, target_rets),
           "replica_returns": replica}
    if factors is not None:
        both = pd.DataFrame({"Target": target_rets, "Replica": replica})
        out["loadings"] = compare_funds(both, factors, model=model)
    return out


def factor_statistics(
    factors: pd.DataFrame,
    model: str = "ff3",
    windows: tuple = (None, 60, 120),
    periods: int = 12,
) -> dict:
    """Factor correlations and risk premia over multiple periods.

    Mirrors PV's Factor Statistics tool: correlation matrix of the factor
    returns, plus annualized mean (the risk premium), volatility and Sharpe
    for the full sample and trailing windows (default 5Y / 10Y of monthly data).

    Returns {"correlation": DataFrame, "premia": {label: DataFrame}, "n_obs": int}.
    """
    if model not in MODEL_FACTORS:
        raise ValueError(f"model must be one of {sorted(MODEL_FACTORS)}")
    cols = MODEL_FACTORS[model]
    f = factors[cols].dropna()
    premia = {}
    for w in windows:
        d = f if w is None else f.iloc[-w:]
        label = "Full sample" if w is None else f"Trailing {w}M"
        mu = d.mean() * periods
        vol = d.std(ddof=1) * np.sqrt(periods)
        premia[label] = pd.DataFrame({
            "Ann. premium": mu,
            "Ann. vol": vol,
            "Sharpe": mu / vol.replace(0, np.nan),
            "Obs": len(d),
        })
    return {"correlation": f.corr(), "premia": premia, "n_obs": len(f)}


# ------------------------------------------------- risk decomposition
def portfolio_factor_exposures(weights: pd.Series,
                               betas: pd.DataFrame) -> pd.Series:
    """Portfolio factor loadings: beta_p = B'w.

    ``weights``: asset -> weight (Series). ``betas``: DataFrame of asset
    (index) x factor (columns) loadings. Returns the K-vector of portfolio
    factor exposures.
    """
    w = pd.Series(weights, dtype=float)
    b = betas.loc[w.index]
    return pd.Series(b.to_numpy().T @ w.to_numpy(), index=b.columns,
                     name="exposure")


def factor_risk_decomposition(weights: pd.Series, betas: pd.DataFrame,
                             factor_cov: pd.DataFrame,
                             idio_var=0.0, scale: float = 1.0) -> dict:
    """Decompose portfolio variance into systematic and idiosyncratic risk.

    With B the (n x K) loading matrix, F the (K x K) factor covariance and
    D = diag(d) the idiosyncratic variances, the model covariance is
    Sigma = B F B' + D and::

        total_var       = w' Sigma w
        systematic_var  = x' F x            with x = B'w (portfolio exposures)
        idiosyncratic_var = w' D w  (= total - systematic)

    Euler contributions (each sums exactly to its total):
      * ``factor_contrib``: x_k * (F x)_k per factor -> sums to systematic_var
        (can be negative for diversifying/negatively-correlated factors)
      * ``asset_contrib``: w_i * (Sigma w)_i per asset -> sums to total_var

    ``scale`` annualizes volatilities (e.g. 252 for daily inputs); variances
    are also reported scaled. The variance identity
    systematic + idiosyncratic == total holds to float precision.
    """
    w = pd.Series(weights, dtype=float)
    b = betas.loc[w.index].astype(float)
    f = factor_cov.loc[b.columns, b.columns].astype(float)
    if np.isscalar(idio_var):
        d = pd.Series(float(idio_var), index=w.index)
    else:
        d = pd.Series(idio_var, dtype=float).reindex(w.index).fillna(0.0)

    wv = w.to_numpy()
    bv = b.to_numpy()
    fv = f.to_numpy()
    dv = d.to_numpy()

    x = bv.T @ wv                              # portfolio factor exposures
    systematic_var = float(x @ fv @ x)
    idio_var_total = float((wv ** 2) @ dv)
    total_var = systematic_var + idio_var_total

    fx = fv @ x
    factor_contrib = pd.Series(x * fx, index=b.columns, name="var_contrib")
    sigma = bv @ fv @ bv.T + np.diag(dv)       # model covariance
    asset_contrib = pd.Series(wv * (sigma @ wv), index=w.index,
                              name="var_contrib")

    return {
        "exposures": pd.Series(x, index=b.columns, name="exposure"),
        "factor_contrib": factor_contrib,
        "asset_contrib": asset_contrib,
        "systematic_var": systematic_var * scale,
        "idiosyncratic_var": idio_var_total * scale,
        "total_var": total_var * scale,
        "systematic_pct": (systematic_var / total_var
                           if total_var > 0 else float("nan")),
        "systematic_vol": float(np.sqrt(systematic_var * scale)),
        "idiosyncratic_vol": float(np.sqrt(idio_var_total * scale)),
        "total_vol": float(np.sqrt(total_var * scale)),
    }


def decompose_from_returns(weights: pd.Series | dict, rets: pd.DataFrame,
                           factors: pd.DataFrame, model: str = "ff3",
                           rf_col: str = "RF", scale: float = 1.0) -> dict:
    """One-stop factor risk decomposition from return histories.

    Runs an OLS of each asset's excess returns on the model's factors
    (same setup as :func:`factor_regression`), takes betas and residual
    variances, and feeds them to :func:`factor_risk_decomposition`.

    Returns the decomposition dict plus ``betas`` (n x K), ``idio_var``
    (residual variances), ``factor_cov``, ``r_squared`` per asset and
    ``n_obs``.
    """
    if model not in MODEL_FACTORS:
        raise ValueError(f"model must be one of {sorted(MODEL_FACTORS)}")
    cols = MODEL_FACTORS[model]
    w = pd.Series(weights, dtype=float)
    r = rets[w.index].astype(float)
    df = pd.concat([r, factors[cols + [rf_col]]], axis=1,
                   join="inner").dropna()
    if len(df) < len(cols) + 10:
        raise ValueError("not enough overlapping observations")
    f = df[cols]
    fcov = f.cov()
    betas, ivar, r2, alphas = {}, {}, {}, {}
    for sym in r.columns:
        y = (df[sym] - df[rf_col]).to_numpy()
        X = np.column_stack([np.ones(len(df)), f.to_numpy()])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)  # [alpha, betas...]
        resid = y - X @ coef
        betas[sym] = coef[1:]
        alphas[sym] = coef[0]
        ivar[sym] = float(resid.var(ddof=X.shape[1]))
        r2[sym] = float(1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum())
    betas = pd.DataFrame(betas, index=cols).T
    out = factor_risk_decomposition(w, betas, fcov, ivar, scale=scale)
    out.update({"betas": betas,
                "idio_var": pd.Series(ivar, name="idio_var"),
                "factor_cov": fcov,
                "alphas": pd.Series(alphas, name="alpha"),
                "r_squared": pd.Series(r2, name="r_squared"),
                "n_obs": len(df),
                "model": model})
    return out
