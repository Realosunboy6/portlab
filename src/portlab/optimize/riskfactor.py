"""Risk-factor-based allocation: match user-specified factor exposure targets.

Parity with Portfolio Visualizer's "Risk Factor Based Allocation" tool
(/risk-factor-allocation). PV's flow, from its own worked example:

1. Estimate each asset's factor loadings by regression over a lookback window
   (default 36 months) on the selected factor model.
2. Compute the provided portfolio's current factor exposures.
3. Find the long-only, fully-invested portfolio whose factor exposures best
   match the user's target exposures, e.g. hold market/size flat while
   raising value (HML) 0.15 -> 0.20 and cutting term risk 0.20 -> 0.10.

Formulation
-----------
min_w  sum_k fw_k * (B_k' w - t_k)^2
s.t.   sum(w) = 1, lo_i <= w_i <= hi_i, [optional] turnover(w, w0) <= tau

where B is the (assets x factors) loading matrix from the regressions, t the
target exposure vector, and fw per-factor weights (default: equal weight).
The objective is convex quadratic; solved with the shared SLSQP plumbing in
`portlab.optimize.core`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..factor import MODEL_FACTORS, compare_funds
from .core import solve_weights


def portfolio_exposures(weights: pd.Series, loadings: pd.DataFrame) -> pd.Series:
    """Current factor exposures of a portfolio: B'w."""
    return loadings.T @ weights.reindex(loadings.index).fillna(0.0).values


def exposure_gap(
    weights: pd.Series,
    loadings: pd.DataFrame,
    targets: pd.Series,
    factor_weights: pd.Series | None = None,
) -> pd.Series:
    """Per-factor gap between achieved exposures and targets (achieved - target)."""
    achieved = portfolio_exposures(weights, loadings)
    return achieved - targets.reindex(achieved.index).fillna(0.0)


def risk_factor_allocation(
    loadings: pd.DataFrame,
    targets: pd.Series,
    current_weights: pd.Series | None = None,
    factor_weights: pd.Series | None = None,
    bounds: tuple[float, float] = (0.0, 1.0),
    max_turnover: float | None = None,
) -> dict:
    """Find long-only, fully-invested weights matching target factor exposures.

    loadings: DataFrame indexed by asset, columns = factor names (regression betas).
    targets: Series indexed by factor name with desired exposures. Factors present
        in `targets` but missing from `loadings` are ignored.
    current_weights: starting portfolio (for the turnover report and the optional
        turnover cap). Defaults to equal weight.
    factor_weights: per-factor importance in the objective. Defaults to 1.0 each.
    bounds: (lo, hi) per-asset weight bounds. Default long-only (0, 1).
    max_turnover: optional cap on sum |w - w0| (two-way turnover vs current).

    Returns dict with `weights`, `achieved`, `targets`, `gap`, `objective`
    (weighted sum of squared gaps), and `turnover` vs current weights.
    """
    assets = list(loadings.index)
    n = len(assets)
    facs = [f for f in targets.index if f in loadings.columns]
    if not facs:
        raise ValueError("no target factor found in loadings columns")
    B = loadings[facs].values  # (n_assets, n_factors)
    t = targets.reindex(facs).values.astype(float)
    fw = np.ones(len(facs)) if factor_weights is None else \
        factor_weights.reindex(facs).fillna(1.0).values.astype(float)

    if current_weights is None:
        w0 = np.full(n, 1.0 / n)
    else:
        w0 = current_weights.reindex(assets).fillna(0.0).values.astype(float)
        if w0.sum() > 0:
            w0 = w0 / w0.sum()

    def objective(w: np.ndarray) -> float:
        gap = B.T @ w - t
        return float(np.sum(fw * gap * gap))

    extra = None
    if max_turnover is not None:
        extra = [{"type": "ineq",
                  "fun": lambda w: max_turnover - np.abs(w - w0).sum()}]

    w = solve_weights(objective, n, bounds=bounds, extra_constraints=extra,
                      x0=w0.copy(), index=pd.Index(assets))
    achieved = pd.Series(B.T @ w.values, index=facs, name="achieved")
    tgt = pd.Series(t, index=facs, name="target")
    gap = achieved - tgt
    return {
        "weights": w,
        "achieved": achieved,
        "targets": tgt,
        "gap": gap,
        "objective": float(np.sum(fw * gap.values * gap.values)),
        "turnover": float(np.abs(w.values - w0).sum()),
    }


def allocate_from_returns(
    rets: pd.DataFrame,
    factors: pd.DataFrame,
    targets: pd.Series | dict,
    model: str = "ff3",
    window: int = 36,
    current_weights: pd.Series | dict | None = None,
    **kwargs,
) -> dict:
    """End-to-end PV flow: regressions over the last `window` periods, then allocate.

    rets: asset returns (monthly for FF factors) as DataFrame.
    factors: factor DataFrame from `portlab.data.factors.get_ff_factors`
        (decimal units, with an RF column).
    targets: {factor: target exposure}.
    Returns the same dict as `risk_factor_allocation`, plus `loadings` and `r_squared`.
    """
    if model not in MODEL_FACTORS:
        raise ValueError(f"model must be one of {sorted(MODEL_FACTORS)}")
    facs = MODEL_FACTORS[model]
    tgt = pd.Series(targets, dtype=float)
    use = [f for f in tgt.index if f in facs]
    if not use:
        raise ValueError("no target factor belongs to the selected model")

    df = pd.concat([rets, factors], axis=1, join="inner").dropna()
    if len(df) < window:
        raise ValueError(f"only {len(df)} overlapping periods, need {window}")
    df = df.iloc[-window:]
    tbl = compare_funds(df[rets.columns], df, model=model, newey_west=False)
    loadings = tbl[facs]
    r2 = tbl["R²"]
    if isinstance(current_weights, dict):
        current_weights = pd.Series(current_weights)
    res = risk_factor_allocation(loadings, tgt[use], current_weights=current_weights,
                                 **kwargs)
    res["loadings"] = loadings
    res["r_squared"] = r2
    res["n_obs"] = len(df)
    return res
