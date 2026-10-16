"""Monte-Carlo scenario generation via Filtered Historical Simulation.

Barone-Adesi, Giannopoulos & Vosper (1999): fit GARCH(1,1) per asset,
standardize returns by conditional vol, then resample the *standardized*
residuals — the same time indices across all assets, so cross-sectional
dependence (correlation) is preserved — and re-scale by a simulated forward
vol path. Unlike plain historical replay (:mod:`portlab.stress`), FHS adapts
to *current* volatility: after a vol spike, all scenarios are scaled up.

Use for: multi-horizon VaR/ES, "10,000 plausible crises" stress testing,
and scenario-based optimization inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .risk.var import _garch11_fit, _garch11_volpath

__all__ = ["fhs_scenarios", "scenario_var_es"]


def fhs_scenarios(rets: pd.DataFrame, n_scenarios: int = 1000,
                  horizon: int = 21, window: int = 250,
                  block: int = 1, seed: int | None = None,
                  demean: bool = True) -> np.ndarray:
    """Generate (n_scenarios, horizon, n_assets) return scenarios.

    ``rets``: DataFrame of per-period returns. The trailing ``window``
    observations drive the GARCH fits and the residual pool. ``block``:
    resample blocks of this length to preserve short-horizon dependence
    in the standardized residuals. ``seed`` for reproducibility.
    """
    if n_scenarios < 1 or horizon < 1 or window < 20 or block < 1:
        raise ValueError("check n_scenarios/horizon/window/block bounds")
    r = rets.dropna()
    if len(r) < window:
        raise ValueError(f"need at least {window} observations")
    w = r.iloc[-window:]
    n = w.shape[1]
    rng = np.random.default_rng(seed)

    # per-asset GARCH fit + standardized residuals
    mus, vols, zs = {}, {}, {}
    params = {}
    for col in w.columns:
        x = w[col].to_numpy()
        mu = x.mean() if demean else 0.0
        try:
            omega, a, b, vt = _garch11_fit(x - mu)
            sig = _garch11_volpath(x - mu, omega, a, b, vt)
            s2_last = sig[-1] ** 2
            s2_next = omega + a * (x[-1] - mu) ** 2 + b * s2_last
            sig0 = float(np.sqrt(max(s2_next, 1e-12)))
        except ValueError:
            # fall back to constant vol: still a valid FHS, just without
            # vol dynamics for this asset
            std = float(x.std(ddof=1))
            if not np.isfinite(std) or std <= 0:
                raise ValueError(f"no usable volatility for {col}")
            omega, a, b, sig0 = std ** 2, 0.0, 0.0, std
            sig = np.full(len(x), std)
        sig = np.maximum(sig, 1e-8)
        mus[col] = mu
        vols[col] = sig
        zs[col] = (x - mu) / sig
        params[col] = (omega, a, b, sig0, mu)
    zmat = np.column_stack([zs[c] for c in w.columns])  # (window, n)

    # block resampling indices (same across assets -> dependence preserved)
    n_blocks = int(np.ceil(horizon / block))
    idx = np.empty((n_scenarios, n_blocks * block), dtype=int)
    for s in range(n_scenarios):
        starts = rng.integers(0, window - block + 1, size=n_blocks)
        idx[s] = np.concatenate([np.arange(st, st + block) for st in starts])
    idx = idx[:, :horizon]

    out = np.empty((n_scenarios, horizon, n))
    for i, col in enumerate(w.columns):
        omega, a, b, sig0, mu = params[col]
        zdraw = zmat[idx, i]                      # (S, H)
        s2 = np.full(n_scenarios, sig0 ** 2)
        for h in range(horizon):
            sig_h = np.sqrt(np.maximum(s2, 1e-12))
            r_star = mu + sig_h * zdraw[:, h]
            out[:, h, i] = r_star
            s2 = omega + a * r_star ** 2 + b * s2  # forward vol path
    return out


def scenario_var_es(scenarios: np.ndarray, weights,
                    alpha: float = 0.05) -> dict:
    """H-horizon VaR/ES (positive = loss) from scenario cubes.

    ``scenarios``: (n_scenarios, horizon, n_assets) per-period returns.
    ``weights``: portfolio weights. Compounds each scenario's path to the
    H-day cumulative return, then takes the historical VaR/ES of the loss
    distribution.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    w = np.asarray(weights, dtype=float)
    if scenarios.shape[2] != len(w):
        raise ValueError("scenario assets != len(weights)")
    port = scenarios @ w                            # (S, H) per-step
    cum = np.prod(1.0 + port, axis=1) - 1.0          # H-day return
    losses = -cum
    var = float(np.quantile(losses, 1 - alpha))
    es = float(losses[losses >= var].mean())
    return {"var": var, "es": es, "mean": float(cum.mean()),
            "n_scenarios": scenarios.shape[0],
            "horizon": scenarios.shape[1]}
