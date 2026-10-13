"""Stress testing: historical worst windows and hypothetical shock scenarios
(from Portfolio_Optimization_COLAB Phase 7)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Named historical stress windows (extendable by the caller).
HISTORICAL_EPISODES = {
    "GFC 2008":           ("2007-10-01", "2009-03-31"),
    "Eurozone 2011":      ("2011-05-01", "2011-10-31"),
    "Taper Tantrum 2013": ("2013-05-01", "2013-09-30"),
    "China Deval 2015":   ("2015-06-01", "2016-02-29"),
    "Volmageddon 2018":   ("2018-01-26", "2018-04-30"),
    "Q4 2018":            ("2018-10-01", "2018-12-31"),
    "COVID Crash 2020":   ("2020-02-19", "2020-03-23"),
    "Inflation Bear 2022": ("2022-01-01", "2022-10-31"),
}


def worst_windows(port_rets: pd.Series, window: int = 21, top: int = 10) -> pd.DataFrame:
    """Worst rolling `window`-period cumulative returns (non-overlapping)."""
    cum = (1 + port_rets).rolling(window).apply(np.prod, raw=True) - 1
    cum = cum.dropna().sort_values()
    rows, used = [], []
    for end_date, val in cum.items():
        end_loc = port_rets.index.get_loc(end_date)
        start_loc = end_loc - window + 1
        if any(not (end_loc < s or start_loc > e) for s, e in used):
            continue
        used.append((start_loc, end_loc))
        rows.append({"Start": port_rets.index[start_loc], "End": end_date,
                     "Return": float(val)})
        if len(rows) >= top:
            break
    return pd.DataFrame(rows)


def episode_returns(port_rets: pd.Series,
                    episodes: dict[str, tuple[str, str]] | None = None) -> pd.DataFrame:
    """Portfolio performance through named historical stress episodes."""
    episodes = episodes or HISTORICAL_EPISODES
    rows = []
    for name, (start, end) in episodes.items():
        window = port_rets.loc[start:end]
        if len(window) < 2:
            continue
        rows.append({"Episode": name, "Start": start, "End": end,
                     "Return": float((1 + window).prod() - 1),
                     "Worst Period": float(window.min())})
    return pd.DataFrame(rows)


def shock_scenario(weights: pd.Series, mu: pd.Series, cov: pd.DataFrame,
                   return_shock: float = 0.0, vol_mult: float = 1.0,
                   corr_add: float = 0.0) -> dict[str, float]:
    """Hypothetical shock: shift returns, scale vols, push correlations up.

    corr_add: added to all off-diagonal correlations (capped at 0.99) —
    models the 'correlations go to 1 in a crisis' effect.
    """
    from ..covariance import corr_from_cov, psd_fix
    w = weights.reindex(mu.index).fillna(0.0).values
    mu_s = mu.values + return_shock
    sd = np.sqrt(np.diag(cov.values)) * vol_mult
    corr = corr_from_cov(cov).values
    if corr_add:
        corr = np.clip(corr + corr_add, -0.99, 0.99)
        np.fill_diagonal(corr, 1.0)
    cov_s = psd_fix(pd.DataFrame(np.outer(sd, sd) * corr,
                                 index=cov.index, columns=cov.columns))
    return {"expected_return": float(w @ mu_s),
            "volatility": float(np.sqrt(w @ cov_s.values @ w))}
