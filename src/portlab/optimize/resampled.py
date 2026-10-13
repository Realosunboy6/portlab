"""Michaud-style resampled optimization: average optimal weights across
bootstrap re-estimates of (mu, cov) to reduce estimation-error fragility."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DEFAULT_RF, TRADING_DAYS
from .meanvar import gmv, max_sharpe


def resampled_weights(
    rets: pd.DataFrame,
    objective: str = "max_sharpe",
    n_samples: int = 100,
    rf: float = DEFAULT_RF,
    periods: int = TRADING_DAYS,
    bounds=(0.0, 1.0),
    seed: int = 42,
) -> pd.Series:
    """Bootstrap rows of `rets`, re-optimize each sample, average the weights."""
    from ..covariance import get_cov
    rng = np.random.default_rng(seed)
    T = len(rets)
    acc = np.zeros(rets.shape[1])
    done = 0
    for _ in range(n_samples):
        sample = rets.iloc[rng.integers(0, T, T)]
        mu = sample.mean() * periods
        cov = get_cov(sample, method="ledoit_wolf", annualize=periods)
        try:
            if objective == "max_sharpe":
                w = max_sharpe(mu, cov, rf=rf, bounds=bounds)
            elif objective == "gmv":
                w = gmv(mu, cov, bounds=bounds)
            else:
                raise ValueError(f"unknown objective {objective!r}")
        except Exception:
            continue
        acc += w.values
        done += 1
    if done == 0:
        raise RuntimeError("all resampled optimizations failed")
    w = acc / done
    return pd.Series(w / w.sum(), index=rets.columns, name="weight")
