"""Mean-variance with turnover and transaction-cost penalties (from the
original Portfolio_Optimization_COLAB Phase 4), extended with group constraints."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DEFAULT_RF
from .core import solve_weights


def opt_constrained(
    mu: pd.Series,
    cov: pd.DataFrame,
    w_prev: pd.Series | None = None,
    risk_aversion: float = 5.0,
    tc_bps: float = 10.0,
    max_turnover: float | None = None,
    bounds=(0.0, 0.05),
    groups=None,
    rf: float = DEFAULT_RF,
) -> pd.Series:
    """Maximize mu·w - (risk_aversion/2)·wΣw - transaction costs vs w_prev.

    tc_bps: one-way transaction cost in basis points applied to |w - w_prev|.
    max_turnover: optional hard cap on sum |w - w_prev| (one-way).
    bounds: per-asset (min, max), default 0-5% like the original pipeline.
    """
    m, C = mu.values, cov.values
    wp = w_prev.reindex(mu.index).fillna(0.0).values if w_prev is not None else None
    tc = tc_bps / 1e4

    def objective(w):
        val = -(w @ m) + 0.5 * risk_aversion * (w @ C @ w)
        if wp is not None:
            val += tc * np.abs(w - wp).sum()
        return val

    extra = []
    if wp is not None and max_turnover is not None:
        extra.append({"type": "ineq",
                      "fun": lambda w: max_turnover - np.abs(w - wp).sum()})
    return solve_weights(objective, len(mu), bounds, groups,
                         extra_constraints=extra or None, index=mu.index)
