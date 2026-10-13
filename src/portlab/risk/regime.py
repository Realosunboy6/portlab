"""Volatility-regime detection and regime-conditional risk.

Trailing-window risk estimates lag regime shifts — the report notes even
Barra's parametric VaR understated tails in the 2020/2022 shifts. This module
classifies each date as calm or stressed from an EWMA (RiskMetrics) volatility
proxy and reports risk *conditional* on the regime:

- :func:`detect_regime`: 0 = calm, 1 = stressed, with hysteresis (a regime
  change needs ``confirm`` consecutive signals) to avoid flicker.
  ``long_baseline=True`` additionally flags *sustained* high-volatility
  regimes that the trailing median would otherwise adapt to.
- :func:`regime_var`: historical VaR estimated on stressed-regime history
  vs calm history vs the full sample — "what is my VaR when it matters".

The stressed-regime VaR is the conservative number to watch in RiskLens:
it answers how bad the tail has been *in past stress*, not diluted by calm
years.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["ewma_vol", "detect_regime", "regime_var"]


def ewma_vol(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA volatility (daily)."""
    r = pd.Series(returns, dtype=float)
    var = r.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var)


def detect_regime(returns: pd.Series, lam: float = 0.94,
                  lookback: int = 252, thresh: float = 1.5,
                  confirm: int = 5, long_baseline: bool = False) -> pd.Series:
    """Classify each date as calm (0) or stressed (1).

    Signal: EWMA vol above ``thresh`` × its trailing ``lookback``-day median.
    With ``long_baseline=True`` the signal is OR-ed with EWMA vol above
    ``thresh`` × its *expanding* median (from the first ``lookback``
    observations). The trailing median adapts to a sustained high-volatility
    regime and stops flagging it; the expanding baseline adapts much more
    slowly, so permanently elevated volatility keeps reading stressed —
    matching the persistent-regime behavior of Markov-switching models
    (Hamilton 1989) without their estimation cost or false-alarm rate.
    A regime change is confirmed only after ``confirm`` consecutive signals
    (hysteresis); the first ``lookback`` observations are NaN (warming up).
    """
    r = pd.Series(returns, dtype=float)
    vol = ewma_vol(r, lam)
    med = vol.rolling(lookback, min_periods=lookback).median()
    raw = vol > thresh * med
    if long_baseline:
        med_long = vol.expanding(min_periods=lookback).median()
        raw = raw | (vol > thresh * med_long)
    raw = raw.astype(int)
    raw[med.isna()] = np.nan
    out = pd.Series(np.nan, index=r.index, dtype=float)
    vals = raw.to_numpy()
    # hysteresis state machine
    state = 0
    run = 0
    started = False
    for i, v in enumerate(vals):
        if np.isnan(v):
            continue
        v = int(v)
        if not started:
            state, started = v, True
            out.iloc[i] = state
            continue
        if v == state:
            run = 0
        else:
            run += 1
            if run >= confirm:
                state, run = v, 0
        out.iloc[i] = state
    return out.astype("Int64")


def regime_var(returns: pd.Series, alpha: float = 0.05,
               regime: pd.Series | None = None, **kwargs) -> dict:
    """Historical VaR conditional on volatility regime (positive = loss).

    Returns ``calm_var`` (VaR on calm-regime history), ``stressed_var``,
    ``full_var``, ``current_regime`` (0/1 at the last date),
    ``stressed_share`` (fraction of history spent stressed) and
    ``n_stressed``. ``regime`` may be supplied; otherwise detected with
    ``kwargs`` forwarded to :func:`detect_regime`.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    r = pd.Series(returns, dtype=float).dropna()
    if regime is None:
        regime = detect_regime(r, **kwargs)
    regime = pd.Series(regime, index=r.index[:len(regime)]
                       if len(regime) != len(r) else r.index)
    regime = regime.reindex(r.index)
    calm = r[regime == 0]
    stressed = r[regime == 1]

    def _hvar(x):
        return float(-x.quantile(alpha)) if len(x) >= 20 else float("nan")

    cur = regime.dropna()
    return {"calm_var": _hvar(calm), "stressed_var": _hvar(stressed),
            "full_var": _hvar(r),
            "current_regime": int(cur.iloc[-1]) if len(cur) else None,
            "stressed_share": float((regime == 1).mean()),
            "n_stressed": int((regime == 1).sum()),
            "n_obs": len(r)}
