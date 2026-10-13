"""Tests for portlab.risk.regime.

Synthetic data for unit properties; REAL market data (SPY) for the external
check: the detector must flag the March 2020 COVID crash as stressed (the
same window in portlab.stress.CRISES).
"""

import numpy as np
import pandas as pd
import pytest

from portlab.risk import regime as R


def _series(x, start="2018-01-01"):
    return pd.Series(np.asarray(x, dtype=float),
                     index=pd.bdate_range(start, periods=len(x)))


def test_calm_stays_calm():
    rng = np.random.default_rng(0)
    r = _series(rng.normal(0, 0.01, 800))
    reg = R.detect_regime(r).dropna()
    assert (reg == 0).mean() > 0.95


def test_vol_spike_detected_with_hysteresis():
    rng = np.random.default_rng(1)
    calm = rng.normal(0, 0.01, 400)
    shock = rng.normal(0, 0.05, 60)
    r = _series(np.concatenate([calm, shock, calm]))
    reg = R.detect_regime(r)
    # the shock block must contain a stressed run (no dropna before iloc:
    # the first `lookback` entries are NaN warmup)
    assert (reg.iloc[400:460] == 1).sum() >= 30
    # ...and calm periods must not flicker: minimum run length >= confirm
    v = reg.dropna().to_numpy()
    runs, cur = [], 1
    for i in range(1, len(v)):
        if v[i] == v[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    assert min(runs) >= 5


def test_stressed_var_exceeds_calm():
    rng = np.random.default_rng(2)
    r = _series(np.concatenate([rng.normal(0, 0.01, 500),
                                rng.normal(0, 0.04, 100),
                                rng.normal(0, 0.01, 200)]))
    out = R.regime_var(r, alpha=0.05)
    assert out["stressed_var"] > out["calm_var"] * 1.5
    assert out["full_var"] > out["calm_var"]


def test_detects_march_2020_on_real_spy():
    """EXTERNAL: real SPY must read stressed through the COVID window."""
    yf = pytest.importorskip("yfinance")
    px = yf.download("SPY", start="2018-01-01", end="2020-06-01",
                     auto_adjust=True, progress=False)["Close"]
    px = px["SPY"] if "SPY" in px else px.iloc[:, 0]
    r = px.pct_change().dropna()
    reg = R.detect_regime(r)
    # the acute crash phase must read stressed throughout (the detector
    # needs confirmation, so the initial Feb slide lags — honest behavior
    # for a vol-based regime filter)
    acute = reg.loc["2020-03-01":"2020-03-23"].dropna()
    assert (acute == 1).mean() > 0.9
    # and early 2019 must be calm
    calm = reg.loc["2019-01-01":"2019-06-01"].dropna()
    assert (calm == 0).mean() > 0.8
    out = R.regime_var(r)
    assert out["current_regime"] in (0, 1)
    assert out["stressed_var"] > out["full_var"]


def test_ewma_vol_positive_and_responsive():
    rng = np.random.default_rng(3)
    r = _series(np.concatenate([rng.normal(0, 0.01, 300),
                                rng.normal(0, 0.05, 50)]))
    v = R.ewma_vol(r)
    assert (v > 0).all()
    assert v.iloc[-1] > v.iloc[300] * 2  # reacts to the shock


def test_long_baseline_default_off_is_backward_compatible():
    rng = np.random.default_rng(4)
    r = _series(rng.normal(0, 0.01, 600))
    a = R.detect_regime(r)
    b = R.detect_regime(r, long_baseline=False)
    pd.testing.assert_series_equal(a, b)


def test_long_baseline_flags_sustained_high_vol():
    """A sustained 3x-vol regime: the trailing median adapts (reads calm
    late), but long_baseline=True keeps it stressed — the persistent-regime
    behavior validated against a 2-state HMM reference in the bake-off."""
    rng = np.random.default_rng(5)
    r = _series(np.concatenate([rng.normal(0, 0.01, 300),
                                rng.normal(0, 0.03, 300)]))
    late = slice(550, 600)
    reg_default = R.detect_regime(r)
    assert (reg_default.iloc[late] == 0).mean() > 0.8  # adapted: calm
    reg_long = R.detect_regime(r, long_baseline=True)
    assert (reg_long.iloc[late] == 1).mean() > 0.8  # still stressed
    # entry still caught, calm start still calm
    assert (reg_long.iloc[300:340].dropna() == 1).mean() > 0.5
    assert (reg_long.iloc[252:295].dropna() == 0).mean() > 0.9
    # warmup unchanged (first lookback-1 obs NaN, index 251 is the 252nd)
    assert reg_long.iloc[:251].isna().all()


def test_long_baseline_calm_stays_calm():
    rng = np.random.default_rng(6)
    r = _series(rng.normal(0, 0.01, 800))
    reg = R.detect_regime(r, long_baseline=True).dropna()
    assert (reg == 0).mean() > 0.95
