"""Tests for portlab.scenarios (filtered historical simulation).

ALL DATA HERE IS SYNTHETIC. External anchors: the FHS method is
Barone-Adesi/Giannopoulos/Vosper (1999); the key property tests below
(vol-adaptation, correlation preservation) are definitional to FHS.
"""

import numpy as np
import pandas as pd
import pytest

from portlab import scenarios as S
from portlab.risk.var import _garch11_forecast


@pytest.fixture()
def garch_rets():
    rng = np.random.default_rng(20261010)
    n, T = 2, 600
    # correlated GARCH(1,1) pair
    out = np.zeros((T, n))
    s2 = np.full(n, 0.0004)
    L = np.linalg.cholesky(np.array([[1.0, 0.6], [0.6, 1.0]]))
    for t in range(1, T):
        s2 = 0.00002 + 0.08 * out[t - 1] ** 2 + 0.90 * s2
        out[t] = np.sqrt(s2) * (L @ rng.normal(size=n))
    idx = pd.bdate_range("2020-01-01", periods=T)
    return pd.DataFrame(out, index=idx, columns=["A", "B"])


def test_adapts_to_current_vol(garch_rets):
    """Scenario first-step vol must match the GARCH forecast, not the
    trailing sample std (FHS adapts to current conditions)."""
    sc = S.fhs_scenarios(garch_rets, n_scenarios=2000, horizon=1,
                         window=250, seed=1)
    for i, col in enumerate(garch_rets.columns):
        fc = _garch11_forecast(garch_rets[col].iloc[-250:].to_numpy())
        assert sc[:, 0, i].std(ddof=1) == pytest.approx(fc, rel=0.15)


def test_spike_raises_scenario_var():
    rng = np.random.default_rng(5)
    calm = rng.normal(0, 0.01, 400)
    # same history, then a vol spike at the end vs calm ending
    base = pd.DataFrame({"A": calm},
                        index=pd.bdate_range("2020-01-01", periods=400))
    spiky = pd.DataFrame({"A": np.concatenate([calm[:-20],
                                               rng.normal(0, 0.05, 20)])},
                         index=base.index)
    w = np.array([1.0])
    v_calm = S.scenario_var_es(
        S.fhs_scenarios(base, n_scenarios=1500, horizon=21, window=250,
                        seed=7), w, alpha=0.05)["var"]
    v_spike = S.scenario_var_es(
        S.fhs_scenarios(spiky, n_scenarios=1500, horizon=21, window=250,
                        seed=7), w, alpha=0.05)["var"]
    assert v_spike > v_calm * 1.5


def test_correlation_preserved(garch_rets):
    sc = S.fhs_scenarios(garch_rets, n_scenarios=3000, horizon=5,
                         window=250, seed=2)
    sample_corr = garch_rets.iloc[-250:].corr().iloc[0, 1]
    scen_corr = np.corrcoef(sc[:, 0, 0], sc[:, 0, 1])[0, 1]
    assert scen_corr == pytest.approx(sample_corr, abs=0.12)


def test_seed_reproducible(garch_rets):
    a = S.fhs_scenarios(garch_rets, n_scenarios=100, horizon=5, seed=42)
    b = S.fhs_scenarios(garch_rets, n_scenarios=100, horizon=5, seed=42)
    assert np.array_equal(a, b)


def test_var_es_properties(garch_rets):
    sc = S.fhs_scenarios(garch_rets, n_scenarios=2000, horizon=21, seed=3)
    out = S.scenario_var_es(sc, np.array([0.5, 0.5]), alpha=0.05)
    assert out["es"] >= out["var"] * 0.999  # ES dominates VaR
    assert out["var"] > 0
    # independent definition check
    cum = np.prod(1 + sc @ np.array([0.5, 0.5]), axis=1) - 1
    assert out["var"] == pytest.approx(np.quantile(-cum, 0.95), rel=1e-9)


def test_block_resampling_shape(garch_rets):
    sc = S.fhs_scenarios(garch_rets, n_scenarios=500, horizon=20,
                         block=5, seed=9)
    assert sc.shape == (500, 20, 2)


def test_validation():
    r = pd.DataFrame({"A": np.random.default_rng(0).normal(0, 0.01, 30)},
                     index=pd.bdate_range("2020-01-01", periods=30))
    with pytest.raises(ValueError, match="at least"):
        S.fhs_scenarios(r, window=250)
    with pytest.raises(ValueError, match="bounds"):
        S.fhs_scenarios(r, n_scenarios=0)
