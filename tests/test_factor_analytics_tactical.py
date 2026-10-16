import numpy as np
import pandas as pd
import pytest

from portlab import analytics, tactical
from portlab.factor import compare_funds, factor_regression, rolling_exposures


@pytest.fixture(scope="module")
def synthetic_factors(rng):
    idx = pd.date_range("2010-01-31", periods=180, freq="ME")
    f = pd.DataFrame({
        "Mkt-RF": rng.normal(0.006, 0.045, 180),
        "SMB": rng.normal(0.001, 0.025, 180),
        "HML": rng.normal(0.001, 0.028, 180),
        "RF": np.full(180, 0.002),
    }, index=idx)
    return f


def test_capm_recovers_planted_beta_alpha(synthetic_factors, rng):
    f = synthetic_factors
    alpha, beta = 0.003, 1.3
    ret = f["RF"] + alpha + beta * f["Mkt-RF"] + rng.normal(0, 0.004, len(f))
    tbl = factor_regression(pd.Series(ret, index=f.index), f, model="capm",
                            periods=12)
    assert tbl.loc["Mkt-RF", "loading"] == pytest.approx(beta, abs=0.05)
    assert tbl.loc["alpha (per period)", "loading"] == pytest.approx(alpha, abs=0.002)
    assert tbl.attrs["r_squared"] > 0.9


def test_ff3_loadings(synthetic_factors, rng):
    f = synthetic_factors
    ret = f["RF"] + 1.0 * f["Mkt-RF"] + 0.5 * f["SMB"] - 0.3 * f["HML"] \
        + rng.normal(0, 0.003, len(f))
    tbl = factor_regression(pd.Series(ret, index=f.index), f, model="ff3")
    assert tbl.loc["SMB", "loading"] == pytest.approx(0.5, abs=0.08)
    assert tbl.loc["HML", "loading"] == pytest.approx(-0.3, abs=0.08)


def test_rolling_exposures_shape(synthetic_factors, rng):
    f = synthetic_factors
    ret = pd.Series(f["RF"] + f["Mkt-RF"] + rng.normal(0, 0.004, len(f)),
                    index=f.index)
    roll = rolling_exposures(ret, f, model="capm", window=36)
    assert len(roll) == len(f) - 36 + 1
    assert "Mkt-RF" in roll.columns


def test_compare_funds(synthetic_factors, rng):
    f = synthetic_factors
    funds = pd.DataFrame({
        "F1": f["RF"] + 0.8 * f["Mkt-RF"] + rng.normal(0, 0.004, len(f)),
        "F2": f["RF"] + 1.2 * f["Mkt-RF"] + rng.normal(0, 0.004, len(f)),
    }, index=f.index)
    tbl = compare_funds(funds, f, model="capm")
    assert tbl.loc["F2", "Mkt-RF"] > tbl.loc["F1", "Mkt-RF"]


# ---------------------------------------------------------------- analytics

def test_correlation_matrix_diag(returns):
    corr = analytics.correlation_matrix(returns)
    assert np.allclose(np.diag(corr.values), 1.0)


def test_autocorrelation_ci(returns):
    ac = analytics.autocorrelation(returns["A0"], lags=5)
    assert len(ac) == 5 and "ci95" in ac.attrs


def test_cointegration_of_cointegrated_pair(rng):
    # b tracks a with stationary noise -> should be cointegrated
    steps = rng.normal(0, 0.01, 1000)
    a = pd.Series(100 * np.exp(np.cumsum(steps)))
    b = a * np.exp(rng.normal(0, 0.005, 1000))
    res = analytics.cointegration_test(a, b)
    assert res["cointegrated (5%)"]


def test_performance_table(returns):
    tbl = analytics.performance_table(returns[["A0", "A1"]])
    assert tbl.shape[1] == 2 and "CAGR" in tbl.index


# ---------------------------------------------------------------- tactical

def test_ma_timing_weights_bounded(prices):
    w = tactical.ma_timing(prices[["A0", "A1"]], windows=50)
    assert ((w.sum(axis=1) <= 1.0 + 1e-9).all())


def test_ma_timing_with_cash_fully_invested(prices):
    w = tactical.ma_timing(prices[["A0", "A1"]], windows=50, out_asset="CASH")
    assert np.allclose(w.sum(axis=1), 1.0)


def test_relative_strength_holds_top_n(prices):
    w = tactical.relative_strength(prices, lookbacks=63, top_n=2)
    held = (w > 0).sum(axis=1)
    assert held.max() <= 2


def test_dual_momentum_goes_to_cash(prices):
    # falling cash-proxy makes absolute momentum easy to pass; use rising cash
    cash = pd.Series(np.linspace(100, 200, len(prices)), index=prices.index)
    w = tactical.dual_momentum(prices, cash, lookback=126, top_n=1,
                               out_asset="CASH")
    assert "CASH" in w.columns
    assert np.allclose(w.sum(axis=1), 1.0)


def test_target_volatility_caps_exposure(returns):
    w = tactical.target_volatility(returns, {"A7": 1.0}, target_annual=0.05,
                                   max_exposure=1.0)
    assert (w["A7"].dropna() <= 1.0 + 1e-9).all()


def test_seasonal_out_in_summer(returns):
    w = tactical.seasonal(returns.index, ["A0"], out_asset="CASH")
    july = w[w.index.month == 7]
    assert (july["A0"] == 0).all() and (july["CASH"] == 1).all()


def test_evaluate_no_lookahead(prices, returns):
    w = tactical.ma_timing(prices[["A0"]], windows=20)
    strat = tactical.evaluate(w, returns[["A0"]])
    # evaluate shifts weights by one period: strategy return at t uses w at t-1
    manual = (w["A0"].shift(1).reindex(returns.index).fillna(0) * returns["A0"])
    assert np.allclose(strat.fillna(0), manual.fillna(0), atol=1e-12)


def test_cape_valuation_weights(rng):
    idx = pd.date_range("1990-01-31", periods=400, freq="ME")
    cape = pd.Series(20 + 10 * np.sin(np.arange(400) / 30) +
                     rng.normal(0, 1, 400), index=idx)
    w = tactical.cape_valuation(cape, idx, "SPY", "AGG")
    assert np.allclose(w.sum(axis=1), 1.0)
    assert w["SPY"].min() >= 0.4 - 1e-9 and w["SPY"].max() <= 1.0 + 1e-9
