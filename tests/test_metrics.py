import numpy as np
import pandas as pd
import pytest

from portlab import metrics as M


def test_cagr_matches_total_growth(returns):
    r = returns["A0"]
    growth = float((1 + r).prod())
    assert (1 + M.cagr(r)) ** (len(r) / 252) == pytest.approx(growth, rel=1e-9)


def test_sharpe_zero_rf_sign(returns):
    r = returns["A7"]
    assert M.sharpe(r, rf=0.0) == pytest.approx(
        r.mean() / r.std(ddof=1) * np.sqrt(252))


def test_max_drawdown_bounds(returns):
    for c in returns.columns:
        mdd = M.max_drawdown(returns[c])
        assert -1.0 <= mdd <= 0.0


def test_drawdown_table_depth_matches_max(returns):
    r = returns["A5"]
    tbl = M.drawdown_table(r)
    assert tbl["Depth"].min() == pytest.approx(M.max_drawdown(r))


def test_cvar_at_least_var(returns):
    r = returns["A3"]
    assert M.cvar_historical(r) >= M.var_historical(r) - 1e-12


def test_capture_ratios_self_is_one(returns):
    r = returns["A2"]
    up, down = M.capture_ratios(r, r)
    assert up == pytest.approx(1.0)
    assert down == pytest.approx(1.0)


def test_beta_of_self_is_one(returns):
    b, a = M.beta_alpha(returns["A1"], returns["A1"], rf=0.0)
    assert b == pytest.approx(1.0)
    assert a == pytest.approx(0.0, abs=1e-12)


def test_money_weighted_return_simple():
    # invest 100, receive 110 one year later -> 10% IRR
    idx = pd.to_datetime(["2020-01-01", "2021-01-01"])
    flows = pd.Series([-100.0, 110.0], index=idx)
    assert M.money_weighted_return(flows) == pytest.approx(0.10, abs=2e-3)


def test_annual_returns_compound(returns):
    r = returns["A0"]
    yr = M.annual_returns(r)
    assert (1 + yr).prod() == pytest.approx(float((1 + r).prod()), rel=1e-9)


def test_summary_has_benchmark_rows(returns):
    s = M.summary(returns["A0"], bench=returns["A1"])
    assert "Beta" in s.index and "Information Ratio" in s.index
