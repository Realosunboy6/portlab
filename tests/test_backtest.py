import numpy as np
import pandas as pd
import pytest

from portlab.backtest import (RollingBacktest, backtest_portfolio,
                              compare_portfolios, equal_weight, worst_windows)


def test_single_asset_reproduces_buy_and_hold(returns):
    r = backtest_portfolio(returns, {"A0": 1.0}, initial=10_000,
                           cashflow=0.0, rebalance="never")
    expected = 10_000 * float((1 + returns["A0"]).prod())
    assert r.balance.iloc[-1] == pytest.approx(expected, rel=1e-9)
    pd.testing.assert_series_equal(r.returns, returns["A0"],
                                   check_names=False, rtol=1e-9)


def test_weights_sum_to_one(returns):
    r = backtest_portfolio(returns, {"A0": 0.6, "A1": 0.4}, rebalance="monthly")
    assert np.allclose(r.weights.sum(axis=1), 1.0)


def test_never_rebalance_drifts(returns):
    r = backtest_portfolio(returns, {"A0": 0.5, "A7": 0.5}, rebalance="never")
    # weights should drift away from 50/50 with different asset paths
    assert abs(r.weights.iloc[-1, 0] - 0.5) > 0.01
    assert len(r.rebalance_dates) == 0


def test_annual_rebalance_happens(returns):
    r = backtest_portfolio(returns, {"A0": 0.5, "A7": 0.5}, rebalance="annually")
    assert 1 <= len(r.rebalance_dates) <= 3  # ~3y of data


def test_band_rebalancing(returns):
    r = backtest_portfolio(returns, {"A0": 0.5, "A7": 0.5},
                           rebalance="bands", rebalance_band=0.02)
    # after any rebalance, weights snap back to target
    if r.rebalance_dates:
        d = r.rebalance_dates[0]
        assert r.weights.loc[d, "A0"] == pytest.approx(0.5, abs=1e-9)


def test_contributions_grow_balance(returns):
    base = backtest_portfolio(returns, {"A0": 1.0})
    plus = backtest_portfolio(returns, {"A0": 1.0}, cashflow=500,
                              cashflow_freq="monthly")
    assert plus.balance.iloc[-1] > base.balance.iloc[-1]
    # time-weighted returns unaffected by external flows
    assert plus.summary().loc["CAGR"].iloc[0] == pytest.approx(
        base.summary().loc["CAGR"].iloc[0], rel=1e-9)


def test_withdrawals_can_ruin(returns):
    r = backtest_portfolio(returns, {"A0": 1.0}, initial=1_000,
                           cashflow=-800, cashflow_freq="monthly")
    assert r.balance.iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_summary_ending_balance(returns):
    r = backtest_portfolio(returns, {"A0": 0.5, "A1": 0.5})
    s = r.summary()
    assert s.loc["Ending Balance"].iloc[0] == pytest.approx(r.balance.iloc[-1])


def test_compare_portfolios(returns):
    table, results = compare_portfolios(
        returns, {"AllA0": {"A0": 1.0}, "Mix": {"A0": 0.5, "A1": 0.5}})
    assert table.shape[1] == 2 and "AllA0" in table.columns


def test_rolling_backtest_out_of_sample_only(returns):
    rb = RollingBacktest(returns, train_window=252, rebalance_every=21)
    out = rb.run(equal_weight, name="ew")
    # OOS series starts after the first training window
    assert out.index[0] >= returns.index[252]
    ew_manual = returns.iloc[252:][returns.columns].mean(axis=1)
    common = out.index.intersection(ew_manual.index)
    assert np.allclose(out.loc[common, "ew"], ew_manual.loc[common], atol=1e-12)


def test_rolling_transaction_costs_reduce_returns(returns):
    def churner(train):
        # alternating concentrated bets -> high turnover
        rng = np.random.default_rng(len(train))
        w = pd.Series(0.0, index=train.columns)
        w.iloc[rng.integers(0, len(w))] = 1.0
        return w

    free = RollingBacktest(returns, tc_bps=0).run(churner)
    costly = RollingBacktest(returns, tc_bps=50).run(churner)
    assert costly.iloc[:, 0].sum() < free.iloc[:, 0].sum()


def test_worst_windows_sorted(returns):
    ww = worst_windows(returns["A7"], window=21, top=5)
    assert (ww["Return"].diff().dropna() >= -1e-12).all()
