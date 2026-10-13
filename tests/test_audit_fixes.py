"""Regression tests for the multi-agent audit findings."""

import numpy as np
import pandas as pd
import pytest

from portlab import metrics as M
from portlab.backtest.portfolio import _is_period_boundary, backtest_portfolio
from portlab.optimize import min_cdar
from portlab.rebalance import rebalance_trades


def test_initial_loss_is_a_drawdown():
    # Audit bug 3: -50% in period 1, never recovers -> max drawdown = -50%
    r = pd.Series([-0.5] + [0.0] * 39,
                  index=pd.bdate_range("2020-01-01", periods=40))
    assert M.max_drawdown(r) == pytest.approx(-0.5)


def test_cdar_ignores_zero_mass():
    # Audit bug 2: mostly-rising series; CDaR must average the worst 5%,
    # not the whole series
    r = pd.Series(np.full(200, 0.01),
                  index=pd.bdate_range("2020-01-01", periods=200))
    r.iloc[[50, 120, 180]] = -0.02
    dd = -M.drawdown_series(r)
    k = int(np.ceil(0.05 * len(dd)))
    expected = np.sort(dd.values)[::-1][:k].mean()
    assert M.cdar(r, 0.95) == pytest.approx(expected)
    assert M.cdar(r, 0.95) > 5 * M.pain_index(r)


def test_min_cdar_rejects_early_crash_asset():
    # Audit bug 1: a -50% first-period loss must count as drawdown in the LP
    rx = np.zeros(40); rx[0] = -0.5
    ry = np.tile([0.01, -0.008], 20)
    rets = pd.DataFrame({"X": rx, "Y": ry},
                        index=pd.bdate_range("2020-01-01", periods=40))
    w = min_cdar(rets, alpha=0.9)
    assert w["Y"] > 0.9          # the crash asset must NOT win


def test_annual_boundary_is_january():
    # Audit bug 4
    dec = pd.Timestamp("2020-11-30"), pd.Timestamp("2020-12-31")
    jan = pd.Timestamp("2020-12-31"), pd.Timestamp("2021-01-31")
    assert not _is_period_boundary(*dec, 12)
    assert _is_period_boundary(*jan, 12)
    # quarterly fires Jan/Apr/Jul/Oct
    assert _is_period_boundary(pd.Timestamp("2021-03-31"),
                               pd.Timestamp("2021-04-30"), 3)


def test_annual_rebalance_lands_in_january(returns):
    res = backtest_portfolio(returns, {"A0": 0.5, "A7": 0.5}, rebalance="annually")
    assert all(d.month == 1 for d in res.rebalance_dates)


def test_rebalance_trades_reports_unallocated():
    t = rebalance_trades({"A": 5000, "B": 5000}, {"A": 0.5, "B": 0.5},
                         cash_to_add=100, min_trade=500)
    assert t.attrs["unallocated_cash"] == pytest.approx(100)


def test_tear_sheet_monthly_window_sensible():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0.006, 0.03, 120), name="M",
                  index=pd.date_range("2010-01-31", periods=120, freq="ME"))
    from portlab.report import tear_sheet
    html = tear_sheet(r, periods=12, output=None)
    assert "Rolling Sharpe (36 periods)" in html
