"""Tests for leverage, taxes, VPW/guardrails, glide paths, lazy
portfolios, and the attribution charts."""

import numpy as np
import pandas as pd
import pytest

from portlab import glide_weights, monte_carlo, plots
from portlab.backtest import backtest_dynamic, backtest_portfolio, glide_path
from portlab.data.lazy_portfolios import (INCEPTION, LAZY_PORTFOLIOS,
                                          get_lazy_portfolio, lazy_tickers)


# ---------------------------------------------------------------- leverage

def _flat_returns(r=0.01, n=36):
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    return pd.DataFrame({"A": np.full(n, r), "B": np.full(n, r)}, index=idx)


def test_leverage_one_reproduces_baseline(returns):
    base = backtest_portfolio(returns, {"A0": 0.6, "A1": 0.4})
    lev = backtest_portfolio(returns, {"A0": 0.6, "A1": 0.4}, leverage=1.0)
    pd.testing.assert_series_equal(base.balance, lev.balance)
    assert lev.margin_calls == []


def test_two_x_doubles_flat_return():
    rets = _flat_returns(0.01)
    r = backtest_portfolio(rets, {"A": 0.5, "B": 0.5}, leverage=2.0,
                           debt_rate=0.0, rebalance="monthly", periods=12)
    assert np.allclose(r.returns.values, 0.02, atol=1e-10)


def test_debt_rate_drags_returns():
    rets = _flat_returns(0.01)
    free = backtest_portfolio(rets, {"A": 1.0}, leverage=2.0, debt_rate=0.0,
                              rebalance="monthly", periods=12)
    costly = backtest_portfolio(rets, {"A": 1.0}, leverage=2.0, debt_rate=0.06,
                                rebalance="monthly", periods=12)
    drag = free.returns.iloc[1] - costly.returns.iloc[1]
    assert drag == pytest.approx(1.06 ** (1 / 12) - 1, rel=1e-3)


def test_levered_ruin_sticks():
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    rets = pd.DataFrame({"A": [0.0, -0.6, 0.5, 0.5, 0.5, 0.5]}, index=idx)
    r = backtest_portfolio(rets, {"A": 1.0}, leverage=2.0, rebalance="never",
                           periods=12)
    assert r.balance.iloc[1] == 0.0
    assert (r.balance.iloc[1:] == 0.0).all()


def test_maintenance_margin_prevents_ruin():
    idx = pd.date_range("2020-01-31", periods=8, freq="ME")
    rets = pd.DataFrame({"A": [0.0, -0.3, -0.3, 0.1, 0.1, 0.1, 0.1, 0.1]}, index=idx)
    r = backtest_portfolio(rets, {"A": 1.0}, leverage=2.0, rebalance="never",
                           maintenance_margin=0.30, periods=12)
    assert len(r.margin_calls) >= 1
    assert (r.balance > 0).all()


def test_debt_rate_series_lookup():
    rets = _flat_returns(0.01)
    tbill = pd.Series(0.05, index=rets.index)
    r = backtest_portfolio(rets, {"A": 1.0}, leverage=2.0, debt_rate=tbill,
                           rebalance="monthly", periods=12)
    assert r.returns.iloc[1] < 0.02          # borrow cost visible


# ---------------------------------------------------------------- taxes

def test_zero_taxes_reproduce_baseline(returns):
    base = backtest_portfolio(returns, {"A0": 0.5, "A7": 0.5}, rebalance="monthly")
    taxed = backtest_portfolio(returns, {"A0": 0.5, "A7": 0.5}, rebalance="monthly",
                               tax_dividend=0.0, tax_capgains=0.0)
    pd.testing.assert_series_equal(base.balance, taxed.balance)


def test_dividend_tax_haircut(returns):
    dy = {"A0": 0.03}
    gross = backtest_portfolio(returns, {"A0": 1.0}, rebalance="never")
    net = backtest_portfolio(returns, {"A0": 1.0}, rebalance="never",
                             tax_dividend=0.30, div_yield=dy)
    n = len(returns)
    expected_drag = 0.03 * 0.30            # annual
    realized_drag = (gross.balance.iloc[-1] / net.balance.iloc[-1]) ** (252 / n) - 1
    assert realized_drag == pytest.approx(expected_drag, rel=0.05)
    assert net.taxes_paid.sum() > 0


def test_no_capgains_tax_without_sales(returns):
    r = backtest_portfolio(returns, {"A0": 1.0}, rebalance="never",
                           tax_capgains=0.20)
    assert r.taxes_paid.sum() == 0.0


def test_capgains_tax_on_rebalance():
    idx = pd.date_range("2020-01-31", periods=14, freq="ME")
    a = np.zeros(14); a[1:13] = 0.05                     # A rallies
    rets = pd.DataFrame({"A": a, "B": np.zeros(14)}, index=idx)
    taxed = backtest_portfolio(rets, {"A": 0.5, "B": 0.5}, rebalance="annually",
                               tax_capgains=0.25, periods=12)
    free = backtest_portfolio(rets, {"A": 0.5, "B": 0.5}, rebalance="annually",
                              periods=12)
    assert taxed.taxes_paid.sum() > 0
    assert taxed.balance.iloc[-1] < free.balance.iloc[-1]


def test_withdrawal_realizes_gains():
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    rets = pd.DataFrame({"A": np.full(24, 0.02)}, index=idx)
    r = backtest_portfolio(rets, {"A": 1.0}, cashflow=-200, periods=12,
                           rebalance="never", tax_capgains=0.20)
    assert r.taxes_paid.sum() > 0


# ---------------------------------------------------------------- glide paths

def test_glide_path_endpoints_and_midpoint():
    sched = glide_path({"VTI": 0.9, "BND": 0.1}, {"VTI": 0.3, "BND": 0.7},
                       "2000-01-01", "2020-01-01", steps=5)
    dates = sorted(sched)
    assert sched[dates[0]]["VTI"] == pytest.approx(0.9)
    assert sched[dates[-1]]["VTI"] == pytest.approx(0.3)
    assert sched[dates[2]]["VTI"] == pytest.approx(0.6)


def test_glide_path_through_dynamic_backtest(returns):
    sched = glide_path({"A0": 1.0}, {"A1": 1.0},
                       str(returns.index[0].date()), str(returns.index[-1].date()),
                       steps=4)
    r = backtest_dynamic(returns, sched)
    assert r.weights.iloc[0]["A0"] == pytest.approx(1.0)
    assert r.weights.iloc[-1]["A1"] == pytest.approx(1.0)


def test_mc_glide_weights_reduce_risk(mu_cov):
    mu, cov = mu_cov
    ws = glide_weights({"A7": 1.0}, {"A0": 1.0}, years=10)  # A7 vol >> A0 vol
    res = monte_carlo(years=10, n_sims=800, model="statistical",
                      asset_mu=mu, asset_cov=cov, weights_schedule=ws, freq=12)
    yearly = res.balances[:, 1:].reshape(800, 10, 12)
    growth = yearly[:, :, -1] / np.maximum(yearly[:, :, 0], 1e-9)
    dispersion = growth.std(axis=0)
    assert dispersion[-1] < dispersion[0]      # de-risking shows up


def test_mc_per_year_mean_sequence():
    res = monte_carlo(years=4, n_sims=500, model="normal",
                      mean_annual=[0.20, 0.10, 0.0, -0.10], vol_annual=0.01,
                      seed=3)
    med = np.median(res.balances, axis=0)
    yr_growth = [med[(k + 1) * 12] / med[k * 12] for k in range(4)]
    assert yr_growth[0] > yr_growth[1] > yr_growth[2] > yr_growth[3]


# ---------------------------------------------------------------- withdrawal rules

def test_vpw_depletes_by_target_age():
    res = monte_carlo(initial=1_000_000, years=35, n_sims=50, model="normal",
                      mean_annual=0.0, vol_annual=1e-6, inflation_annual=0.0,
                      withdrawal="vpw", current_age=65, vpw_depletion_age=100,
                      vpw_stock_return=0.0, vpw_bond_return=0.0, freq=12)
    # spends down substantially by the depletion age, but by construction the
    # 10% cap means it glides low without ever hitting zero
    assert res.balances[:, -1].max() < 1_000_000 * 0.12
    assert res.success_rate == 1.0
    # withdrawals rise as the horizon shrinks (annuity property)
    annual = res.withdrawals.reshape(50, 35, 12).sum(axis=2)
    assert annual[0, 5] > annual[0, 0] * 0.9 and annual[0, 0] > 0


def test_vpw_rate_matches_table():
    from portlab.montecarlo import _vpw_pct
    pct = _vpw_pct(65, 100, 0.6 * 0.05 + 0.4 * 0.019, 0.10)
    assert 0.043 <= pct <= 0.055          # Bogleheads table: ~4.8% at 65, 60/40


def test_guardrails_cut_after_crash():
    # deterministic crash path: -40% year 1, flat after
    res = monte_carlo(initial=1_000_000, years=20, n_sims=10, model="normal",
                      mean_annual=[-0.40] + [0.0] * 19, vol_annual=1e-6,
                      inflation_annual=0.0, withdrawal="guardrails",
                      withdrawal_pct=0.05, freq=12, seed=1)
    annual_wd = res.withdrawals.reshape(10, 20, 12).sum(axis=2)
    assert annual_wd[0, 1] < annual_wd[0, 0]              # cut triggered
    assert annual_wd[0, 1] == pytest.approx(annual_wd[0, 0] * 0.9, rel=1e-6)


def test_guardrails_prosperity_raise():
    res = monte_carlo(initial=1_000_000, years=10, n_sims=10, model="normal",
                      mean_annual=0.30, vol_annual=1e-6, inflation_annual=0.0,
                      withdrawal="guardrails", withdrawal_pct=0.05, freq=12)
    annual_wd = res.withdrawals.reshape(10, 10, 12).sum(axis=2)
    assert annual_wd[0, -1] > annual_wd[0, 0]             # raises over time


def test_guardrails_inflation_cap():
    res = monte_carlo(initial=1_000_000, years=3, n_sims=5, model="normal",
                      mean_annual=0.08, vol_annual=1e-6, inflation_annual=0.10,
                      withdrawal="guardrails", withdrawal_pct=0.04, freq=12)
    annual_wd = res.withdrawals.reshape(5, 3, 12).sum(axis=2)
    growth = annual_wd[0, 1] / annual_wd[0, 0]
    assert growth <= 1.06 * 1.101 ** 0  # increase capped at 6% (+PR possible)


def test_income_stats():
    res = monte_carlo(years=5, n_sims=100, model="normal", withdrawal="vpw",
                      current_age=70)
    stats = res.income_stats()
    assert stats["Median Annual Withdrawal"] > 0


# ---------------------------------------------------------------- lazy portfolios

def test_lazy_portfolios_sum_to_one():
    for name, alloc in LAZY_PORTFOLIOS.items():
        assert sum(alloc.values()) == pytest.approx(1.0, abs=1e-9), name
        assert name in INCEPTION


def test_lazy_lookup_flexible():
    assert get_lazy_portfolio("golden butterfly") == LAZY_PORTFOLIOS["Golden Butterfly"]
    assert get_lazy_portfolio("All Weather") == LAZY_PORTFOLIOS["All Weather (Dalio)"]
    with pytest.raises(KeyError):
        get_lazy_portfolio("Moon Rocket")


def test_lazy_tickers_unique():
    ticks = lazy_tickers()
    assert len(ticks) == len(set(ticks)) and "VTI" in ticks


# ---------------------------------------------------------------- charts

def test_risk_budget_chart(mu_cov):
    mu, cov = mu_cov
    w = pd.Series(1 / len(mu), index=mu.index)
    fig = plots.risk_budget_chart(w, cov)
    assert len(fig.data) == 2
    assert sum(fig.data[1].x) == pytest.approx(1.0, abs=1e-6)


def test_return_contribution_matches_portfolio(returns):
    r = backtest_portfolio(returns, {"A0": 0.5, "A1": 0.5}, rebalance="monthly")
    per_period = (r.weights * returns[r.weights.columns]).sum(axis=1)
    assert np.allclose(per_period.values, r.returns.values, atol=1e-12)
    fig = plots.return_contribution_chart(r.weights, returns)
    assert len(fig.data) == 2
