"""Tests for the v0.2.0 additions: parametric risk metrics, HRP, CDaR,
semicovariance, tear sheet, and the rebalance utilities."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sstats

from portlab import metrics as M
from portlab import plots, rebalance_trades
from portlab.covariance import cov_semi, get_cov
from portlab.optimize import frontier, gmv, hrp, min_cdar
from portlab.report import tear_sheet
from portlab.returns import growth_of, log_returns, simple_returns


# --------------------------------------------- A: parametric metrics & variants

def test_parametric_var_matches_closed_form():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.02, 20_000))
    v = M.var_parametric(r, alpha=0.95)
    expected = 0.02 * sstats.norm.ppf(0.95) - 0.001
    assert v == pytest.approx(expected, rel=0.05)
    # normal data: CF adjustment should barely move it
    cf = M.var_parametric(r, alpha=0.95, method="cornish_fisher")
    assert cf == pytest.approx(v, rel=0.05)


def test_cornish_fisher_reacts_to_skew():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 0.01, 20_000)
    skewed = pd.Series(np.where(base < -0.015, base * 3, base))  # fat left tail
    assert M.var_parametric(skewed, method="cornish_fisher") > \
        M.var_parametric(skewed, method="normal")


def test_parametric_cvar_exceeds_var():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0005, 0.012, 5_000))
    assert M.cvar_parametric(r) > M.var_parametric(r)


def test_multi_period_returns():
    prices = pd.Series([100, 110, 121, 133.1],
                       index=pd.date_range("2020-01-31", periods=4, freq="ME"))
    r2 = simple_returns(prices, periods_lag=2)
    assert r2.iloc[0] == pytest.approx(0.21)
    lr = log_returns(prices, periods_lag=2)
    assert lr.iloc[0] == pytest.approx(np.log(1.21))


def test_geometric_mean_consistent_with_cagr(returns):
    r = returns["A0"]
    gm = M.geometric_mean(r)
    assert (1 + gm) ** 252 - 1 == pytest.approx(M.cagr(r), rel=1e-9)


def test_arithmetic_growth_and_drawdown(returns):
    r = returns["A0"]
    add = growth_of(r, geometric=False)
    assert add.iloc[-1] == pytest.approx(1 + r.sum())
    dd = M.drawdown_series(r, geometric=False)
    assert dd.min() >= -1.0 and (dd <= 1e-12).all()


def test_frontier_transition_chart(mu_cov):
    mu, cov = mu_cov
    fr = frontier(mu, cov, n_points=8)
    fig = plots.frontier_transition_chart(fr)
    assert len(fig.data) == len(mu)


# --------------------------------------------- B: metrics expansion

def test_treynor_and_m2_signs(returns):
    r, b = returns["A7"], returns["A0"]
    assert np.isfinite(M.treynor(r, b))
    assert np.isfinite(M.m_squared(r, b))


def test_trade_stats_on_known_series():
    r = pd.Series([0.02, -0.01, 0.02, -0.01, 0.02, -0.01],
                  index=pd.date_range("2020-01-31", periods=6, freq="ME"))
    assert M.win_rate(r) == pytest.approx(0.5)
    assert M.payoff_ratio(r) == pytest.approx(2.0)
    assert M.profit_factor(r) == pytest.approx(2.0)
    assert M.gain_to_pain(r) == pytest.approx(1.0)


def test_best_worst_shape(returns):
    tbl = M.best_worst(returns["A0"])
    assert list(tbl.index) == ["Period (native)", "Month", "Quarter", "Year"]
    assert (tbl["Best"] >= tbl["Worst"]).all()


def test_drawdown_family(returns):
    r = returns["A7"]
    assert M.pain_index(r) > 0
    assert np.isfinite(M.sterling_ratio(r))
    assert np.isfinite(M.burke_ratio(r))
    assert np.isfinite(M.recovery_factor(r))


def test_rolling_helpers(returns):
    r, b = returns["A1"], returns["A0"]
    rs = M.rolling_sharpe(r, 63)
    rv = M.rolling_vol(r, 63)
    rb = M.rolling_beta(r, b, 63)
    assert len(rs.dropna()) == len(r) - 62
    assert (rv.dropna() > 0).all()
    assert np.isfinite(rb.dropna()).all()


def test_monthly_heatmap(returns):
    fig = plots.monthly_heatmap(returns["A0"])
    assert fig.data


# --------------------------------------------- C: optimizers

def test_hrp_prefers_low_vol(returns):
    w = hrp(returns)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    assert (w > 0).all()
    # A0 has the lowest vol in the fixture, A7 the highest
    assert w["A0"] > w["A7"]


def test_hrp_accepts_cov(mu_cov):
    _, cov = mu_cov
    w = hrp(cov, is_cov=True)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


def test_min_cdar_valid_and_effective(returns):
    w = min_cdar(returns, alpha=0.90)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    ew = pd.Series(1 / returns.shape[1], index=returns.columns)
    assert M.cdar(returns @ w, 0.90) <= M.cdar(returns @ ew, 0.90) + 1e-6


def test_cdar_at_least_avg_drawdown(returns):
    r = returns["A5"]
    assert M.cdar(r) >= M.pain_index(r) - 1e-12


def test_semicov_psd_and_smaller(returns):
    semi = cov_semi(returns)
    full = returns.cov()
    assert np.linalg.eigvalsh(semi.values).min() >= -1e-12
    assert np.trace(semi.values) < np.trace(full.values)


def test_l2_gamma_spreads_weights(mu_cov):
    mu, cov = mu_cov
    tight = gmv(mu, cov)
    spread = gmv(mu, cov, gamma=10.0)
    assert float((spread ** 2).sum()) <= float((tight ** 2).sum()) + 1e-9
    n = len(mu)
    assert np.allclose(gmv(mu, cov, gamma=1e6).values, 1 / n, atol=1e-3)


def test_get_cov_semi_method(returns):
    cov = get_cov(returns, method="semi")
    assert cov.shape == (returns.shape[1], returns.shape[1])


# --------------------------------------------- D: tear sheet

def test_tear_sheet_contents(returns, tmp_path):
    out = tmp_path / "ts.html"
    html = tear_sheet(returns["A0"].rename("Demo"), returns["A1"],
                      output=str(out))
    assert out.exists()
    assert html.count('class="plotly-graph-div"') >= 4
    assert "CAGR" in html and "Worst Drawdowns" in html and "Metrics" in html


def test_backtest_tear_sheet_method(returns, tmp_path):
    from portlab.backtest import backtest_portfolio
    res = backtest_portfolio(returns, {"A0": 0.5, "A1": 0.5},
                             benchmark=returns["A2"])
    html = res.tear_sheet(output=None)
    assert "Tear Sheet" in html


# --------------------------------------------- E: investor utilities

def test_rebalance_trades_balances():
    t = rebalance_trades({"A": 8000, "B": 1000}, {"A": 0.5, "B": 0.5},
                         cash_to_add=1000)
    assert t["target"].sum() == pytest.approx(10_000, abs=0.02)
    assert t.loc["A", "action"] == "SELL" and t.loc["B", "action"] == "BUY"
    assert t["trade"].sum() == pytest.approx(1000, abs=0.02)  # net = new cash


def test_rebalance_min_trade_suppression():
    t = rebalance_trades({"A": 5001, "B": 4999}, {"A": 0.5, "B": 0.5},
                         min_trade=10)
    assert (t["action"] == "HOLD").all()


def test_rebalance_tax_aware_orders_losses_first():
    t = rebalance_trades({"A": 6000, "B": 6000, "C": 0},
                         {"A": 0.2, "B": 0.2, "C": 0.6},
                         basis={"A": 7000, "B": 3000}, tax_aware=True)
    sells = t[t["action"] == "SELL"]
    assert list(sells.index)[0] == "A"          # loss harvested first
    assert sells.loc["A", "est_realized_gain"] < 0


def test_convert_currency_identity(prices):
    from portlab.data.prices import convert_currency
    out = convert_currency(prices, "USD", "USD")
    pd.testing.assert_frame_equal(out, prices)
