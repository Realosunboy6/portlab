"""Tests for portlab.costs.

ALL DATA HERE IS SYNTHETIC — hand-computed cost arithmetic, not market
evidence.
"""

import numpy as np
import pandas as pd
import pytest

from portlab import costs as C


# ------------------------------------------------------------------ models
def test_fixed_cost():
    m = C.FixedCost(4.95)
    assert m.cost(10_000.0) == pytest.approx(4.95)
    assert m.cost(-10_000.0) == pytest.approx(4.95)
    assert m.cost(0.0) == 0.0  # no ticket, no charge


def test_proportional_cost():
    m = C.ProportionalCost(10.0)  # 10 bps
    assert m.cost(10_000.0) == pytest.approx(10.0)
    assert m.cost(-10_000.0) == pytest.approx(10.0)


def test_spread_cost_is_half_spread():
    m = C.SpreadCost(20.0)  # 20 bps full spread -> 10 bps crossing
    assert m.cost(10_000.0) == pytest.approx(10.0)


def test_market_impact_sqrt():
    m = C.MarketImpact(coef=1.0, power=0.5)
    # 1% of ADV at 2% daily vol: 1 x 0.02 x sqrt(0.01) x 10_000 = $20
    c = m.cost(10_000.0, shares=100.0, sigma=0.02, adv=10_000.0)
    assert c == pytest.approx(20.0)


def test_market_impact_missing_data_is_zero_not_crash():
    m = C.MarketImpact()
    assert m.cost(10_000.0) == 0.0
    assert m.cost(10_000.0, shares=100.0, sigma=None, adv=10_000.0) == 0.0
    assert m.cost(10_000.0, shares=100.0, sigma=0.02, adv=0.0) == 0.0


def test_combined_cost_sums():
    m = C.CombinedCost([C.FixedCost(1.0), C.ProportionalCost(10.0)])
    assert m.cost(10_000.0) == pytest.approx(11.0)
    assert m.cost(0.0) == 0.0


# ---------------------------------------------------------------- portfolio
def test_portfolio_costs_breakdown():
    trades = {"AAA": 100.0, "BBB": -50.0, "CCC": 0.0, "DDD": 10.0}
    market = {
        "AAA": {"price": 100.0, "sigma": 0.02, "adv": 1_000_000.0},
        "BBB": {"price": 50.0},
        # DDD missing from market -> skipped
    }
    model = C.CombinedCost([C.ProportionalCost(10.0),
                            C.MarketImpact(coef=1.0)])
    df = C.portfolio_costs(trades, market, model)
    assert set(df.index) == {"AAA", "BBB"}
    # AAA: 10bps x 10_000 = $10 + impact 0.02*sqrt(100/1e6)*10_000 = $2
    assert df.loc["AAA", "cost"] == pytest.approx(12.0)
    # BBB: 10bps x 2_500 = $2.50, no sigma -> no impact
    assert df.loc["BBB", "cost"] == pytest.approx(2.50)
    assert df.loc["AAA", "cost_bps"] == pytest.approx(12.0)
    assert (df["cost"] >= 0).all()


def test_turnover_and_penalty():
    tvs = [10_000.0, -5_000.0, 2_500.0]
    assert C.turnover(tvs, 100_000.0) == pytest.approx(0.175)
    assert C.turnover_penalty(tvs, 0.001) == pytest.approx(17.5)
    assert np.isnan(C.turnover(tvs, 0.0))


def test_net_of_costs():
    assert C.net_of_costs(100.0, 12.5) == pytest.approx(87.5)
