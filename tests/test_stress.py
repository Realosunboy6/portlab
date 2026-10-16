"""Tests for portlab.stress.

ALL DATA HERE IS SYNTHETIC except the external-verification step (run
separately, not in this file): scenario shocks for the 2020 COVID window were
cross-checked against an independent pandas extraction from live Yahoo data.
"""

import numpy as np
import pandas as pd
import pytest

from portlab import stress as ST


def _prices():
    idx = pd.bdate_range("2020-01-02", periods=20)
    return pd.DataFrame(
        {"AAA": np.linspace(100.0, 110.0, 20),   # +10% over the frame
         "BBB": np.linspace(200.0, 180.0, 20)},  # -10% over the frame
        index=idx)


def test_scenario_shocks_hand_computed():
    shocks = ST.scenario_shocks(_prices(), "2020-01-02", "2020-01-29")
    assert shocks["AAA"] == pytest.approx(0.10)
    assert shocks["BBB"] == pytest.approx(-0.10)


def test_scenario_shocks_weekend_bounds():
    # bounds on non-trading days snap to the last available close:
    # start 2020-01-04 (Sat) -> Fri 2020-01-03, end 2020-02-01 (Sat) -> Wed 2020-01-29
    px = _prices()
    expected = px["AAA"].iloc[-1] / px["AAA"].iloc[1] - 1.0
    shocks = ST.scenario_shocks(px, "2020-01-04", "2020-02-01")
    assert shocks["AAA"] == pytest.approx(expected)
    assert shocks["BBB"] == pytest.approx(px["BBB"].iloc[-1] / px["BBB"].iloc[1] - 1.0)


def test_scenario_shocks_bad_window_raises():
    with pytest.raises(ValueError):
        ST.scenario_shocks(_prices(), "2020-01-29", "2020-01-02")


def test_apply_scenario_hand_computed():
    # 10 sh AAA @ $110 = $1100, shock +10% -> +$110
    # 5 sh BBB @ $180 = $900,  shock -10% -> -$90
    holdings = {"AAA": 10.0, "BBB": 5.0}
    prices = {"AAA": 110.0, "BBB": 180.0}
    shocks = {"AAA": 0.10, "BBB": -0.10}
    df = ST.apply_scenario(holdings, prices, shocks)
    assert df.loc["AAA", "pnl"] == pytest.approx(110.0)
    assert df.loc["BBB", "pnl"] == pytest.approx(-90.0)
    assert df["pnl"].sum() == pytest.approx(20.0)
    assert df.loc["AAA", "pnl_pct_of_portfolio"] == pytest.approx(110.0 / 2000.0)


def test_apply_scenario_missing_shock_is_zero():
    df = ST.apply_scenario({"AAA": 10.0}, {"AAA": 110.0}, {})
    assert df.loc["AAA", "pnl"] == 0.0


def test_stress_report_multiple_scenarios():
    holdings = {"AAA": 10.0}
    prices = {"AAA": 100.0}
    out = ST.stress_report(holdings, prices,
                           {"down 10%": {"AAA": -0.10},
                            "up 5%": {"AAA": 0.05}})
    assert out["down 10%"]["pnl"] == pytest.approx(-100.0)
    assert out["down 10%"]["portfolio_return"] == pytest.approx(-0.10)
    assert out["up 5%"]["shocked_value"] == pytest.approx(1050.0)


def test_worst_window_finds_planted_crash():
    # single asset: +1%/day everywhere except days 10-14 at -5%/day
    rets = np.full(30, 0.01)
    rets[10:15] = -0.05
    px = 100.0 * np.cumprod(np.concatenate([[1.0], 1.0 + rets]))
    idx = pd.bdate_range("2021-01-01", periods=31)
    prices = pd.DataFrame({"AAA": px}, index=idx)
    res = ST.worst_window(prices, {"AAA": 1.0}, window_days=5)
    # the five -5% price returns sit at idx[11]..idx[15]
    assert res["start"] == idx[11].date().isoformat()
    assert res["end"] == idx[15].date().isoformat()
    assert res["window_return"] == pytest.approx(0.95 ** 5 - 1.0)
    # the returned shocks must reproduce the window return on the portfolio
    assert res["shocks"]["AAA"] == pytest.approx(0.95 ** 5 - 1.0)


def test_worst_window_needs_enough_history():
    with pytest.raises(ValueError, match="only"):
        ST.worst_window(_prices(), {"AAA": 1.0}, window_days=500)


def test_uniform_shock_for_loss():
    assert ST.uniform_shock_for_loss(-0.20) == pytest.approx(-0.20)
    with pytest.raises(ValueError):
        ST.uniform_shock_for_loss(0.10)
    with pytest.raises(ValueError):
        ST.uniform_shock_for_loss(-1.5)


def test_crises_have_sane_windows():
    for name, (s, e) in ST.CRISES.items():
        assert pd.Timestamp(e) > pd.Timestamp(s), name
    assert set(ST.CRISES) == {"GFC 2008", "COVID crash 2020", "Rate shock 2022"}


def test_scenario_shocks_tz_aware_index():
    """REGRESSION (found 2026-10-10 via live Yahoo data): yfinance returns
    tz-aware indexes; tz-naive bounds must not raise."""
    px = _prices()
    px.index = px.index.tz_localize("America/New_York")
    shocks = ST.scenario_shocks(px, "2020-01-02", "2020-01-29")
    assert shocks["AAA"] == pytest.approx(0.10)
    assert shocks["BBB"] == pytest.approx(-0.10)
