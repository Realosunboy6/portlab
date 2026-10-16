"""Tests for the long-history, screener, attribution and tactical features."""

import numpy as np
import pandas as pd
import pytest

from portlab import analytics, tactical
from portlab.backtest import backtest_dynamic, backtest_portfolio
from portlab.data import asset_classes as ac
from portlab.factor import attribution, match_exposure
from portlab.optimize import geometric_frontier


# ---------------------------------------------------------------- dynamic backtest

def test_dynamic_matches_static_for_single_entry(returns):
    sched = {returns.index[0]: {"A0": 0.5, "A1": 0.5}}
    dyn = backtest_dynamic(returns, sched)
    stat = backtest_portfolio(returns, {"A0": 0.5, "A1": 0.5}, rebalance="never")
    assert dyn.balance.iloc[-1] == pytest.approx(stat.balance.iloc[-1], rel=1e-9)


def test_dynamic_switches_allocation(returns):
    mid = returns.index[len(returns) // 2]
    sched = {returns.index[0]: {"A0": 1.0}, mid: {"A1": 1.0}}
    r = backtest_dynamic(returns, sched)
    assert r.weights.loc[:mid, "A0"].iloc[0] == pytest.approx(1.0)
    assert r.weights.iloc[-1]["A1"] == pytest.approx(1.0)
    assert len(r.rebalance_dates) == 2


# ---------------------------------------------------------------- PCA

def test_pca_explained_variance(returns):
    load = analytics.pca(returns, n_components=3)
    ev = load.attrs["explained_variance"]
    assert load.shape == (returns.shape[1], 3)
    assert (ev.diff().dropna() <= 1e-12).all()      # descending
    assert 0 < ev.sum() <= 1.0 + 1e-9
    # with common correlation 0.3, PC1 should dominate
    assert ev.iloc[0] > 1.0 / returns.shape[1]


# ---------------------------------------------------------------- screener

def test_screener_filters_and_sorts(returns):
    tbl = analytics.screener(returns, filters={"CAGR": (None, None)},
                             sort_by="Annualized Volatility", ascending=True)
    assert (tbl["Annualized Volatility"].diff().dropna() >= -1e-12).all()
    strict = analytics.screener(returns, filters={"Sharpe Ratio": (0.0, None)})
    assert (strict["Sharpe Ratio"] >= 0).all()


def test_screener_unknown_metric(returns):
    with pytest.raises(ValueError):
        analytics.screener(returns, filters={"Nope": (0, 1)})


# ---------------------------------------------------------------- attribution / match

@pytest.fixture(scope="module")
def factors(rng):
    idx = pd.date_range("2010-01-31", periods=180, freq="ME")
    return pd.DataFrame({
        "Mkt-RF": rng.normal(0.006, 0.045, 180),
        "SMB": rng.normal(0.001, 0.025, 180),
        "HML": rng.normal(0.001, 0.028, 180),
        "RF": np.full(180, 0.002)}, index=idx)


def test_attribution_sums_to_total(factors, rng):
    ret = factors["RF"] + 0.002 + 1.1 * factors["Mkt-RF"] \
        + rng.normal(0, 0.004, len(factors))
    contrib = attribution(pd.Series(ret, index=factors.index), factors, "capm")
    total = contrib.attrs["total"]
    assert total.sum() == pytest.approx(ret.sum(), rel=1e-9)


def test_match_exposure_recovers_mix(factors, rng):
    a = factors["RF"] + 1.0 * factors["Mkt-RF"] + rng.normal(0, 0.002, len(factors))
    b = factors["RF"] + 0.5 * factors["SMB"] + rng.normal(0, 0.002, len(factors))
    target = 0.7 * a + 0.3 * b
    cands = pd.DataFrame({"A": a, "B": b}, index=factors.index)
    res = match_exposure(pd.Series(target, index=factors.index), cands,
                         factors=factors, model="ff3")
    assert res["weights"]["A"] == pytest.approx(0.7, abs=0.05)
    assert res["tracking_error"] < 0.02
    assert "loadings" in res


# ---------------------------------------------------------------- adaptive allocation

def test_adaptive_allocation_holds_top_n(prices, returns):
    w = tactical.adaptive_allocation(prices, returns, lookback=63,
                                     top_n=3, vol_window=42)
    active = w.loc[(w.sum(axis=1) > 0)]
    assert ((active > 1e-9).sum(axis=1) <= 3).all()
    assert (active.sum(axis=1) <= 1.0 + 1e-9).all()


def test_adaptive_allocation_cash_completes(prices, returns):
    w = tactical.adaptive_allocation(prices, returns, lookback=63, top_n=2,
                                     vol_window=42, out_asset="CASH")
    active = w.loc[w.drop(columns="CASH").sum(axis=1) > 0]
    assert np.allclose(active.sum(axis=1), 1.0)


# ---------------------------------------------------------------- ISSUE-003
def test_adaptive_allocation_short_returns_calendar(prices, returns):
    """ISSUE-003: returns (N-1 rows, from pct_change) vs prices (N rows).

    The rebalance loop indexes vol positionally over the price calendar; with
    rebalance_every=1 the loop reaches the final row, which used to raise
    IndexError because vol was one row shorter than prices.
    """
    short_rets = returns.iloc[1:]  # mimic prices.pct_change().dropna()
    assert len(short_rets) == len(prices) - 1
    w = tactical.adaptive_allocation(prices, short_rets, lookback=63, top_n=2,
                                     vol_window=42, out_asset="CASH",
                                     rebalance_every=1)
    assert len(w) == len(prices)
    active = w.loc[w.drop(columns="CASH").sum(axis=1) > 0]
    assert np.allclose(active.sum(axis=1), 1.0)


# ---------------------------------------------------------------- geometric frontier

def test_geometric_frontier_monotone_vol(returns):
    fr = geometric_frontier(returns, n_points=6)
    assert len(fr) >= 3
    assert (fr["realized_vol"].diff().dropna() >= -0.02).all()
    # geometric mean should never exceed arithmetic-implied plausibility
    assert fr["geometric_mean"].max() < 0.5


# ---------------------------------------------------------------- asset classes

@pytest.fixture()
def stub_macro(monkeypatch):
    def fake_fred(series, start="1950-01-01", **kw):
        idx = pd.date_range("1960-01-01", "2024-12-01", freq="MS")
        rng = np.random.default_rng(len(series))
        if series.startswith("GOLD"):
            return pd.Series(35 * np.exp(np.cumsum(rng.normal(0.004, 0.04, len(idx)))),
                             index=idx, name=series)
        base = {"GS20": 5.5, "GS10": 5.0, "GS5": 4.5, "GS1": 4.0, "TB3MS": 3.5}[series]
        return pd.Series(base + np.cumsum(rng.normal(0, 0.08, len(idx))).clip(-3, 6),
                         index=idx, name=series)

    def fake_ff(model="ff3", freq="monthly", start="1963-07-01", **kw):
        idx = pd.date_range("1926-07-31", "2024-12-31", freq="ME")
        rng = np.random.default_rng(1)
        return pd.DataFrame({"Mkt-RF": rng.normal(0.006, 0.05, len(idx)),
                             "SMB": rng.normal(0.001, 0.03, len(idx)),
                             "HML": rng.normal(0.002, 0.03, len(idx)),
                             "RF": np.full(len(idx), 0.003)}, index=idx)

    def fake_6port(start):
        idx = pd.date_range("1926-07-31", "2024-12-31", freq="ME")
        rng = np.random.default_rng(2)
        cols = ["SMALL LoBM", "ME1 BM2", "SMALL HiBM", "BIG LoBM", "ME2 BM2", "BIG HiBM"]
        raw = pd.DataFrame({c: rng.normal(0.008, 0.06, len(idx)) for c in cols}, index=idx)
        return pd.DataFrame({
            "US Large Cap Growth": raw["BIG LoBM"],
            "US Large Cap Value": raw["BIG HiBM"],
            "US Small Cap Growth": raw["SMALL LoBM"],
            "US Small Cap Value": raw["SMALL HiBM"],
            "US Small Cap": raw[["SMALL LoBM", "ME1 BM2", "SMALL HiBM"]].mean(axis=1)})

    import portlab.data.factors as f
    import portlab.data.macro as m
    monkeypatch.setattr(m, "get_fred", fake_fred)
    monkeypatch.setattr(f, "get_ff_factors", fake_ff)
    monkeypatch.setattr(ac, "_french_size_value", fake_6port)


def test_synthetic_treasury_math():
    y = pd.Series([5.0, 5.0, 4.0], index=pd.date_range("2020-01-31", periods=3, freq="ME"))
    r = ac.synthetic_treasury_returns(y, duration=8.0)
    # flat month: carry only; falling-yield month: carry + duration gain
    assert r.iloc[0] == pytest.approx(0.05 / 12)
    assert r.iloc[1] == pytest.approx(0.05 / 12 + 8.0 * 0.01)


def test_asset_class_returns_long_history(stub_macro):
    df = ac.get_asset_class_returns(["US Stock Market", "Long Term Treasury",
                                     "Cash (T-Bills)", "Gold"], start="1972-01-01")
    assert list(df.columns) == ["US Stock Market", "Long Term Treasury",
                                "Cash (T-Bills)", "Gold"]
    assert df.index.min().year == 1972
    assert df.index.max().year >= 2024
    assert df["Cash (T-Bills)"].dropna().between(-0.01, 0.02).all()


def test_asset_class_unknown_raises(stub_macro):
    with pytest.raises(ValueError):
        ac.get_asset_class_returns(["Moon Rocks"])
