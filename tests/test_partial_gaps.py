"""PV partial-gap completions: financial goals, factor statistics, manager
performance, fund rankings, fund performance vs benchmark, asset-class
correlations. All tests use deterministic synthetic data — no network."""

import numpy as np
import pandas as pd
import pytest

from portlab import analytics, factor
from portlab.data.asset_classes import ASSET_CLASSES, get_asset_class_returns
from portlab.montecarlo import MonteCarloResult, goal_probability, monte_carlo


def test_goal_probability_unit():
    # SYNTHETIC: two paths, one reaches the target.
    res = MonteCarloResult(balances=np.array([[100.0, 150.0], [100.0, 90.0]]),
                           withdrawals=np.zeros((2, 1)))
    gp = goal_probability(res, 100.0)
    assert gp["probability"] == pytest.approx(0.5)
    assert gp["median_ending"] == pytest.approx(120.0)
    assert gp["p10_ending"] == pytest.approx(96.0)
    assert gp["n_sims"] == 2


def test_goal_probability_deterministic_mc():
    # SYNTHETIC: zero returns + fixed contributions -> exact ending balance.
    n = 120
    hist = pd.Series(np.zeros(n),
                     index=pd.date_range("2020-01-31", periods=n, freq="ME"))
    res = monte_carlo(initial=0.0, years=10, n_sims=20, model="bootstrap",
                      hist_returns=hist, withdrawal="none",
                      contribution_annual=12_000.0, inflation_annual=0.0,
                      seed=1)
    assert goal_probability(res, 120_000.0)["probability"] == pytest.approx(1.0)
    assert goal_probability(res, 120_000.01)["probability"] == pytest.approx(0.0)


def test_factor_statistics():
    # SYNTHETIC factors: known correlation structure and premia.
    rng = np.random.default_rng(11)
    n = 120
    idx = pd.date_range("2016-01-31", periods=n, freq="ME")
    mkt = rng.normal(0.006, 0.04, n)
    smb = 0.3 * mkt + rng.normal(0.001, 0.02, n)
    fac = pd.DataFrame({"Mkt-RF": mkt, "SMB": smb, "HML": rng.normal(0.002, 0.03, n)},
                       index=idx)
    fs = factor.factor_statistics(fac, model="ff3")
    corr = fs["correlation"]
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr.values), 1.0)
    assert np.allclose(corr.values, corr.values.T)
    assert corr.loc["Mkt-RF", "SMB"] > 0.2  # constructed correlation
    full = fs["premia"]["Full sample"]
    assert full.loc["Mkt-RF", "Ann. premium"] == pytest.approx(mkt.mean() * 12)
    assert full.loc["Mkt-RF", "Obs"] == 120
    assert fs["premia"]["Trailing 60M"].loc["SMB", "Obs"] == 60
    assert fs["n_obs"] == 120


def test_manager_performance_ranking():
    # SYNTHETIC: fund C has a built-in +0.8%/mo alpha; it must rank first.
    rng = np.random.default_rng(21)
    n = 120
    idx = pd.date_range("2016-01-31", periods=n, freq="ME")
    mkt = rng.normal(0.006, 0.04, n)
    fac = pd.DataFrame({"Mkt-RF": mkt, "SMB": rng.normal(0, 0.02, n),
                        "HML": rng.normal(0, 0.02, n), "RF": 0.001}, index=idx)
    noise = rng.normal(0, 0.01, (n, 3))
    rets = pd.DataFrame(
        {"A": mkt + noise[:, 0],
         "B": 0.9 * mkt + noise[:, 1],
         "C": 0.008 + 1.1 * mkt + noise[:, 2]},  # true alpha 0.8%/mo
        index=idx)
    tbl = factor.compare_funds(rets, fac, model="ff3", newey_west=False)
    ranked = tbl.sort_values("alpha (annualized)", ascending=False)
    assert list(ranked.index)[0] == "C"
    assert ranked.loc["C", "alpha (annualized)"] > ranked.loc["A", "alpha (annualized)"]


def test_rank_within_category_engine():
    # SYNTHETIC: screener must sort a category subset by the ranking metric.
    rng = np.random.default_rng(31)
    n = 120
    idx = pd.date_range("2016-01-31", periods=n, freq="ME")
    cols = [f"F{i}" for i in range(5)]
    rets = pd.DataFrame(
        {c: rng.normal(0.001 + 0.005 * i, 0.02, n) for i, c in enumerate(cols)},
        index=idx)
    tbl = analytics.screener(rets, periods=12, sort_by="Sharpe Ratio",
                             top=None, filters=None)
    assert list(tbl.index) == ["F4", "F3", "F2", "F1", "F0"]
    tbl["Rank"] = range(1, len(tbl) + 1)  # page-level ranking step
    assert tbl.loc["F4", "Rank"] == 1


def test_fund_performance_vs_benchmark():
    # SYNTHETIC: fund is exactly 1.5x the benchmark -> beta 1.5, alpha 0.
    n = 60
    idx = pd.date_range("2021-01-31", periods=n, freq="ME")
    bench = pd.Series(np.linspace(-0.02, 0.03, n), index=idx, name="BENCH")
    rets = pd.DataFrame({"FUND": 1.5 * bench.values}, index=idx)
    # rf=0: a pure 1.5x levered fund then has exactly beta 1.5, alpha 0
    tbl = analytics.performance_table(rets, rf=0.0, bench=bench, periods=12)
    assert tbl.loc["Beta", "FUND"] == pytest.approx(1.5)
    assert tbl.loc["Annualized Alpha", "FUND"] == pytest.approx(0.0, abs=1e-9)


def test_asset_class_list_and_unknown():
    assert len(ASSET_CLASSES) == 12
    with pytest.raises(ValueError):
        get_asset_class_returns(["Not A Class"])


def test_asset_class_correlation_path():
    # SYNTHETIC monthly class returns -> correlation matrix properties.
    rng = np.random.default_rng(41)
    n = 120
    idx = pd.date_range("2016-01-31", periods=n, freq="ME")
    ac = pd.DataFrame(rng.normal(0.004, 0.03, (n, 4)), index=idx,
                      columns=["US Stock Market", "Gold", "Cash (T-Bills)",
                               "10-Year Treasury"])
    corr = analytics.correlation_matrix(ac)
    assert corr.shape == (4, 4)
    assert np.allclose(np.diag(corr.values), 1.0)
    assert np.allclose(corr.values, corr.values.T, equal_nan=True)
    pt = analytics.performance_table(ac, periods=12)
    assert set(pt.columns) == set(ac.columns)
