"""Tests for portlab.risk (RiskLens).

ALL DATA HERE IS SYNTHETIC — these are statistical correctness checks on
known distributions, not market evidence. Seeded RNGs keep them deterministic.
"""

import datetime as dt
import warnings

import numpy as np
import pandas as pd
import pytest

from portlab.risk import var as V
from portlab.risk import backtest as B
from portlab.risk import tail as T


@pytest.fixture()
def rng():
    return np.random.default_rng(20261009)


@pytest.fixture()
def normal_returns(rng):
    """SYNTHETIC: 2000 iid N(0.0004, 0.01) daily returns."""
    idx = pd.bdate_range("2020-01-01", periods=2000)
    return pd.Series(rng.normal(0.0004, 0.01, 2000), index=idx)


@pytest.fixture()
def t_returns(rng):
    """SYNTHETIC: 20000 Student-t(df=4) returns (fat tails, tail index 1/4)."""
    idx = pd.bdate_range("2020-01-01", periods=20000)
    return pd.Series(rng.standard_t(4, 20000) * 0.01, index=idx)


# ---------------------------------------------------------------- VaR

def test_historical_var_matches_parametric_on_normal(normal_returns):
    """SYNTHETIC: on normal data, historical and parametric VaR must agree."""
    h = V.historical_var(normal_returns, alpha=0.05, window=500).dropna()
    p = V.parametric_var(normal_returns, alpha=0.05, window=500).dropna()
    # Loose: sampling noise on 500-obs windows is ~5-10% of VaR.
    np.testing.assert_allclose(h.iloc[-100:].to_numpy(),
                               p.iloc[-100:].to_numpy(), rtol=0.15)
def test_expected_shortfall_exceeds_var(normal_returns):
    """ES >= VaR by construction (mean of the tail beyond the quantile)."""
    es = V.expected_shortfall(normal_returns, alpha=0.05, window=500).dropna()
    hv = V.historical_var(normal_returns, alpha=0.05, window=500).dropna()
    assert (es >= hv - 1e-12).all()
def test_var_series_aligned_and_no_lookahead(normal_returns):
    """VaR at t uses only data strictly before t (expanding spot check)."""
    out = V.historical_var(normal_returns, alpha=0.05, window=None)
    assert out.index.equals(normal_returns.index)
    # Expanding VaR at position 100 == quantile of first 100 obs only.
    first100 = normal_returns.iloc[:100]
    assert out.iloc[100] == pytest.approx(-np.quantile(first100, 0.05))
def test_rolling_var_alignment_forecast_convention():
    """Rolling VaR at t is a forecast for day t: window strictly before t.

    Regression for the _rolling alignment bug (found by external audit,
    2026-10-12): the window's value used to be stamped on the window's
    last day, so day t sat inside its own estimation window and the
    first forecast appeared one day early (249 NaNs, not 250).
    """
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0, 0.01, 400))
    out = V.historical_var(r, alpha=0.05, window=250)
    # First forecast at position 250, from the 250 days strictly before it.
    assert out.iloc[:250].isna().all()
    assert out.iloc[250] == pytest.approx(-np.quantile(r.iloc[:250], 0.05))
    assert out.notna().sum() == 150
    # Self-inclusion probe: perturbing day t's return must not move the
    # VaR stamped at t (it is a forecast made before day t happens)...
    r2 = r.copy()
    r2.iloc[300] = -0.50
    out2 = V.historical_var(r2, alpha=0.05, window=250)
    assert out2.iloc[300] == pytest.approx(out.iloc[300])
    # ...but must move the next day's VaR, whose window contains day 300.
    assert out2.iloc[301] != pytest.approx(out.iloc[301])
def test_rolling_var_backtest_counts_forecast_days_only():
    """exceptions() over a rolling VaR scores exactly n - window days.

    Ties the warmup fix and the alignment fix together: with window=250
    on 400 obs there are 150 scored days (not 151); the 250 warmup days
    are present but marked <NA>, never scored.
    """
    rng = np.random.default_rng(11)
    r = pd.Series(rng.normal(0, 0.01, 400))
    var = V.historical_var(r, alpha=0.05, window=250)
    exc = B.exceptions(r, var)
    assert len(exc) == 400
    assert int(exc.notna().sum()) == 150
    assert exc.iloc[:250].isna().all()


# ---------------------------------------------------------------- backtests

def _exc_at_rate(n, alpha, seed=1):
    """SYNTHETIC: iid Bernoulli(alpha) exceptions, unclustered."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.random(n) < alpha)
def test_kupiec_does_not_reject_at_nominal_rate():
    """SYNTHETIC: exceptions at exactly the nominal 5% rate → no reject."""
    n = 1000
    exc = pd.Series([(i % 20 == 0) for i in range(n)])  # exactly 5%
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # n >= 250 anyway; silence anyway
        res = B.kupiec_pof(exc, 0.05)
    assert res["lr_stat"] == pytest.approx(0.0, abs=1e-9)
    assert not res["reject"]
    assert res["n_exceptions"] == 50
def test_exceptions_exclude_nan_warmup():
    """Warmup days (NaN VaR) issued no forecast: excluded, not False.

    Regression for the Kupiec warmup bug: exceptions() used to pad the
    warmup as non-exceptions, so kupiec_pof counted forecast-less days in
    n and deflated the exception rate, inflating the LR statistic (on real
    SPY data, n = 1,953 / LR = 2.104 instead of n = 1,704 / LR = 0.018).
    """
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    rets = pd.Series(-0.001, index=idx)
    var = pd.Series(0.02, index=idx)
    var.iloc[:250] = np.nan            # rolling-estimator warmup
    rets.iloc[260] = -0.05             # one true exception, after warmup
    exc = B.exceptions(rets, var)
    assert len(exc) == 300             # full length, warmup marked <NA>
    assert exc.index.equals(idx)
    assert exc.iloc[:250].isna().all()
    assert int(exc.notna().sum()) == 50
    assert int(exc.sum()) == 1
    res = B.kupiec_pof(exc, 0.05)
    assert res["n"] == 50              # warmup must not inflate n
    assert res["n_exceptions"] == 1
def test_exceptions_exclude_nan_returns():
    """A NaN return on a scored day is excluded, not a non-exception.

    Found by external fix-verification (2026-10-12): the warmup mask
    only looked at the VaR side, so a missing outcome (halt, data gap)
    silently padded n exactly like the warmup bug did.
    """
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    rets = pd.Series(-0.001, index=idx)
    var = pd.Series(0.02, index=idx)
    rets.iloc[150] = np.nan
    exc = B.exceptions(rets, var)
    assert len(exc) == 300
    assert pd.isna(exc.iloc[150])      # marked unscored, not False
    assert int(exc.notna().sum()) == 299
def test_exceptions_ndarray_var_is_positional():
    """Non-Series VaR aligns positionally; wrong length raises."""
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    rets = pd.Series(-0.001, index=idx)
    exc = B.exceptions(rets, np.full(300, 0.02))
    assert len(exc) == 300
    with pytest.raises(ValueError, match="align"):
        B.exceptions(rets, np.full(295, 0.02))
def test_exceptions_disjoint_index_raises():
    """A VaR Series sharing no index labels must not silently score 0."""
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    rets = pd.Series(-0.001, index=idx)
    var = pd.Series(0.02, index=pd.RangeIndex(300))
    with pytest.raises(ValueError, match="no common index"):
        B.exceptions(rets, var)
def test_exceptions_warmup_slice_empty_not_misalignment():
    """A returns slice inside the VaR warmup shares every index label
    but has NaN VaR throughout: exceptions() must return empty (and
    kupiec_pof then raises its own n=0 error), NOT the disjoint-index
    error. Regression for the guard misfire found in verification
    round 2 — the first guard tested post-reindex NaN counts instead
    of label overlap.
    """
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0, 0.01, 300),
                  index=pd.bdate_range("2024-01-01", periods=300))
    var = V.historical_var(r, alpha=0.05, window=250)  # NaN for 250 days
    exc = B.exceptions(r.iloc[:100], var)
    assert len(exc) == 100
    assert int(exc.notna().sum()) == 0  # all unscored, but no misfire
    with pytest.raises(ValueError, match="n=0"):
        B.kupiec_pof(exc, 0.05)
def test_empty_backtest_raises_not_passes():
    """n=0 used to return LR 0 / p=1.0 — a silent 'perfect pass'."""
    empty = pd.Series([], dtype=bool)
    with pytest.raises(ValueError, match="n=0"):
        B.kupiec_pof(empty, 0.05)
    with pytest.raises(ValueError, match="n=0"):
        B.christoffersen_ind(empty)
def test_kupiec_rejects_wrong_rate():
    """SYNTHETIC: 15% exceptions against a 5% VaR → reject."""
    n = 1000
    exc = pd.Series([(i % 20 < 3) for i in range(n)])  # 15%
    res = B.kupiec_pof(exc, 0.05)
    assert res["reject"]
    assert res["p_value"] < 1e-6
def test_christoffersen_rejects_clustered_exceptions():
    """SYNTHETIC: one solid block of exceptions → independence rejected."""
    n = 500
    exc = pd.Series([i < 40 for i in range(n)])  # 40 in a row, then none
    res = B.christoffersen_ind(exc)
    assert res["reject"], "clustered exceptions must fail independence"
    assert res["n11"] == 39
def test_christoffersen_no_reject_on_iid():
    """SYNTHETIC: iid exceptions → independence not rejected."""
    exc = _exc_at_rate(2000, 0.05)
    res = B.christoffersen_ind(exc)
    assert not res["reject"]
def test_christoffersen_skips_pairs_across_gaps():
    """Interior <NA> holes break adjacency: exception, hole, exception
    must NOT register an n11 transition.

    Regression for the gap-joining flaw (verification round 2): when
    exceptions() returned scored days only, the two exceptions either
    side of a hole were joined into a spurious 1->1 pair — with SPY's
    n11 = 12, one fake pair moves LR_ind materially.
    """
    vals = [False] * 300
    vals[100] = True
    vals[102] = True
    exc = pd.Series(vals, dtype="boolean")
    exc.iloc[101] = pd.NA
    res = B.christoffersen_ind(exc)
    assert res["n"] == 299             # scored days only
    assert res["n11"] == 0             # the gap pair is skipped, not joined
    assert res["n01"] == 1 and res["n10"] == 1
    assert res["n00"] + res["n01"] + res["n10"] + res["n11"] == 297
    # Same layout with the hole filled True: two genuine 1->1 pairs.
    exc2 = exc.copy()
    exc2.iloc[101] = True
    assert B.christoffersen_ind(exc2)["n11"] == 2
def test_christoffersen_pair_restricted_mle():
    """The restricted likelihood uses the pair-based MLE
    pi = (n01+n11)/pairs, not the day-count mean x.mean().

    Regression for verification round 3: with x.mean() the restricted
    likelihood is not maximized, so LR_ind is biased upward (on SPY:
    10.589804 vs the correct 10.589773), and under interior gaps the
    mean-based statistic can fabricate rejections.
    """
    import math
    seq = [False] * 50 + [True] + [False] * 50 + [True, True] + [False] * 47
    res = B.christoffersen_ind(pd.Series(seq, dtype="boolean"))
    # Counts by construction: 00=144, 01=2, 10=2, 11=1 (149 pairs).
    assert (res["n00"], res["n01"], res["n10"], res["n11"]) == (144, 2, 2, 1)
    n00, n01, n10, n11 = 144, 2, 2, 1
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_r = (n00 + n10) * math.log(1 - pi) + (n01 + n11) * math.log(pi)
    ll_u = (n00 * math.log(1 - n01 / (n00 + n01)) + n01 * math.log(n01 / (n00 + n01))
            + n10 * math.log(1 - n11 / (n10 + n11)) + n11 * math.log(n11 / (n10 + n11)))
    assert res["lr_stat"] == pytest.approx(-2.0 * (ll_r - ll_u), rel=1e-9)
def test_christoffersen_undefined_when_transitions_unidentified():
    """Exceptions with no usable adjacent pairs must return NaN, not a
    fabricated LR 0 / p 1 'pass'. Regression for verification round 3.
    """
    # (a) Every exception isolated by <NA> holes: zero adjacent pairs.
    exc = pd.Series([pd.NA] * 300, dtype="boolean")
    for i in (10, 100, 200):
        exc.iloc[i] = True
    res = B.christoffersen_ind(exc)
    assert np.isnan(res["lr_stat"]) and not res["reject"]
    assert "undefined" in res["note"]
    # (b) Plain series, single exception on the final day: no pair
    # starts in state 1, so p11 is unidentified.
    vals = [False] * 300
    vals[299] = True
    res2 = B.christoffersen_ind(pd.Series(vals, dtype="boolean"))
    assert np.isnan(res2["lr_stat"]) and not res2["reject"]
def test_christoffersen_undefined_on_boundary_pi():
    """No pair ENDS in an exception (the sole exception is the first
    scored day): the restricted MLE sits on the boundary and the LR
    degenerates to 0 / p 1. Must return NaN like the other unidentified
    shapes. Regression for verification round 4, item E1.
    """
    vals = [True] + [False] * 299
    res = B.christoffersen_ind(pd.Series(vals, dtype="boolean"))
    assert (res["n00"], res["n01"], res["n10"], res["n11"]) == (298, 0, 1, 0)
    assert np.isnan(res["lr_stat"]) and not res["reject"]
    assert "undefined" in res["note"]
def test_nonbinary_exceptions_input_raises():
    """Exceptions input must be boolean / 0-1: a stray 2 (or raw
    returns passed by mistake) raises instead of coercing silently.
    Regression for verification round 4, item E4.
    """
    bad = pd.Series([0.0, 1.0, 2.0, 0.0] * 100)
    with pytest.raises(ValueError, match="0/1"):
        B.kupiec_pof(bad, 0.05)
    with pytest.raises(ValueError, match="0/1"):
        B.christoffersen_ind(bad)
    # Exotic bad input must still surface as ValueError, not leak a
    # TypeError from the boolean fallback cast (round 5 nit).
    exotic = pd.Series([True, pd.NA, 2] * 100, dtype=object)
    with pytest.raises(ValueError):
        B.kupiec_pof(exotic, 0.05)
    with pytest.raises(ValueError):
        B.kupiec_pof(pd.Series(["yes", "no"] * 150), 0.05)
def test_joint_fallback_distinguishes_untestable():
    """joint_test's Kupiec fallback (df=1) says WHY independence was
    undefined: no exceptions vs exceptions present but untestable.
    Regression for verification round 4, item E2.
    """
    jt0 = B.joint_test(pd.Series([False] * 300, dtype="boolean"), 0.05)
    assert jt0["df"] == 1 and "No exceptions" in jt0["note"]
    exc = pd.Series([pd.NA] * 300, dtype="boolean")
    for i in range(0, 300, 30):
        exc.iloc[i] = True       # 10 exceptions, all isolated by holes
    jt1 = B.joint_test(exc, 0.05)
    assert jt1["df"] == 1 and "untestable" in jt1["note"]
    jt2 = B.joint_test(_exc_at_rate(1000, 0.05), 0.05)
    assert jt2["df"] == 2
def test_zero_exceptions_degenerate_gracefully():
    """Edge: no exceptions → independence test returns NaN + note, no crash."""
    exc = pd.Series([False] * 300)
    res = B.christoffersen_ind(exc)
    assert np.isnan(res["lr_stat"]) and np.isnan(res["p_value"])
    assert "undefined" in res["note"]
    joint = B.joint_test(exc, 0.05)
    assert "falls back to Kupiec" in joint["note"]
def test_short_sample_warns():
    """Edge: T < 250 → UserWarning that the test is asymptotic."""
    exc = _exc_at_rate(100, 0.05)
    with pytest.warns(UserWarning, match="asymptotic"):
        B.kupiec_pof(exc, 0.05)
    with pytest.warns(UserWarning, match="asymptotic"):
        B.christoffersen_ind(exc)
def test_basel_traffic_light_zones():
    assert B.basel_traffic_light(3)["zone"] == "green"
    assert B.basel_traffic_light(7)["zone"] == "yellow"
    assert B.basel_traffic_light(12)["zone"] == "red"


# ---------------------------------------------------------------- tail
def test_jb_rejects_t_dist(t_returns):
    """SYNTHETIC: fat-tailed t(4) returns → JB rejects normality hard."""
    res = T.jarque_bera(t_returns)
    assert res["reject"]
    assert res["jb_stat"] > 5.99
    assert res["kurtosis"] > 4.0
def test_jb_no_reject_on_normal(normal_returns):
    """SYNTHETIC: truly normal returns → JB does not reject."""
    res = T.jarque_bera(normal_returns)
    assert not res["reject"]
def test_hill_recovers_tail_index(rng):
    """SYNTHETIC: Pareto(shape=4) losses have tail index xi = 1/4 = 0.25.

    (On Student-t the Hill estimator has known finite-sample upward bias,
    so the precision check uses Pareto, where it is nearly unbiased.)
    """
    pareto = rng.pareto(4, 30000) + 1.0
    res = T.hill_estimator(pareto, k=500)
    assert res["xi"] == pytest.approx(0.25, abs=0.03)
def test_hill_positive_on_t_dist(t_returns):
    """SYNTHETIC: t(4) losses give a clearly positive (heavy) tail index."""
    res = T.hill_estimator(-t_returns, k=500)
    assert res["xi"] > 0.15
def test_hill_plot_data_shape(t_returns):
    df = T.hill_plot_data(-t_returns, k_range=range(50, 200, 10))
    assert list(df.columns) == ["k", "xi", "se"]
    assert len(df) == 15
    assert (df["xi"] > 0).all()
