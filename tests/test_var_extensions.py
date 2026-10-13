"""Tests for the VaR/ES extensions in portlab.risk.var.

ALL DATA HERE IS SYNTHETIC. External verification:
- Student-t VaR/ES closed forms are checked against direct scipy
  evaluations and Monte Carlo (independent code paths in each test).
- Cornish-Fisher is checked by reduction to the normal case plus an
  independent transcription of the Zangari (1996) expansion.
- GARCH(1,1) forecasts are cross-checked against the `arch` package
  (independent implementation, variance-targeted MLE vs full MLE).
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from portlab.risk import var as V


@pytest.fixture()
def rng():
    return np.random.default_rng(20261010)


def _series(x):
    return pd.Series(np.asarray(x, dtype=float),
                     index=pd.bdate_range("2020-01-01", periods=len(x)))


# ------------------------------------------------------- Student-t VaR
def test_tvar_converges_to_normal_as_df_grows(rng):
    r = _series(rng.normal(0.001, 0.02, 2000))
    t = V.parametric_var_t(r, alpha=0.05, window=500, df=1000).dropna()
    n = V.parametric_var(r, alpha=0.05, window=500).dropna()
    assert np.allclose(t, n, rtol=0.02)


def test_tvar_fixed_df_hand_check(rng):
    """Independent path: assemble -(mu + scale*t_ppf) directly from scipy."""
    w = rng.normal(0.001, 0.02, 500)
    # One extra day: under the forecast convention (VaR at t uses data
    # strictly before t), the last stamp's window is exactly w. A series
    # of exactly `window` obs has no forecast day at all.
    r = _series(np.concatenate([w, [0.0]]))
    got = V.parametric_var_t(r, alpha=0.05, window=500, df=5).iloc[-1]
    mu, sd = w.mean(), w.std(ddof=1)
    scale = sd * np.sqrt(3.0 / 5.0)
    want = -(mu + scale * stats.t.ppf(0.05, 5))
    assert got == pytest.approx(want, rel=1e-10)


def test_tvar_tail_ordering_flips_with_alpha(rng):
    # Subtle but real (verified 2026-10-10): variance-matched, the t(4)
    # 5% quantile (-1.508 sd) is LESS extreme than normal (-1.645 sd) —
    # the fat tail puts mass *beyond* the 5% point. At 1% the ordering
    # flips (-2.649 vs -2.326 sd). This is why RiskLens reports both levels.
    r = _series(rng.standard_t(4, 3000) * 0.02)
    t95 = V.parametric_var_t(r, alpha=0.05, window=1000, df=4).dropna()
    n95 = V.parametric_var(r, alpha=0.05, window=1000).dropna()
    t99 = V.parametric_var_t(r, alpha=0.01, window=1000, df=4).dropna()
    n99 = V.parametric_var(r, alpha=0.01, window=1000).dropna()
    assert (t95 < n95).mean() > 0.9
    assert (t99 > n99 * 1.05).mean() > 0.9


def test_tvar_fits_df_from_data(rng):
    r = _series(rng.standard_t(6, 3000) * 0.02)
    out = V.parametric_var_t(r, alpha=0.05, window=1500).dropna()
    assert (out > 0).all() and np.isfinite(out).all()


# ------------------------------------------------------- Student-t ES
def test_tes_closed_form_vs_monte_carlo(rng):
    """Independent check: closed-form t-ES vs 5M-draw empirical ES."""
    nu, loc, scale, alpha = 5, 0.001, 0.02, 0.05
    sim = rng.standard_t(nu, 5_000_000) * scale + loc
    emp_es = -sim[sim <= np.quantile(sim, alpha)].mean()
    q = stats.t.ppf(alpha, nu)
    g = stats.t.pdf(q, nu)
    closed = -loc + scale * (g / alpha) * (nu + q ** 2) / (nu - 1)
    assert closed == pytest.approx(emp_es, rel=0.02)


def test_tes_ge_tvar_and_positive(rng):
    r = _series(rng.standard_t(5, 2000) * 0.02 + 0.001)
    es = V.expected_shortfall_t(r, alpha=0.05, window=500, df=5).dropna()
    tv = V.parametric_var_t(r, alpha=0.05, window=500, df=5).dropna()
    assert ((es >= tv * 0.999) & np.isfinite(es)).all()


def test_tes_rejects_bad_df():
    r = _series(np.random.default_rng(0).normal(0, 0.01, 100))
    with pytest.raises(ValueError, match="df > 2"):
        V.parametric_var_t(r, df=2.0)


# ------------------------------------------------------- Cornish-Fisher
def test_cf_reduces_to_normal_on_gaussian(rng):
    r = _series(rng.normal(0.001, 0.02, 5000))
    cf = V.cornish_fisher_var(r, alpha=0.05, window=2500).dropna()
    n = V.parametric_var(r, alpha=0.05, window=2500).dropna()
    assert np.allclose(cf, n, rtol=0.05)


def test_cf_independent_transcription(rng):
    """The Zangari expansion written out directly, compared term by term."""
    w = rng.normal(0.001, 0.02, 1000)
    # One extra day, as in test_tvar_fixed_df_hand_check: the last
    # stamp's estimation window is exactly w under the forecast
    # convention.
    r = _series(np.concatenate([w, [0.0]]))
    got = V.cornish_fisher_var(r, alpha=0.05, window=1000).iloc[-1]
    z = stats.norm.ppf(0.05)
    s = stats.skew(w, bias=False)
    k = stats.kurtosis(w, bias=False)
    z_cf = (z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24
            - (2 * z ** 3 - 5 * z) * s ** 2 / 36)
    want = -(w.mean() + w.std(ddof=1) * z_cf)
    assert got == pytest.approx(want, rel=1e-10)


def test_cf_penalizes_left_skew(rng):
    # NOTE: probed with skew-normal (finite kurtosis), not t(4): the
    # Cornish-Fisher *expansion* is only valid for mild departures from
    # normality, and infinite-kurtosis tails make its higher-order terms
    # misbehave (documented in the docstring). With equity-like
    # skew/kurtosis, left skew must widen VaR vs the normal model.
    w = stats.skewnorm.rvs(-5, size=3000, random_state=rng) * 0.02
    left = _series(w)
    cf = V.cornish_fisher_var(left, alpha=0.05, window=1500).dropna()
    n = V.parametric_var(left, alpha=0.05, window=1500).dropna()
    assert (cf > n).mean() > 0.9  # skew/kurtosis widen VaR vs normal


# ------------------------------------------------------- GARCH
def test_garch_matches_arch_package(rng):
    """EXTERNAL ORACLE: one-step forecast vs the `arch` package."""
    arch = pytest.importorskip("arch")
    sim = rng.standard_t(8, 600) * 0.02
    # plant genuine clustering so the forecast is identified
    sim[300:330] *= 3.0
    w = sim[-250:]
    mine = V._garch11_forecast(w)
    am = arch.arch_model(w, mean="Zero", vol="GARCH", p=1, q=1)
    theirs = float(np.sqrt(am.fit(disp="off")
                           .forecast(horizon=1).variance.iloc[-1, 0]))
    assert mine == pytest.approx(theirs, rel=0.10)


def test_garch_var_reacts_to_clustering(rng):
    calm = rng.normal(0, 0.01, 300)
    shock = np.concatenate([rng.normal(0, 0.01, 250),
                            rng.normal(0, 0.05, 50)])
    v_calm = V.garch_var(_series(calm), alpha=0.05, window=200).dropna()
    v_shock = V.garch_var(_series(shock), alpha=0.05, window=200).dropna()
    # VaR after the volatile block must exceed calm-period VaR
    assert v_shock.iloc[-1] > v_calm.median() * 1.5


def test_garch_var_validation():
    r = _series(np.random.default_rng(1).normal(0, 0.01, 100))
    with pytest.raises(ValueError, match="dist"):
        V.garch_var(r, dist="cauchy")
    with pytest.raises(ValueError, match="window"):
        V.garch_var(r, window=10)


def test_garch_var_t_dist_option(rng):
    r = _series(rng.standard_t(6, 400) * 0.015)
    v = V.garch_var(r, alpha=0.05, window=200, dist="t").dropna()
    assert (v > 0).all() and np.isfinite(v).all()


def test_garch_var_mean_argument(rng):
    """mean= selects the location convention (external adjudication
    2026-10-12): the default "zero" reproduces the historical behavior
    exactly; "constant" fits on demeaned returns and adds the window
    mean back into the quantile (the arch package's default spec), so
    on a positive-drift series its VaR sits below the zero-mean one.
    """
    r = _series(rng.normal(0.001, 0.01, 400))  # planted positive drift
    v_default = V.garch_var(r, alpha=0.05, window=200)
    v_zero = V.garch_var(r, alpha=0.05, window=200, mean="zero")
    pd.testing.assert_series_equal(v_default, v_zero)
    v_const = V.garch_var(r, alpha=0.05, window=200, mean="constant")
    both = pd.concat([v_zero, v_const], axis=1).dropna()
    assert (both.iloc[:, 1] < both.iloc[:, 0]).mean() > 0.9
    assert both.iloc[-1, 1] < both.iloc[-1, 0]
    with pytest.raises(ValueError, match="mean"):
        V.garch_var(r, mean="ar")


def test_parametric_var_mean_argument():
    """Hand-checked, expanding window (last stamp uses all but the
    final observation, per the forecast-alignment contract):
    "constant" (default) is the historical -(mu - z*sd); "zero" is
    z * RMS(w). On positive drift the zero-mean VaR is the larger.
    """
    rng = np.random.default_rng(3)
    r = _series(rng.normal(0.001, 0.01, 300))
    w = r.to_numpy()[:-1]
    z = stats.norm.ppf(0.95)
    v_const = V.parametric_var(r, alpha=0.05)
    assert v_const.iloc[-1] == pytest.approx(
        -(w.mean() - z * w.std(ddof=1)), rel=1e-9)
    pd.testing.assert_series_equal(
        v_const, V.parametric_var(r, alpha=0.05, mean="constant"))
    v_zero = V.parametric_var(r, alpha=0.05, mean="zero")
    assert v_zero.iloc[-1] == pytest.approx(
        z * float(np.sqrt(np.mean(w * w))), rel=1e-9)
    assert v_zero.iloc[-1] > v_const.iloc[-1]
    with pytest.raises(ValueError, match="mean"):
        V.parametric_var(r, mean="ar")
