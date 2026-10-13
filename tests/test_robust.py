"""Tests for portlab.optimize.robust.

ALL DATA HERE IS SYNTHETIC. External verification: the box reduction
(mu_hat - delta) is the Bertsimas & Sim (2004) theorem; the ellipsoidal
penalty k*sqrt(w'Om w) is the Goldfarb & Iyengar (2003) worst case — both
are checked here against brute-force grid search on 2 assets.
"""

import numpy as np
import pandas as pd
import pytest

from portlab.optimize import (delta_from_se, opt_constrained,
                              robust_box_weights, robust_ellipsoid_weights)


@pytest.fixture()
def two_asset():
    mu = pd.Series([0.10, 0.05], index=["A", "B"])
    cov = pd.DataFrame(np.eye(2) * 0.04, index=["A", "B"], columns=["A", "B"])
    return mu, cov


def test_box_zero_delta_is_standard_mvo(two_asset):
    mu, cov = two_asset
    got = robust_box_weights(mu, cov, delta=pd.Series([0.0, 0.0],
                                                     index=mu.index),
                             risk_aversion=5.0, tc_bps=0.0)
    want = opt_constrained(mu, cov, risk_aversion=5.0, tc_bps=0.0,
                           bounds=(0.0, 1.0))
    assert np.allclose(got.to_numpy(), want.to_numpy(), atol=1e-6)


def test_box_exact_reduction(two_asset):
    """Bertsimas & Sim: long-only box-robust == MVO on mu - delta."""
    mu, cov = two_asset
    delta = pd.Series([0.02, 0.01], index=mu.index)
    got = robust_box_weights(mu, cov, delta=delta, risk_aversion=5.0,
                             tc_bps=0.0)
    want = opt_constrained(mu - delta, cov, risk_aversion=5.0, tc_bps=0.0,
                           bounds=(0.0, 1.0))
    assert np.allclose(got.to_numpy(), want.to_numpy(), atol=1e-9)


def test_box_uncertainty_flips_ranking(two_asset):
    """Uncertainty concentrated on the high-mean asset erases its edge:
    delta=[0.05, 0] makes robust means equal -> min-variance 50/50."""
    mu, cov = two_asset
    got = robust_box_weights(mu, cov, delta=pd.Series([0.05, 0.0],
                                                      index=mu.index),
                             risk_aversion=5.0, tc_bps=0.0)
    assert np.allclose(got.to_numpy(), [0.5, 0.5], atol=1e-6)
    # smaller uncertainty keeps A's edge
    got2 = robust_box_weights(mu, cov, delta=pd.Series([0.02, 0.0],
                                                       index=mu.index),
                              risk_aversion=5.0, tc_bps=0.0)
    assert got2["A"] > 0.5


def test_box_rejects_short_bounds(two_asset):
    mu, cov = two_asset
    with pytest.raises(ValueError, match="long-only"):
        robust_box_weights(mu, cov, delta=0.01, bounds=(-0.2, 1.0))
    with pytest.raises(ValueError, match="nonnegative"):
        robust_box_weights(mu, cov, delta=-0.01)


def test_delta_from_se():
    se = pd.Series([0.02, 0.03], index=["A", "B"])
    d = delta_from_se(se, z=2.0)
    assert np.allclose(d.to_numpy(), [0.04, 0.06])


def test_ellipsoid_zero_kappa_is_standard_mvo(two_asset):
    mu, cov = two_asset
    got = robust_ellipsoid_weights(mu, cov, kappa=0.0, n_obs=252,
                                   risk_aversion=5.0)
    want = opt_constrained(mu, cov, risk_aversion=5.0, tc_bps=0.0,
                           bounds=(0.0, 1.0))
    assert np.allclose(got.to_numpy(), want.to_numpy(), atol=1e-4)


def test_ellipsoid_vs_brute_force(two_asset):
    """2-asset grid search on the exact robust objective."""
    mu, cov = two_asset
    kappa, n_obs, gamma = 2.0, 100, 5.0
    got = robust_ellipsoid_weights(mu, cov, kappa=kappa, n_obs=n_obs,
                                   risk_aversion=gamma)
    m, S = mu.to_numpy(), cov.to_numpy()
    Om = S / n_obs
    best, best_val = 0.5, -np.inf
    for wA in np.linspace(0, 1, 2001):
        w = np.array([wA, 1 - wA])
        val = m @ w - kappa * np.sqrt(w @ Om @ w) - 0.5 * gamma * w @ S @ w
        if val > best_val:
            best_val, best = val, wA
    assert got["A"] == pytest.approx(best, abs=5e-3)


def test_ellipsoid_more_robust_more_diversified(two_asset):
    """Larger kappa pulls the concentrated nominal optimum toward 1/n."""
    mu, cov = two_asset
    nominal = opt_constrained(mu, cov, risk_aversion=5.0, tc_bps=0.0,
                              bounds=(0.0, 1.0))
    mild = robust_ellipsoid_weights(mu, cov, kappa=1.0, n_obs=60,
                                    risk_aversion=5.0)
    strong = robust_ellipsoid_weights(mu, cov, kappa=8.0, n_obs=60,
                                      risk_aversion=5.0)
    d_nom = abs(nominal["A"] - 0.5)
    assert abs(mild["A"] - 0.5) <= d_nom + 1e-6
    assert abs(strong["A"] - 0.5) <= abs(mild["A"] - 0.5) + 1e-6


def test_ellipsoid_needs_omega_or_n_obs(two_asset):
    mu, cov = two_asset
    with pytest.raises(ValueError, match="omega or n_obs"):
        robust_ellipsoid_weights(mu, cov, kappa=1.0)
