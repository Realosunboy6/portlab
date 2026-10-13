"""Tests for portlab.optimize.constraint_attribution.

ALL DATA HERE IS SYNTHETIC. External verification: the two-asset hand
calculation below is worked out from first principles (closed-form QP
solution), and the relaxation-monotonicity property is a theorem.
"""

import numpy as np
import pandas as pd
import pytest

from portlab.optimize import attribute_constraints
from portlab.optimize.constrained import opt_constrained


@pytest.fixture()
def two_asset():
    mu = pd.Series([0.10, 0.05], index=["A", "B"])
    cov = pd.DataFrame(np.eye(2) * 0.04, index=["A", "B"], columns=["A", "B"])
    w0 = pd.Series([0.5, 0.5], index=["A", "B"])
    gamma = 5.0
    obj = lambda w: w @ mu.to_numpy() - 0.5 * gamma * w @ cov.to_numpy() @ w
    solve = lambda bounds, max_turnover: opt_constrained(
        mu, cov, w_prev=w0, risk_aversion=gamma, tc_bps=0.0,
        bounds=bounds, max_turnover=max_turnover)
    return mu, cov, w0, gamma, obj, solve


def test_hand_calculated_single_binding_cap(two_asset):
    """Closed form: unconstrained optimum w=(0.625,0.375) obj=0.028125;
    with a 50% cap w=(0.5,0.5) obj=0.025 -> the cap costs 0.003125."""
    mu, cov, w0, gamma, obj, solve = two_asset
    solve1 = lambda bounds: solve(bounds, None)
    tbl = attribute_constraints(solve1, params={"bounds": (0.0, 0.5)},
                                relaxed={"bounds": (0.0, 1.0)},
                                objective_fn=obj)
    assert tbl.loc["bounds", "delta"] == pytest.approx(0.003125, rel=1e-3)
    # single constraint: leave-one-out delta IS the total gap
    assert tbl.loc["TOTAL (all relaxed)", "delta"] == pytest.approx(
        tbl.loc["bounds", "delta"], rel=1e-6)


def test_nonbinding_constraint_costs_nothing(two_asset):
    mu, cov, w0, gamma, obj, solve = two_asset
    tbl = attribute_constraints(
        solve, params={"bounds": (0.0, 1.0), "max_turnover": 10.0},
        relaxed={"bounds": (0.0, 1.0), "max_turnover": None},
        objective_fn=obj)
    assert tbl.loc["max_turnover", "delta"] == pytest.approx(0.0, abs=1e-6)
    assert tbl.loc["bounds", "delta"] == pytest.approx(0.0, abs=1e-6)


def test_two_constraints_monotone(two_asset):
    """Relaxing never hurts; TOTAL >= each marginal delta."""
    mu, cov, w0, gamma, obj, solve = two_asset
    tbl = attribute_constraints(
        solve, params={"bounds": (0.0, 0.5), "max_turnover": 0.05},
        relaxed={"bounds": (0.0, 1.0), "max_turnover": None},
        objective_fn=obj)
    assert (tbl["delta"].dropna() >= 0).all()
    total = tbl.loc["TOTAL (all relaxed)", "delta"]
    assert total >= tbl.loc["bounds", "delta"] - 1e-9
    assert total >= tbl.loc["max_turnover", "delta"] - 1e-9


def test_annualization(two_asset):
    mu, cov, w0, gamma, obj, solve = two_asset
    solve1 = lambda bounds: solve(bounds, None)
    tbl = attribute_constraints(solve1, params={"bounds": (0.0, 0.5)},
                                relaxed={"bounds": (0.0, 1.0)},
                                objective_fn=obj, annualize=252.0)
    assert tbl.loc["bounds", "delta_annualized"] == pytest.approx(
        tbl.loc["bounds", "delta"] * 252.0)


def test_mismatched_keys_rejected(two_asset):
    mu, cov, w0, gamma, obj, solve = two_asset
    with pytest.raises(ValueError, match="same keys"):
        attribute_constraints(solve, params={"bounds": (0, 0.5)},
                              relaxed={}, objective_fn=obj)


def test_generic_interface_with_mpo():
    """The generic solve_fn interface also works with multi-period weights."""
    from portlab.mpo import multiperiod_weights
    rng = np.random.default_rng(3)
    n, T = 3, 2
    assets = ["A", "B", "C"]
    w0 = pd.Series([1 / 3] * n, index=assets)
    mus = pd.DataFrame(rng.normal(0.01, 0.02, (T, n)), columns=assets)
    A = rng.normal(0, 1, (n, n))
    cov = pd.DataFrame(A @ A.T / n * 0.01 + np.eye(n) * 0.005,
                       index=assets, columns=assets)
    mu1, cov1 = mus.iloc[0], cov
    gamma = 5.0
    obj = lambda w: w @ mu1.to_numpy() - 0.5 * gamma * w @ cov1.to_numpy() @ w
    solve = lambda tc_bps: multiperiod_weights(
        w0, mus.iloc[[0]], [cov1], risk_aversion=gamma,
        tc_bps=tc_bps)["weights"].iloc[0].to_numpy()
    tbl = attribute_constraints(solve, params={"tc_bps": 50.0},
                                relaxed={"tc_bps": 0.0}, objective_fn=obj)
    assert tbl.loc["tc_bps", "delta"] >= 0
