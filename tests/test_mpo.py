"""Tests for portlab.mpo (multi-period turnover-aware optimization).

ALL DATA HERE IS SYNTHETIC. External verification strategy:
- T=1 must reproduce the single-period `opt_constrained` solution
  (independent solver, same objective).
- The two structural limits (kappa->inf => hold; kappa=0 => decoupled
  single-period optima) are theorems of the Boyd et al. (2017) formulation.
"""

import numpy as np
import pandas as pd
import pytest

from portlab.mpo import multiperiod_weights
from portlab.optimize.constrained import opt_constrained


@pytest.fixture()
def setup():
    rng = np.random.default_rng(20261010)
    n, T = 4, 3
    assets = ["A", "B", "C", "D"]
    w0 = pd.Series([0.25] * n, index=assets)
    mus = pd.DataFrame(rng.normal(0.01, 0.02, (T, n)),
                       index=[f"t{i}" for i in range(T)], columns=assets)
    A = rng.normal(0, 1, (n, n))
    cov = A @ A.T / n * 0.01 + np.eye(n) * 0.005
    covs = [pd.DataFrame(cov, index=assets, columns=assets)] * T
    return w0, mus, covs


def test_single_period_matches_opt_constrained(setup):
    """T=1 solves the same problem as opt_constrained (different solver)."""
    w0, mus, covs = setup
    mu1, cov1 = mus.iloc[[0]], covs[0]
    got = multiperiod_weights(w0, mu1, [cov1], risk_aversion=5.0,
                              tc_bps=10.0)["weights"].iloc[0]
    want = opt_constrained(mu1.iloc[0], cov1, w_prev=w0, risk_aversion=5.0,
                           tc_bps=10.0, bounds=(0.0, 1.0))
    assert np.allclose(got.to_numpy(), want.to_numpy(), atol=1e-3)


def test_huge_tc_holds_forever(setup):
    """kappa -> inf: the optimal plan is to never trade."""
    w0, mus, covs = setup
    out = multiperiod_weights(w0, mus, covs, tc_bps=1e7)
    assert np.allclose(out["weights"].to_numpy(),
                       np.tile(w0.to_numpy(), (len(mus), 1)), atol=1e-6)
    assert out["total_turnover"] == pytest.approx(0.0, abs=1e-8)


def test_zero_tc_decouples_periods(setup):
    """kappa = 0: each period is an independent single-period optimum."""
    w0, mus, covs = setup
    out = multiperiod_weights(w0, mus, covs, risk_aversion=5.0, tc_bps=0.0)
    for t in range(len(mus)):
        want = opt_constrained(mus.iloc[t], covs[t], w_prev=None,
                               risk_aversion=5.0, tc_bps=0.0,
                               bounds=(0.0, 1.0))
        assert np.allclose(out["weights"].iloc[t].to_numpy(),
                           want.to_numpy(), atol=1e-3)


def test_turnover_accounting(setup):
    w0, mus, covs = setup
    out = multiperiod_weights(w0, mus, covs, tc_bps=25.0)
    w = out["weights"]
    prev = w0
    for t in range(len(w)):
        want = (w.iloc[t] - prev).abs().sum() / 2.0
        assert out["turnover"].iloc[t] == pytest.approx(want, rel=1e-8)
        prev = w.iloc[t]
    assert out["total_turnover"] == pytest.approx(out["turnover"].sum())


def test_tc_dampens_first_trade(setup):
    """A positive turnover penalty shrinks (never grows) the first trade."""
    w0, mus, covs = setup
    free = multiperiod_weights(w0, mus, covs, tc_bps=0.0)
    penalized = multiperiod_weights(w0, mus, covs, tc_bps=50.0)
    assert penalized["turnover"].iloc[0] <= free["turnover"].iloc[0] + 1e-9


def test_constraints_hold(setup):
    w0, mus, covs = setup
    out = multiperiod_weights(w0, mus, covs, tc_bps=10.0)
    assert (out["weights"].to_numpy() >= -1e-9).all()
    assert np.allclose(out["weights"].sum(axis=1).to_numpy(), 1.0, atol=1e-8)


def test_validation(setup):
    w0, mus, covs = setup
    with pytest.raises(ValueError, match="sum to 1"):
        multiperiod_weights(w0 * 2, mus, covs)
    with pytest.raises(ValueError, match="risk_aversion"):
        multiperiod_weights(w0, mus, covs, risk_aversion=0.0)
    with pytest.raises(ValueError, match="tc_bps"):
        multiperiod_weights(w0, mus, covs, tc_bps=-1.0)
    with pytest.raises(ValueError, match="shape"):
        multiperiod_weights(w0, mus, covs[:2])
