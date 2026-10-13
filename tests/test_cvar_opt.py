"""Tests for scenario-based tail optimization (cvar.py additions).

ALL DATA HERE IS SYNTHETIC. External verification: the Rockafellar-Uryasev
LP is checked against the textbook CVaR definition (metrics.cvar_historical,
an independent code path) and against brute-force grid search.
"""

import numpy as np
import pandas as pd
import pytest

from portlab.metrics import cvar_historical
from portlab.optimize import max_return_at_cvar, min_cvar
from portlab.optimize.cvar import cvar_efficient_frontier


@pytest.fixture()
def rets():
    rng = np.random.default_rng(20261010)
    # A: higher mean but fat left tail (high CVaR); B: lower mean, calm.
    # The tradeoff is what makes the CVaR budget bind.
    a = np.concatenate([rng.normal(0.016, 0.02, 400),
                        rng.normal(-0.08, 0.03, 25)])
    b = rng.normal(0.006, 0.015, 425)
    idx = pd.bdate_range("2020-01-01", periods=425)
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def test_cvar_budget_binds(rets):
    w = max_return_at_cvar(rets, alpha=0.95, max_cvar=0.03)
    c = cvar_historical(rets @ w, 0.95)
    # budget binds: realized CVaR at the budget (LP tolerance)
    assert c == pytest.approx(0.03, rel=0.05)


def test_consistent_with_min_cvar(rets):
    """Both directions trace the same mean-CVaR frontier."""
    w_star = max_return_at_cvar(rets, alpha=0.95, max_cvar=0.035)
    target = float((rets @ w_star).mean())
    w2 = min_cvar(rets, alpha=0.95, min_return=target)
    assert np.allclose(w_star.to_numpy(), w2.to_numpy(), atol=0.02)


def test_brute_force_two_assets(rets):
    w = max_return_at_cvar(rets[["A", "B"]], alpha=0.95, max_cvar=0.035)
    best, best_ret = None, -np.inf
    for wA in np.linspace(0, 1, 201):
        pr = rets["A"] * wA + rets["B"] * (1 - wA)
        if cvar_historical(pr, 0.95) <= 0.035 * 1.001:
            r = pr.mean()
            if r > best_ret:
                best_ret, best = r, wA
    assert w["A"] == pytest.approx(best, abs=0.02)


def test_unbinding_budget_gives_max_return_asset(rets):
    w = max_return_at_cvar(rets, alpha=0.95, max_cvar=10.0)
    assert w.idxmax() == rets.mean().idxmax()
    assert w.max() == pytest.approx(1.0)


def test_frontier_monotone(rets):
    fr = cvar_efficient_frontier(rets, alpha=0.95, n_points=8)
    assert len(fr) == 8
    # more return demands more tail risk
    assert (fr["cvar"].diff().dropna() >= -1e-9).all()
    assert (fr["mean_return"].diff().dropna() >= -1e-9).all()
    wcols = [c for c in fr.columns if c.startswith("w_")]
    assert np.allclose(fr[wcols].sum(axis=1), 1.0, atol=1e-8)


def test_rejects_bad_budget(rets):
    with pytest.raises(ValueError, match="max_cvar"):
        max_return_at_cvar(rets, max_cvar=0.0)
