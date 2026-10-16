"""Solver backends: exactness vs SLSQP and graceful fallback.

Gurobi tests use the session mu_cov fixture (8 synthetic assets, well under
the free pip license's ~200-variable limit), so they run without a full
license. FISTA/OSQP have no size limit.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from portlab import optimize as opt
from portlab.optimize import gurobi_backend as gb
from portlab.optimize.core import portfolio_stats


def _needs_gurobi():
    return pytest.importorskip("gurobipy")


def _sharpe(w, mu, cov, rf=0.0):
    return portfolio_stats(np.asarray(w), mu, cov, rf)["sharpe"]


# ------------------------------------------------- exactness vs SLSQP

def test_gurobi_max_sharpe_matches_slsqp(mu_cov):
    _needs_gurobi()
    mu, cov = mu_cov
    wg = opt.max_sharpe(mu, cov, rf=0.0, solver="gurobi")
    ws = opt.max_sharpe(mu, cov, rf=0.0, solver="slsqp")
    assert _sharpe(wg, mu, cov) == pytest.approx(_sharpe(ws, mu, cov), abs=1e-6)
    assert np.abs(wg.values - ws.values).max() < 1e-3
    assert abs(wg.sum() - 1.0) < 1e-9


def test_gurobi_gmv_matches_slsqp(mu_cov):
    _needs_gurobi()
    mu, cov = mu_cov
    wg = opt.gmv(mu, cov, solver="gurobi")
    ws = opt.gmv(mu, cov, solver="slsqp")
    assert np.abs(wg.values - ws.values).max() < 1e-3


def test_gurobi_min_vol_at_return_matches_slsqp(mu_cov):
    _needs_gurobi()
    mu, cov = mu_cov
    tr = float(mu.quantile(0.7))
    wg = opt.min_vol_at_return(mu, cov, tr, solver="gurobi")
    ws = opt.min_vol_at_return(mu, cov, tr, solver="slsqp")
    assert wg @ mu.values == pytest.approx(tr, abs=1e-6)
    vg = np.sqrt(wg.values @ cov.values @ wg.values)
    vs = np.sqrt(ws.values @ cov.values @ ws.values)
    assert vg == pytest.approx(vs, abs=1e-6)


def test_gurobi_max_return_at_vol_matches_slsqp(mu_cov):
    _needs_gurobi()
    mu, cov = mu_cov
    tv = float(np.sqrt(np.diag(cov.values)).mean())
    wg = opt.max_return_at_vol(mu, cov, tv, solver="gurobi")
    ws = opt.max_return_at_vol(mu, cov, tv, solver="slsqp")
    assert (wg @ mu.values) == pytest.approx(ws @ mu.values, abs=1e-6)


def test_gurobi_frontier_matches_slsqp(mu_cov):
    _needs_gurobi()
    mu, cov = mu_cov
    fg = opt.frontier(mu, cov, n_points=5, solver="gurobi")
    fs = opt.frontier(mu, cov, n_points=5, solver="slsqp")
    assert len(fg) == len(fs) > 0
    assert fg["volatility"].values == pytest.approx(fs["volatility"].values, abs=1e-6)


@pytest.mark.parametrize("solver", ["auglag", "fista", "osqp"])
def test_alt_backend_max_sharpe_matches_slsqp(mu_cov, solver):
    pytest.importorskip("osqp" if solver == "osqp" else "numpy")
    mu, cov = mu_cov
    w = opt.max_sharpe(mu, cov, rf=0.0, solver=solver)
    ws = opt.max_sharpe(mu, cov, rf=0.0, solver="slsqp")
    assert _sharpe(w, mu, cov) == pytest.approx(_sharpe(ws, mu, cov), abs=1e-5)
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w.values >= -1e-6).all()  # first-order solvers stop at ~1e-8 tol


@pytest.mark.parametrize("solver", ["auglag", "fista", "osqp"])
def test_alt_backend_gmv_and_min_vol(mu_cov, solver):
    pytest.importorskip("osqp" if solver == "osqp" else "numpy")
    mu, cov = mu_cov
    wg = opt.gmv(mu, cov, solver=solver)
    ws = opt.gmv(mu, cov, solver="slsqp")
    assert np.abs(wg.values - ws.values).max() < 1e-3
    tr = float(mu.quantile(0.7))
    wt = opt.min_vol_at_return(mu, cov, tr, solver=solver)
    assert wt @ mu.values == pytest.approx(tr, abs=1e-5)


def test_fista_max_return_at_vol_via_bisection(mu_cov):
    mu, cov = mu_cov
    tv = float(np.sqrt(np.diag(cov.values)).mean())
    w = opt.max_return_at_vol(mu, cov, tv, solver="fista")
    ws = opt.max_return_at_vol(mu, cov, tv, solver="slsqp")
    assert (w @ mu.values) == pytest.approx(ws @ mu.values, abs=1e-5)
    assert np.sqrt(w.values @ cov.values @ w.values) <= tv * (1 + 1e-6)


def test_auto_uses_gurobi_when_available(mu_cov):
    _needs_gurobi()
    mu, cov = mu_cov
    w_auto = opt.max_sharpe(mu, cov, rf=0.0, solver="auto")
    w_g = opt.max_sharpe(mu, cov, rf=0.0, solver="gurobi")
    assert np.abs(w_auto.values - w_g.values).max() < 1e-9


# ------------------------------------------------- fallback behavior

def test_auto_falls_back_when_gurobi_missing(mu_cov, monkeypatch):
    mu, cov = mu_cov
    monkeypatch.setattr(gb, "_import_gurobi", _raise_unavailable)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        w = opt.max_sharpe(mu, cov, rf=0.0, solver="auto")
    assert not [x for x in rec if "Gurobi" in str(x.message)]  # silent: not installed is normal
    # auto chain is gurobi -> auglag -> fista -> osqp -> slsqp, so with gurobi
    # missing it must take auglag and match it exactly
    wf = opt.max_sharpe(mu, cov, rf=0.0, solver="auglag")
    assert np.abs(w.values - wf.values).max() < 1e-9


def _raise_unavailable():
    raise gb._GurobiUnavailable("gurobipy is not installed")


def test_solver_gurobi_raises_when_unavailable(mu_cov, monkeypatch):
    mu, cov = mu_cov
    monkeypatch.setattr(gb, "_import_gurobi", _raise_unavailable)
    with pytest.raises(RuntimeError, match="solver='gurobi'"):
        opt.max_sharpe(mu, cov, rf=0.0, solver="gurobi")


def test_capped_max_sharpe_falls_back_to_slsqp(mu_cov):
    """Per-asset cap ub<1 makes the QP reformulation inexact -> auto falls back."""
    _needs_gurobi()
    mu, cov = mu_cov
    with pytest.warns(RuntimeWarning, match="unavailable"):
        w = opt.max_sharpe(mu, cov, rf=0.0, bounds=(0.0, 0.5), solver="auto")
    assert (w.values <= 0.5 + 1e-9).all()
    ws = opt.max_sharpe(mu, cov, rf=0.0, bounds=(0.0, 0.5), solver="slsqp")
    assert np.abs(w.values - ws.values).max() < 1e-9


def test_max_sharpe_qp_rejects_nontrivial_bounds(mu_cov):
    _needs_gurobi()
    mu, cov = mu_cov
    with pytest.raises(gb._GurobiUnavailable):
        gb.max_sharpe_qp(mu, cov, 0.0, bounds=(0.0, 0.5))


def test_invalid_solver_raises(mu_cov):
    mu, cov = mu_cov
    with pytest.raises(ValueError, match="solver must be"):
        opt.gmv(mu, cov, solver="bogus")


def test_deprecated_solver_alm_alias_warns_and_matches(mu_cov):
    """solver="alm" still works (deprecated alias of "auglag")."""
    mu, cov = mu_cov
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        w_old = opt.max_sharpe(mu, cov, rf=0.0, solver="alm")
    assert any(issubclass(x.category, DeprecationWarning) for x in rec)
    w_new = opt.max_sharpe(mu, cov, rf=0.0, solver="auglag")
    assert np.abs(w_old.values - w_new.values).max() < 1e-9


def test_deprecated_auglag_function_aliases_warn():
    """Old optimize.auglag function names warn but still compute."""
    from portlab.optimize import auglag as ag
    C = np.eye(3)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        w = ag.gmv_alm(C)
    assert any(issubclass(x.category, DeprecationWarning) for x in rec)
    assert abs(w.sum() - 1.0) < 1e-9
