import numpy as np
import pandas as pd
import pytest

from portlab import optimize as opt
from portlab.covariance import corr_from_cov, get_cov, psd_fix


def _valid_weights(w, lo=0.0, hi=1.0):
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= lo - 1e-6).all() and (w <= hi + 1e-6).all()


def test_gmv_is_minimum_variance(mu_cov):
    mu, cov = mu_cov
    w_gmv = opt.gmv(mu, cov)
    _valid_weights(w_gmv)
    gmv_vol = np.sqrt(w_gmv @ cov @ w_gmv)
    fr = opt.frontier(mu, cov, n_points=12)
    assert (fr["volatility"] >= gmv_vol - 1e-6).all()


def test_max_sharpe_beats_gmv(mu_cov):
    mu, cov = mu_cov
    rf = 0.02
    w_ms = opt.max_sharpe(mu, cov, rf=rf)
    w_gmv = opt.gmv(mu, cov)
    _valid_weights(w_ms)
    s = lambda w: (w @ mu - rf) / np.sqrt(w @ cov @ w)
    assert s(w_ms) >= s(w_gmv) - 1e-8


def test_min_vol_at_return_hits_target(mu_cov):
    mu, cov = mu_cov
    target = float(mu.mean())
    w = opt.min_vol_at_return(mu, cov, target)
    assert w @ mu == pytest.approx(target, abs=1e-5)


def test_ledoit_wolf_psd(returns):
    cov = get_cov(returns, method="ledoit_wolf")
    eig = np.linalg.eigvalsh(cov.values)
    assert eig.min() >= -1e-10


def test_psd_fix_repairs_negative_eigenvalue():
    bad = pd.DataFrame([[1.0, 0.999, 0.0], [0.999, 1.0, 0.999], [0.0, 0.999, 1.0]])
    fixed = psd_fix(bad)
    assert np.linalg.eigvalsh(fixed.values).min() >= 0


def test_corr_from_cov_diag_one(mu_cov):
    _, cov = mu_cov
    corr = corr_from_cov(cov)
    assert np.allclose(np.diag(corr.values), 1.0)


def test_min_cvar_valid_and_lower_cvar_than_ew(returns):
    from portlab.metrics import cvar_historical
    w = opt.min_cvar(returns, alpha=0.95)
    _valid_weights(w)
    ew = pd.Series(1 / returns.shape[1], index=returns.columns)
    assert cvar_historical(returns @ w) <= cvar_historical(returns @ ew) + 1e-9


def test_risk_parity_equalizes_contributions(mu_cov):
    _, cov = mu_cov
    w = opt.equal_risk_contribution(cov)
    _valid_weights(w)
    rc = opt.risk_contributions(w, cov)
    assert rc.max() - rc.min() < 0.02


def test_kelly_valid(returns):
    w = opt.kelly(returns)
    _valid_weights(w)


def test_max_sortino_valid(returns):
    w = opt.max_sortino(returns, rf=0.0)
    _valid_weights(w)


def test_min_max_drawdown_beats_worst_asset(returns):
    from portlab.metrics import max_drawdown
    w = opt.min_max_drawdown(returns)
    _valid_weights(w)
    port_mdd = max_drawdown(returns @ w)
    worst_single = min(max_drawdown(returns[c]) for c in returns.columns)
    assert port_mdd >= worst_single  # diversified dd no worse than worst asset


def test_min_tracking_error_replicates_benchmark(returns):
    bench = returns @ np.full(returns.shape[1], 1 / returns.shape[1])
    w = opt.min_tracking_error(returns, bench)
    active = (returns @ w) - bench
    assert active.std() < 1e-4  # benchmark is replicable -> ~zero TE


def test_black_litterman_no_views_recovers_prior(mu_cov):
    mu, cov = mu_cov
    mkt = pd.Series(1.0, index=mu.index)
    prior = opt.implied_returns(cov, mkt, rf=0.02)
    P = pd.DataFrame(np.zeros((1, len(mu))), columns=mu.index)
    P.iloc[0, 0] = 1.0
    view = pd.Series([float(prior.iloc[0])])
    post = opt.black_litterman(cov, mkt, P, view, rf=0.02)
    # a view equal to the prior should barely move the posterior
    assert np.abs(post - prior).max() < 0.01


def test_resampled_weights_valid(returns):
    w = opt.resampled_weights(returns, n_samples=8)
    _valid_weights(w)


def test_group_constraints_respected(mu_cov):
    mu, cov = mu_cov
    groups = {"first_half": (list(range(4)), 0.0, 0.30)}
    w = opt.gmv(mu, cov, groups=groups)
    assert w.iloc[:4].sum() <= 0.30 + 1e-5
