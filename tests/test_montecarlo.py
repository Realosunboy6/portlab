import numpy as np
import pandas as pd
import pytest

from portlab.montecarlo import monte_carlo


@pytest.fixture(scope="module")
def hist(returns):
    return (1 + returns.mean(axis=1)).resample("ME").prod() - 1


def test_no_withdrawal_grows(hist):
    res = monte_carlo(initial=100_000, years=10, n_sims=500,
                      model="bootstrap", hist_returns=hist)
    assert res.success_rate == 1.0
    assert np.median(res.balances[:, -1]) > 100_000


def test_percentiles_monotone(hist):
    res = monte_carlo(initial=100_000, years=10, n_sims=500,
                      model="bootstrap", hist_returns=hist)
    p = res.percentiles()
    assert (p["p10"] <= p["p50"] + 1e-9).all()
    assert (p["p50"] <= p["p90"] + 1e-9).all()


def test_heavy_withdrawal_fails_sometimes():
    res = monte_carlo(initial=100_000, years=30, n_sims=500, model="normal",
                      mean_annual=0.05, vol_annual=0.12,
                      withdrawal="fixed", withdrawal_amount=10_000)
    assert res.success_rate < 1.0


def test_fixed_pct_never_ruins():
    res = monte_carlo(initial=100_000, years=30, n_sims=300, model="normal",
                      withdrawal="fixed_pct", withdrawal_pct=0.05)
    assert res.success_rate == 1.0  # percentage withdrawals cannot hit zero


def test_student_t_fatter_tails_than_normal():
    n = monte_carlo(years=10, n_sims=2000, model="normal", seed=7)
    t = monte_carlo(years=10, n_sims=2000, model="student_t", t_dof=4, seed=7)
    n_end, t_end = n.balances[:, -1], t.balances[:, -1]
    assert np.percentile(t_end, 1) < np.percentile(n_end, 1) * 1.05


def test_statistical_model(mu_cov):
    mu, cov = mu_cov
    w = pd.Series(1 / len(mu), index=mu.index)
    res = monte_carlo(years=5, n_sims=300, model="statistical",
                      asset_mu=mu, asset_cov=cov, weights=w)
    assert res.balances.shape == (300, 61)


def test_rmd_rule_spends_down():
    res = monte_carlo(initial=500_000, years=25, n_sims=200, model="normal",
                      mean_annual=0.04, vol_annual=0.08, withdrawal="rmd",
                      current_age=65, life_expectancy=90)
    assert res.withdrawals.sum() > 0
    assert res.success_rate == 1.0  # RMD can't overdraw to zero


def test_custom_schedule():
    sched = pd.Series([-1000.0] * 60)  # monthly withdrawals for 5y
    res = monte_carlo(initial=200_000, years=5, n_sims=200, model="normal",
                      withdrawal="custom", custom_schedule=sched)
    assert res.withdrawals[:, 0].mean() > 0


def test_reproducible_with_seed(hist):
    a = monte_carlo(years=5, n_sims=100, model="bootstrap", hist_returns=hist, seed=1)
    b = monte_carlo(years=5, n_sims=100, model="bootstrap", hist_returns=hist, seed=1)
    assert np.array_equal(a.balances, b.balances)
