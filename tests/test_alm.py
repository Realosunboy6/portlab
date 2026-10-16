import numpy as np
import pandas as pd

from portlab import alm


def test_pv_matches_perpetuity():
    sched = alm.liability_schedule(1200, 300, growth=0.0)  # $100/mo, ~perpetuity
    pv = alm.present_value(sched, 0.06)
    perpetuity = 100 / ((1 + 0.06) ** (1 / 12) - 1)
    assert abs(pv - perpetuity) / perpetuity < 0.01


def test_funded_status_arithmetic():
    sched = alm.liability_schedule(12000, 10)
    st = alm.funded_status(100_000, sched, 0.05)
    assert st["funded_ratio"] == 100_000 / st["pv_liabilities"]
    assert st["surplus"] == 100_000 - st["pv_liabilities"]
    assert st["funded_ratio"] > 1.0  # $100k covers ten $12k payments at 5%


def _toy_market(seed=0):
    rng = np.random.default_rng(seed)
    R = rng.normal(0.0008, 0.012, (1500, 3))
    L = 0.5 * R[:, 0] + rng.normal(0.0002, 0.006, 1500)
    df = pd.DataFrame(np.column_stack([R, L]), columns=["A", "B", "C", "L"])
    mu = df[["A", "B", "C"]].mean() * 252
    cov = df[["A", "B", "C"]].cov() * 252
    liab_mu = df["L"].mean() * 252
    liab_var = df["L"].var() * 252
    cross = np.array([df[a].cov(df["L"]) for a in ["A", "B", "C"]]) * 252
    return mu, cov, liab_mu, liab_var, cross


def test_surplus_optimize_simplex_and_positive():
    mu, cov, liab_mu, liab_var, cross = _toy_market()
    assert (mu > liab_mu).any()
    w = alm.surplus_optimize(mu, cov, liab_mu, liab_var, cross)
    assert abs(w.sum() - 1) < 1e-6
    assert (w >= -1e-8).all()
    s_sharpe = (w @ mu - liab_mu) / np.sqrt(w @ cov @ w - 2 * w @ cross + liab_var)
    assert s_sharpe > 0


def test_hedge_portfolio_minimizes_surplus_var():
    mu, cov, liab_mu, liab_var, cross = _toy_market()
    w = alm.hedge_portfolio(cov, cross)
    assert abs(w.sum() - 1) < 1e-6
    ew = np.full(3, 1 / 3)
    svar = lambda x: x @ cov.values @ x - 2 * x @ cross + liab_var
    assert svar(w.values) <= svar(ew) + 1e-8


def test_funded_projection_monotone_pv():
    sched = alm.liability_schedule(12000, 10, start="2026-01-01")
    proj = alm.funded_projection(200_000, sched, 0.06, 0.04)
    assert (proj["pv_liabilities"].dropna().diff().dropna() <= 0).all()
    assert proj["funded_ratio"].iloc[0] > 1.0
