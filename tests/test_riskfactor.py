"""Risk-factor-based allocation (PV Risk Factor Based Allocation parity).

All tests use deterministic synthetic data — no network required.
"""

import numpy as np
import pandas as pd
import pytest

from portlab.optimize import riskfactor as rfa


def _loadings():
    # SYNTHETIC loadings, shaped like PV's worked example
    # (Market, Size, Value, Term Risk, Credit Risk).
    return pd.DataFrame(
        {
            "Mkt": [1.00, 1.00, 0.00, 0.00],
            "SMB": [0.00, 0.60, 0.00, 0.00],
            "HML": [0.00, 0.50, 0.00, 0.00],
            "TRM": [0.00, 0.00, 0.60, 0.55],
            "CDT": [0.00, 0.00, 0.00, 0.40],
        },
        index=["EQ", "SCV", "IEF", "LQD"],
    )


def test_exact_targets_attainable():
    # SYNTHETIC: target is an exact convex combination of loadings.
    B = pd.DataFrame({"F1": [1.0, 0.0, 0.5], "F2": [0.0, 1.0, 0.5]},
                     index=["A", "B", "C"])
    tgt = pd.Series({"F1": 0.6, "F2": 0.4})
    res = rfa.risk_factor_allocation(B, tgt)
    assert res["objective"] == pytest.approx(0.0, abs=1e-8)
    assert res["weights"].sum() == pytest.approx(1.0)
    assert (res["weights"] >= -1e-9).all() and (res["weights"] <= 1 + 1e-9).all()
    assert res["gap"].abs().max() < 1e-6


def test_pv_example_direction():
    # SYNTHETIC, mirrors PV's gallery example: hold market/size, raise value
    # (HML 0.125 -> 0.20), cut term risk (TRM 0.2875 -> 0.10).
    # Exact match is infeasible long-only, so assert directional improvement.
    B = _loadings()
    w0 = pd.Series(0.25, index=B.index)
    exist = rfa.portfolio_exposures(w0, B)
    tgt = pd.Series({"Mkt": 0.50, "SMB": 0.15, "HML": 0.20,
                     "TRM": 0.10, "CDT": 0.15})
    res = rfa.risk_factor_allocation(B, tgt, current_weights=w0)
    assert res["weights"].sum() == pytest.approx(1.0)
    assert (res["weights"] >= -1e-9).all()
    # moved toward the value target and away from term risk
    assert res["achieved"]["HML"] > exist["HML"]
    assert res["achieved"]["TRM"] < exist["TRM"]
    # lower total squared exposure gap than the starting portfolio
    # (least squares trades factors off against each other)
    start_sse = float(((exist - tgt) ** 2).sum())
    assert res["objective"] < start_sse


def test_factor_weights_prioritize():
    # SYNTHETIC: heavy weight on F1 must shrink the F1 gap vs equal weighting.
    B = pd.DataFrame({"F1": [1.0, 0.2], "F2": [0.2, 1.0]}, index=["A", "B"])
    tgt = pd.Series({"F1": 0.9, "F2": 0.9})  # infeasible: F1+F2 = 1.2 for any
    # feasible w, but the target sums to 1.8
    eq = rfa.risk_factor_allocation(B, tgt)
    pri = rfa.risk_factor_allocation(
        B, tgt, factor_weights=pd.Series({"F1": 100.0, "F2": 1.0}))
    assert abs(pri["gap"]["F1"]) < abs(eq["gap"]["F1"])
    assert pri["weights"].sum() == pytest.approx(1.0)


def test_max_turnover_cap():
    # SYNTHETIC: turnover cap must bind.
    B = _loadings()
    w0 = pd.Series({"EQ": 1.0, "SCV": 0.0, "IEF": 0.0, "LQD": 0.0})
    tgt = pd.Series({"Mkt": 0.0, "SMB": 0.0, "HML": 0.0, "TRM": 0.6, "CDT": 0.0})
    res = rfa.risk_factor_allocation(B, tgt, current_weights=w0, max_turnover=0.2)
    assert res["turnover"] <= 0.2 + 1e-6
    assert res["weights"].sum() == pytest.approx(1.0)


def test_allocate_from_returns_synthetic():
    # SYNTHETIC end-to-end: known loadings -> regressions recover them ->
    # allocation reduces the exposure gap vs equal weight. No network.
    rng = np.random.default_rng(7)
    n, f = 60, 3
    dates = pd.date_range("2020-01-31", periods=n, freq="ME")
    fac = pd.DataFrame(rng.normal(0, 0.04, (n, f)), index=dates,
                       columns=["Mkt-RF", "SMB", "HML"])
    fac["RF"] = 0.001
    B_true = np.array([[1.0, 0.0, 0.0],
                       [1.0, 0.5, 0.4],
                       [0.1, 0.0, 0.0],
                       [0.0, 0.0, 0.3]])
    noise = rng.normal(0, 0.005, (n, 4))
    rets = pd.DataFrame(fac[["Mkt-RF", "SMB", "HML"]].values @ B_true.T + noise,
                        index=dates, columns=["A", "B", "C", "D"])
    tgt = {"Mkt-RF": 0.9, "SMB": 0.3, "HML": 0.25}
    res = rfa.allocate_from_returns(rets, fac, tgt, model="ff3", window=48)
    assert res["weights"].sum() == pytest.approx(1.0)
    assert (res["weights"] >= -1e-9).all()
    assert res["n_obs"] == 48
    # loadings recovered close to truth for the high-R² assets
    assert abs(res["loadings"].loc["A", "Mkt-RF"] - 1.0) < 0.15
    # allocation beats equal weight on the objective
    w_eq = pd.Series(0.25, index=rets.columns)
    gap_eq = rfa.exposure_gap(w_eq, res["loadings"], pd.Series(tgt))
    assert res["objective"] <= float((gap_eq ** 2).sum()) + 1e-9


def test_no_common_factor_raises():
    B = pd.DataFrame({"F1": [1.0, 0.5]}, index=["A", "B"])
    with pytest.raises(ValueError):
        rfa.risk_factor_allocation(B, pd.Series({"ZZZ": 1.0}))
