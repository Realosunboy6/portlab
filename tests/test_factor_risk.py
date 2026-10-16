"""Tests for the factor risk decomposition in portlab.factor.

ALL DATA HERE IS SYNTHETIC except the external-verification step (run
separately): the decomposition was cross-checked on real Ken French FF3
factors + live Yahoo returns for the pilot stocks.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from portlab import factor as F


@pytest.fixture()
def rng():
    return np.random.default_rng(20261010)


def test_portfolio_exposures_hand_check():
    w = pd.Series({"A": 0.6, "B": 0.4})
    betas = pd.DataFrame({"Mkt": [1.2, 0.8], "SMB": [-0.3, 0.5]},
                         index=["A", "B"])
    x = F.portfolio_factor_exposures(w, betas)
    assert x["Mkt"] == pytest.approx(0.6 * 1.2 + 0.4 * 0.8)
    assert x["SMB"] == pytest.approx(0.6 * -0.3 + 0.4 * 0.5)


def test_decomposition_identities_random(rng):
    n, k = 6, 3
    syms = [f"S{i}" for i in range(n)]
    w = pd.Series(rng.random(n), index=syms)
    w /= w.sum()
    betas = pd.DataFrame(rng.normal(0, 1, (n, k)), index=syms,
                         columns=["Mkt", "SMB", "HML"])
    A = rng.normal(0, 1, (k, k))
    fcov = pd.DataFrame(A @ A.T + np.eye(k) * 0.1,
                        index=betas.columns, columns=betas.columns)
    idio = pd.Series(rng.uniform(0.001, 0.01, n), index=syms)
    out = F.factor_risk_decomposition(w, betas, fcov, idio)
    assert out["systematic_var"] + out["idiosyncratic_var"] == \
        pytest.approx(out["total_var"], rel=1e-10)
    assert out["factor_contrib"].sum() == \
        pytest.approx(out["systematic_var"] / 1.0, rel=1e-10)
    assert out["asset_contrib"].sum() == pytest.approx(out["total_var"],
                                                       rel=1e-10)
    assert 0 <= out["systematic_pct"] <= 1 or np.isnan(out["systematic_pct"])
    assert out["total_vol"] == pytest.approx(np.sqrt(out["total_var"]))


def test_capm_two_asset_hand_check():
    # one factor, two assets: everything checkable by hand
    w = pd.Series({"A": 0.5, "B": 0.5})
    betas = pd.DataFrame({"Mkt": [1.5, 0.5]}, index=["A", "B"])
    fcov = pd.DataFrame([[0.04]], index=["Mkt"], columns=["Mkt"])
    idio = pd.Series({"A": 0.01, "B": 0.02})
    out = F.factor_risk_decomposition(w, betas, fcov, idio)
    # exposure = 0.5*1.5 + 0.5*0.5 = 1.0
    assert out["exposures"]["Mkt"] == pytest.approx(1.0)
    # systematic = 1.0^2 * 0.04 = 0.04; idio = 0.25*0.01 + 0.25*0.02 = 0.0075
    assert out["systematic_var"] == pytest.approx(0.04)
    assert out["idiosyncratic_var"] == pytest.approx(0.0075)
    assert out["total_var"] == pytest.approx(0.0475)
    assert out["factor_contrib"]["Mkt"] == pytest.approx(0.04)
    # asset contributions: w_i * (Sigma w)_i
    # Sigma = B F B' + D; B F B' = 0.04 * [[2.25, .75],[.75,.25]]
    s11, s12, s22 = 0.04 * 2.25 + 0.01, 0.04 * 0.75, 0.04 * 0.25 + 0.02
    assert out["asset_contrib"]["A"] == pytest.approx(0.5 * (s11 * 0.5 + s12 * 0.5))
    assert out["asset_contrib"]["B"] == pytest.approx(0.5 * (s12 * 0.5 + s22 * 0.5))


def test_scale_annualizes():
    w = pd.Series({"A": 1.0})
    betas = pd.DataFrame({"Mkt": [1.0]}, index=["A"])
    fcov = pd.DataFrame([[0.04]], index=["Mkt"], columns=["Mkt"])
    out = F.factor_risk_decomposition(w, betas, fcov, 0.0, scale=252.0)
    assert out["total_var"] == pytest.approx(0.04 * 252.0)
    assert out["total_vol"] == pytest.approx(np.sqrt(0.04 * 252.0))


def test_decompose_from_returns_matches_statsmodels(rng):
    """Independent path: betas via statsmodels OLS vs the lstsq inside
    decompose_from_returns — two implementations of the same regression."""
    n, t = 4, 300
    idx = pd.bdate_range("2020-01-01", periods=t)
    fm = pd.DataFrame(rng.normal(0, 0.02, (t, 3)), index=idx,
                      columns=["Mkt-RF", "SMB", "HML"])
    fm["RF"] = 0.0001
    true_b = rng.normal(1, 0.5, (n, 3))
    syms = [f"S{i}" for i in range(n)]
    rets = pd.DataFrame(
        {s: 0.0001 + fm[["Mkt-RF", "SMB", "HML"]].to_numpy() @ true_b[i]
         + rng.normal(0, 0.01, t) for i, s in enumerate(syms)}, index=idx)
    w = pd.Series([0.25] * n, index=syms)
    out = F.decompose_from_returns(w, rets, fm, model="ff3")

    for s in syms:
        y = rets[s] - fm["RF"]
        X = sm.add_constant(fm[["Mkt-RF", "SMB", "HML"]])
        fit = sm.OLS(y, X).fit()
        assert np.allclose(out["betas"].loc[s].to_numpy(),
                           fit.params[["Mkt-RF", "SMB", "HML"]].to_numpy(),
                           rtol=1e-8)
    # identities still hold end to end
    assert out["systematic_var"] + out["idiosyncratic_var"] == \
        pytest.approx(out["total_var"], rel=1e-9)
    assert out["n_obs"] == t


def test_decompose_from_returns_validates(rng):
    idx = pd.bdate_range("2020-01-01", periods=10)
    fm = pd.DataFrame(rng.normal(0, 0.02, (10, 4)), index=idx,
                      columns=["Mkt-RF", "SMB", "HML", "RF"])
    rets = pd.DataFrame(rng.normal(0, 0.02, (10, 2)), index=idx,
                        columns=["A", "B"])
    with pytest.raises(ValueError, match="not enough"):
        F.decompose_from_returns({"A": 0.5, "B": 0.5}, rets, fm)
    with pytest.raises(ValueError, match="model"):
        F.decompose_from_returns({"A": 1.0}, rets, fm, model="ff6")
