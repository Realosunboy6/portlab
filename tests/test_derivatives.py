import numpy as np

from portlab import derivatives as D


def test_bs_known_value():
    # Hull textbook example: S=42, K=40, T=0.5, r=0.10, sigma=0.20 -> call 4.76
    assert abs(D.bs_price(42, 40, 0.5, 0.10, 0.20, "call") - 4.76) < 0.01


def test_put_call_parity():
    c = D.bs_price(100, 100, 1.0, 0.05, 0.25, "call")
    p = D.bs_price(100, 100, 1.0, 0.05, 0.25, "put")
    assert abs(c - p - (100 - 100 * np.exp(-0.05))) < 1e-10


def test_greeks_sanity():
    g = D.bs_greeks(100, 100, 1.0, 0.05, 0.20, "call")
    assert 0 < g["delta"] < 1
    assert g["gamma"] > 0
    assert g["theta"] < 0  # long call bleeds time value
    assert g["vega"] > 0
    gp = D.bs_greeks(100, 100, 1.0, 0.05, 0.20, "put")
    assert -1 < gp["delta"] < 0


def test_implied_vol_roundtrip():
    for kind in ("call", "put"):
        px = D.bs_price(100, 105, 0.75, 0.03, 0.30, kind)
        assert abs(D.implied_vol(px, 100, 105, 0.75, 0.03, kind) - 0.30) < 1e-6


def test_binomial_converges_to_bs():
    euro = D.binomial_price(100, 100, 1.0, 0.05, 0.20, n=400, kind="call")
    bs = D.bs_price(100, 100, 1.0, 0.05, 0.20, "call")
    assert abs(euro - bs) < 0.02
    amer_put = D.binomial_price(100, 100, 1.0, 0.05, 0.20, n=400,
                               kind="put", american=True)
    euro_put = D.bs_price(100, 100, 1.0, 0.05, 0.20, "put")
    assert amer_put >= euro_put - 1e-9  # early exercise premium >= 0
