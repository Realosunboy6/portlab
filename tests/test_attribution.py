"""Tests for portlab.attribution.

The Bacon (2008) Ch.5 worked example below is PUBLISHED data (via the
pybrinson README's 30-second demo, MIT-licensed): it is the external oracle
for the classic BHB numbers. Everything else is synthetic.
"""

import numpy as np
import pandas as pd
import pytest

from portlab import attribution as AT


def _bacon_segments():
    """Bacon (2008), Practical Portfolio Performance Measurement and
    Attribution, ch. 5 — via pybrinson's README demo."""
    return pd.DataFrame([
        {"segment": "UK Equity", "w_p": 0.40, "w_b": 0.40,
         "r_p": 0.20, "r_b": 0.10},
        {"segment": "Japan Equity", "w_p": 0.30, "w_b": 0.20,
         "r_p": -0.05, "r_b": -0.04},
        {"segment": "US Equity", "w_p": 0.30, "w_b": 0.40,
         "r_p": 0.06, "r_b": 0.08},
    ])


def test_bacon_ch5_published_numbers():
    """EXTERNAL ORACLE: must reproduce Bacon's published effects exactly."""
    out = AT.brinson(_bacon_segments())
    t = out["table"]
    #                  alloc    select    inter     total
    exp = {"UK Equity":    (0.0000,  0.0400,  0.0000,  0.0400),
           "Japan Equity": (-0.0040, -0.0020, -0.0010, -0.0070),
           "US Equity":    (-0.0080, -0.0080,  0.0020, -0.0140)}
    for seg, (a, s, i, tot) in exp.items():
        assert t.loc[seg, "allocation"] == pytest.approx(a, abs=1e-12)
        assert t.loc[seg, "selection"] == pytest.approx(s, abs=1e-12)
        assert t.loc[seg, "interaction"] == pytest.approx(i, abs=1e-12)
        assert t.loc[seg, "total"] == pytest.approx(tot, abs=1e-12)
    assert out["portfolio_return"] == pytest.approx(0.083)
    assert out["benchmark_return"] == pytest.approx(0.064)
    assert out["excess_return"] == pytest.approx(0.019)
    assert out["allocation"] == pytest.approx(-0.012)
    assert out["selection"] == pytest.approx(0.030)
    assert out["interaction"] == pytest.approx(0.001)


def test_identity_holds_on_random_inputs():
    rng = np.random.default_rng(20261010)
    for _ in range(20):
        n = int(rng.integers(2, 8))
        w_p = rng.random(n); w_p /= w_p.sum()
        w_b = rng.random(n); w_b /= w_b.sum()
        df = pd.DataFrame({"segment": [f"S{i}" for i in range(n)],
                           "w_p": w_p, "w_b": w_b,
                           "r_p": rng.normal(0, 0.1, n),
                           "r_b": rng.normal(0, 0.1, n)})
        out = AT.brinson(df)
        assert abs(out["allocation"] + out["selection"] + out["interaction"]
                   - out["excess_return"]) < 1e-9


def test_fachler_variant():
    # allocation measured vs the overall benchmark return instead of the
    # segment's: overweighting a segment that beat R_b is rewarded.
    out = AT.brinson(_bacon_segments(), method="fachler")
    t = out["table"]
    # Japan: (0.30-0.20) x (-0.04-0.064) = -0.0104
    assert t.loc["Japan Equity", "allocation"] == pytest.approx(-0.0104)
    # UK: zero active weight -> zero allocation under either method
    assert t.loc["UK Equity", "allocation"] == pytest.approx(0.0)
    # identity still closes under Fachler
    assert abs(out["allocation"] + out["selection"] + out["interaction"]
               - out["excess_return"]) < 1e-9


def test_accepts_list_of_dicts():
    out = AT.brinson([{"segment": "A", "w_p": 0.6, "w_b": 0.5,
                       "r_p": 0.10, "r_b": 0.05},
                      {"segment": "B", "w_p": 0.4, "w_b": 0.5,
                       "r_p": 0.05, "r_b": 0.10}])
    assert out["excess_return"] == pytest.approx(
        (0.6 * 0.10 + 0.4 * 0.05) - (0.5 * 0.05 + 0.5 * 0.10))


def test_weight_validation():
    bad = _bacon_segments()
    bad.loc[0, "w_p"] = 0.99
    with pytest.raises(ValueError, match="sum to"):
        AT.brinson(bad)
    with pytest.raises(ValueError, match="at least one segment"):
        AT.brinson(pd.DataFrame(columns=["segment", "w_p", "w_b", "r_p", "r_b"]))
    with pytest.raises(ValueError, match="method"):
        AT.brinson(_bacon_segments(), method="brinson-fachler-smith")
