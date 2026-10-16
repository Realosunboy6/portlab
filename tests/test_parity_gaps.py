"""PV parity gap closures: asset-class mode, sectors, screener, CAPE timing."""

import numpy as np
import pandas as pd
import pytest


def test_sector_weights_static_map():
    # No network: AAPL/JNJ are in the static universe map.
    from portlab.data.sectors import sector_weights
    sw = sector_weights({"AAPL": 0.6, "JNJ": 0.4})
    assert sw["Information Technology"] == pytest.approx(0.6)
    assert sw["Health Care"] == pytest.approx(0.4)
    assert sw.sum() == pytest.approx(1.0)


def test_asset_class_proxies_known():
    from portlab.data.universe import asset_class_tickers
    m = asset_class_tickers(["US Stock Market", "Gold", "Cash (T-Bills)"])
    assert m == {"US Stock Market": "VTI", "Gold": "GLD", "Cash (T-Bills)": "BIL"}


def test_screener_runs():
    from portlab import analytics
    rng = np.random.default_rng(3)
    idx = pd.date_range("2022-01-31", periods=36, freq="ME")
    rets = pd.DataFrame(rng.normal(0.006, 0.05, (36, 5)), index=idx,
                        columns=["A", "B", "C", "D", "E"])
    tbl = analytics.screener(rets, periods=12, sort_by="Sharpe Ratio", top=3,
                             filters={"CAGR": (0.0, None)})
    assert len(tbl) <= 3
    assert (tbl["CAGR"] >= 0).all()


def test_cape_valuation_weights():
    from portlab import tactical
    idx = pd.date_range("2000-01-31", periods=300, freq="ME")
    # synthetic CAPE: cheap early, expensive late
    cape = pd.Series(np.linspace(15, 35, 300), index=idx, name="CAPE")
    w = tactical.cape_valuation(cape, idx, "SPY", "TLT", lookback_years=10)
    assert set(w.columns) == {"SPY", "TLT"}
    assert ((w.sum(axis=1) - 1.0).abs() < 1e-9).all()
    # cheap regime -> more equity than expensive regime
    assert w["SPY"].iloc[20] > w["SPY"].iloc[-1]


def test_arima_implied_mu_finite():
    from portlab.forecast import arima_forecast
    rng = np.random.default_rng(5)
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 500))),
                   index=pd.date_range("2023-01-01", periods=500, freq="B"))
    fc = arima_forecast(px, steps=21)
    mu_ann = float(fc["forecast"].iloc[-1] / px.iloc[-1] - 1) * 12
    assert np.isfinite(mu_ann)


def test_cape_positive_only():
    """Yale pre-fills future months with CAPE=0; loader must drop them."""
    import pandas as pd
    from portlab.data import macro

    raw = pd.DataFrame({
        "Date": ["2020.01", "2020.02", "2026.10"],
        "CAPE": [30.0, 31.0, 0.0],
    })
    dates = pd.to_datetime(raw["Date"], format="%Y.%m")
    s = pd.Series(pd.to_numeric(raw["CAPE"], errors="coerce").values, index=dates)
    s = s[s > 0]  # the rule enforced in get_shiller_cape
    assert (s > 0).all() and len(s) == 2
    assert macro.get_shiller_cape().gt(0).all()
