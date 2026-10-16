"""Tests for the GitHub dataset-mirror loader (network mocked)."""

import numpy as np
import pandas as pd
import pytest

from portlab.data import github_mirrors as gm


@pytest.fixture(autouse=True)
def stub_fetch(monkeypatch):
    rng = np.random.default_rng(7)
    months = pd.date_range("1950-01-01", "2026-06-01", freq="MS")

    frames = {}

    def fake_fetch(name, refresh=False):
        if name in frames:
            return frames[name]
        frames[name] = _build(name)
        return frames[name]

    def _build(name):
        if name == "shiller":
            px = 20 * np.exp(np.cumsum(rng.normal(0.006, 0.04, len(months))))
            cpi = 24 * 1.003 ** np.arange(len(months))
            cpi[-30:] = 0.0                      # trailing placeholder zeros
            return pd.DataFrame({
                "Date": months.strftime("%Y-%m-%d"), "SP500": px,
                "Dividend": px * 0.03, "Earnings": px * 0.05,
                "Consumer Price Index": cpi, "Long Interest Rate": 4.0,
                "PE10": 15 + rng.normal(0, 3, len(months))})
        if name == "us10y":
            return pd.DataFrame({"Date": months.strftime("%Y-%m-%d"),
                                 "Rate": 4.5 + np.cumsum(rng.normal(0, 0.1, len(months))).clip(-3, 8)})
        if name == "gold":
            return pd.DataFrame({"Date": months.strftime("%Y-%m"),
                                 "Price": 35 * np.exp(np.cumsum(rng.normal(0.004, 0.05, len(months))))})
        raise AssertionError(name)

    monkeypatch.setattr(gm, "_fetch_csv", fake_fetch)


def test_sp500_total_returns_include_dividends():
    tr = gm.sp500_total_returns()
    sp = gm.shiller_sp500()
    price_only = (sp["SP500"] / sp["SP500"].shift(1) - 1).dropna()
    assert (tr - price_only).mean() > 0        # dividend yield adds return


def test_cpi_drops_placeholder_zeros():
    cpi = gm.cpi_series()
    assert (cpi > 0).all()
    assert cpi.index[-1] < pd.Timestamp("2026-06-01")


def test_cape_available():
    cape = gm.shiller_cape()
    assert len(cape) > 500 and cape.median() > 5


def test_treasury_returns_reasonable():
    tr = gm.treasury10y_returns()
    assert tr.abs().max() < 0.25
    assert len(tr) > 800


def test_mirror_panel_aligned():
    panel = gm.get_mirror_returns(start="1972-01-01")
    assert list(panel.columns) == ["US Stocks (S&P 500 TR)", "10Y Treasury", "Gold"]
    assert panel.index.min().year == 1972
    assert not panel.isna().any().any()
