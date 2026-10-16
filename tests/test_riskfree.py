"""Risk-free rate handling: scalar or T-bill Series (PV convention)."""

import numpy as np
import pandas as pd
import pytest

from portlab import metrics as M

rng = np.random.default_rng(7)
IDX = pd.date_range("2020-01-31", periods=60, freq="ME")
RETS = pd.Series(rng.normal(0.008, 0.04, 60), index=IDX)


def test_series_rf_matches_scalar_when_flat():
    rf_series = pd.Series(0.03 / 12, index=IDX)
    assert M.sharpe(RETS, rf_series, 12) == pytest.approx(M.sharpe(RETS, 0.03, 12))
    assert M.sortino(RETS, rf_series, 12) == pytest.approx(M.sortino(RETS, 0.03, 12))


def test_series_rf_changes_result_when_varying():
    rising = pd.Series(np.linspace(0.0, 0.06 / 12, 60), index=IDX)
    flat = pd.Series(0.03 / 12, index=IDX)
    assert M.sharpe(RETS, rising, 12) != pytest.approx(M.sharpe(RETS, flat, 12))


def test_summary_accepts_series_rf():
    rf_series = pd.Series(0.02 / 12, index=IDX)
    s = M.summary(RETS, rf_series, 12)
    assert np.isfinite(s["Sharpe Ratio"])


def test_tbill_download():
    rf = pytest.importorskip("portlab.data.riskfree")
    tb = rf.get_tbill("2020-01-01", "2020-03-01")
    assert len(tb) > 30 and ((tb >= 0) & (tb < 0.2)).all()
    per = rf.period_rf(tb, IDX[:3], 12)
    assert len(per) == 3 and (per >= 0).all()
