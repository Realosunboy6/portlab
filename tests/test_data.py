"""Data-layer tests: get_returns must return complete first periods."""

import numpy as np
import pandas as pd
import pytest

from portlab.data import get_returns


@pytest.mark.regression
def test_first_return_is_complete():
    # ISSUE-001: 2017-01-03 was a trading day; the Jan 2017 monthly return
    # must include it. Before the fix the first period silently dropped its
    # leading trading day (PV cross-check 2026-10-08: Jan 2017 was 1.54%
    # vs PV's 2.22%).
    rets = get_returns(["SPY", "TLT", "GLD"], "2017-01-01", "2017-02-01")
    assert rets.index[0].date() == pd.Timestamp("2017-01-03").date()
    jan = (1 + rets.loc["2017-01"]).prod() - 1
    w = pd.Series({"SPY": 0.5, "TLT": 0.3, "GLD": 0.2})
    assert abs(float((jan * w).sum()) - 0.0222) < 0.002


def test_start_slice_respected():
    rets = get_returns(["SPY"], "2020-06-15", "2020-07-15")
    assert (rets.index >= pd.Timestamp("2020-06-15")).all()
    assert rets.index[0].date() == pd.Timestamp("2020-06-15").date()
    assert np.isfinite(rets.values).all()
