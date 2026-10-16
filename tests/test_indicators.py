import numpy as np
import pandas as pd

from portlab import indicators as I


def _ohlc(n=300, seed=0):
    rng = np.random.default_rng(seed)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n))),
                      index=pd.date_range("2024-01-01", periods=n, freq="B"))
    high = close * (1 + rng.uniform(0, 0.005, n))
    low = close * (1 - rng.uniform(0, 0.005, n))
    return high, low, close


def test_rsi_bounds():
    _, _, close = _ohlc()
    r = I.rsi(close).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_macd_flat_series_near_zero():
    close = pd.Series(100.0, index=pd.date_range("2024-01-01", periods=200, freq="B"))
    m = I.macd(close).dropna()
    assert (m.abs() < 1e-8).all().all()


def test_bollinger_ordering():
    _, _, close = _ohlc()
    b = I.bollinger(close).dropna()
    assert ((b["Lower"] <= b["Middle"]) & (b["Middle"] <= b["Upper"])).all().all()
    # %B may leave [0, 1] on breakouts; it must stay finite
    assert np.isfinite(b["%B"]).all()


def test_atr_positive():
    high, low, close = _ohlc()
    a = I.atr(high, low, close).dropna()
    assert (a > 0).all()


def test_stochastic_bounds():
    high, low, close = _ohlc()
    s = I.stochastic(high, low, close).dropna()
    assert (((s >= 0) & (s <= 100)) | s.isna()).all().all()
