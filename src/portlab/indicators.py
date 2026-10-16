"""Technical indicators: RSI (Wilder), MACD, Bollinger Bands, ATR (Wilder),
and Stochastic oscillator. All take OHLC-style Series/DataFrames and return
new Series/DataFrames aligned to the input index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder's smoothing; 0–100."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.rename(f"RSI_{window}")


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"MACD": line, "Signal": sig,
                         "Histogram": line - sig})


def bollinger(close: pd.Series, window: int = 20,
              num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands plus %B (position within the bands)."""
    mid = close.rolling(window).mean()
    sd = close.rolling(window).std()
    upper, lower = mid + num_std * sd, mid - num_std * sd
    pct_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame({"Middle": mid, "Upper": upper, "Lower": lower,
                         "%B": pct_b})


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        window: int = 14) -> pd.Series:
    """Average True Range with Wilder's smoothing."""
    prev_close = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    out = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return out.rename(f"ATR_{window}")


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k: int = 14, d: int = 3) -> pd.DataFrame:
    """Stochastic oscillator %K and %D; 0–100."""
    lowest = low.rolling(k).min()
    highest = high.rolling(k).max()
    pct_k = 100.0 * (close - lowest) / (highest - lowest).replace(0.0, np.nan)
    pct_d = pct_k.rolling(d).mean()
    return pd.DataFrame({"%K": pct_k, "%D": pct_d})
