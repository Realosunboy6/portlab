"""Free factor data: Ken French library (via pandas_datareader) and AQR.

Returns are converted from percent to decimals, indexed by period end, ready
for portlab.factor regressions.
"""

from __future__ import annotations

import pandas as pd

from . import cache

# pandas_datareader famafrench dataset names
_FF_DATASETS = {
    ("ff3", "monthly"): "F-F_Research_Data_Factors",
    ("ff3", "daily"):   "F-F_Research_Data_Factors_daily",
    ("ff5", "monthly"): "F-F_Research_Data_5_Factors_2x3",
    ("ff5", "daily"):   "F-F_Research_Data_5_Factors_2x3_daily",
    ("mom", "monthly"): "F-F_Momentum_Factor",
    ("mom", "daily"):   "F-F_Momentum_Factor_daily",
}


def get_ff_factors(model: str = "ff3", freq: str = "monthly",
                   start: str = "1963-07-01", refresh: bool = False) -> pd.DataFrame:
    """Fama-French factors (+ momentum for carhart/ff5_mom) in decimals.

    model: 'capm' | 'ff3' | 'carhart' | 'ff5' | 'ff5_mom'.
    """
    from pandas_datareader.data import DataReader

    base = "ff5" if model in ("ff5", "ff5_mom") else "ff3"
    want_mom = model in ("carhart", "ff5_mom")
    key = cache.cache_key("ff", model, freq, start)
    if not refresh:
        hit = cache.load(key)
        if hit is not None:
            return hit

    ds = DataReader(_FF_DATASETS[(base, freq)], "famafrench", start=start)[0]
    if want_mom:
        mom = DataReader(_FF_DATASETS[("mom", freq)], "famafrench", start=start)[0]
        mom.columns = ["Mom"]
        ds = ds.join(mom, how="inner")
    ds = ds / 100.0
    if isinstance(ds.index, pd.PeriodIndex):
        ds.index = ds.index.to_timestamp(how="end").normalize()
    ds.columns = [c.strip() for c in ds.columns]
    cache.save(key, ds)
    return ds


# AQR monthly datasets (long CSV headers; parsed lazily).
AQR_URLS = {
    "QMJ": "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Quality-Minus-Junk-Factors-Monthly.xlsx",
    "BAB": "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Betting-Against-Beta-Equity-Factors-Monthly.xlsx",
}


def get_aqr_factor(name: str, sheet: str = "QMJ Factors",
                   country: str = "USA") -> pd.Series:
    """AQR factor series (e.g. quality-minus-junk). Requires openpyxl."""
    if name not in AQR_URLS:
        raise ValueError(f"name must be one of {sorted(AQR_URLS)}")
    key = cache.cache_key("aqr", name, country)
    hit = cache.load(key)
    if hit is not None:
        return hit.iloc[:, 0]
    raw = pd.read_excel(AQR_URLS[name], sheet_name=sheet, skiprows=18, index_col=0)
    s = raw[country].dropna()
    s.index = pd.to_datetime(s.index)
    cache.save(key, s.to_frame(name))
    return s
