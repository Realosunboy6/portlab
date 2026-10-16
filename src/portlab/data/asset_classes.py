"""Long-history asset-class returns from free academic/government data.

These series are stitched from free academic and government data, reaching
back to 1926 where the data allows. How each one is built:

  - US equity classes: Ken French library (1926+). Market = Mkt-RF + RF;
    size/value cells from the 6 Portfolios Formed on Size and Book-to-Market.
  - Treasuries: synthetic total returns from FRED constant-maturity yields
    using the standard duration approximation
    ret_m ≈ y_prev/12 + D_mod * (y_prev - y_cur). Good to first order; ignores
    convexity and roll-down.
  - Cash: 3-month T-bill rate (TB3MS) accrued monthly.
  - Gold: London fix from FRED when available, else GLD/GC=F via yfinance.

These are *proxy* series for research — not investable fund histories.
"""

from __future__ import annotations

import pandas as pd

from . import cache

# FRED constant-maturity yield series and assumed modified durations.
_TREASURY_SPECS = {
    "Long Term Treasury":         ("GS20", 15.0),
    "10-Year Treasury":           ("GS10", 8.5),
    "Intermediate Treasury":      ("GS5", 4.5),
    "Short Term Treasury":        ("GS1", 1.0),
}

EQUITY_CLASSES = ["US Stock Market", "US Large Cap Value", "US Large Cap Growth",
                  "US Small Cap", "US Small Cap Value", "US Small Cap Growth"]

ASSET_CLASSES = EQUITY_CLASSES + list(_TREASURY_SPECS) + ["Cash (T-Bills)", "Gold"]


def _french_market(start: str) -> pd.Series:
    from .factors import get_ff_factors
    ff = get_ff_factors("ff3", "monthly", start=start)
    return (ff["Mkt-RF"] + ff["RF"]).rename("US Stock Market")


def _french_size_value(start: str) -> pd.DataFrame:
    """Monthly value-weighted returns of the 2x3 size/book-to-market cells."""
    from pandas_datareader.data import DataReader
    key = cache.cache_key("ff6", start)
    hit = cache.load(key)
    if hit is None:
        raw = DataReader("6_Portfolios_2x3", "famafrench", start=start)[0] / 100.0
        if isinstance(raw.index, pd.PeriodIndex):
            raw.index = raw.index.to_timestamp(how="end").normalize()
        raw.columns = [c.strip() for c in raw.columns]
        cache.save(key, raw)
        hit = raw
    out = pd.DataFrame({
        "US Large Cap Growth": hit["BIG LoBM"],
        "US Large Cap Value": hit["BIG HiBM"],
        "US Small Cap Growth": hit["SMALL LoBM"],
        "US Small Cap Value": hit["SMALL HiBM"],
        "US Small Cap": hit[["SMALL LoBM", "ME1 BM2", "SMALL HiBM"]].mean(axis=1),
    })
    return out


def synthetic_treasury_returns(yields_pct: pd.Series, duration: float) -> pd.Series:
    """Monthly total return from a monthly constant-maturity yield series (%)."""
    y = yields_pct.dropna() / 100.0
    carry = y.shift(1) / 12.0
    price_move = duration * (y.shift(1) - y)
    return (carry + price_move).dropna()


def _treasury(name: str, start: str) -> pd.Series:
    from .macro import get_fred
    series, duration = _TREASURY_SPECS[name]
    try:
        yields = get_fred(series, start)
    except Exception:
        if series == "GS20":     # GS20 has a 1987-1993 gap; fall back to GS10
            yields = get_fred("GS10", start)
        else:
            raise
    yields = yields.resample("ME").last()
    return synthetic_treasury_returns(yields, duration).rename(name)


def _cash(start: str) -> pd.Series:
    from .macro import get_fred
    tb = get_fred("TB3MS", start).resample("ME").last() / 100.0
    return (tb / 12.0).rename("Cash (T-Bills)")


def _gold(start: str) -> pd.Series:
    from .macro import get_fred
    try:
        fix = get_fred("GOLDAMGBD228NLBM", start).resample("ME").last()
        return fix.pct_change().dropna().rename("Gold")
    except Exception:
        from .prices import get_prices
        px = get_prices(["GLD"], start)
        monthly = px["GLD"].resample("ME").last()
        return monthly.pct_change().dropna().rename("Gold")


def get_asset_class_returns(classes: list[str] | None = None,
                            start: str = "1972-01-01") -> pd.DataFrame:
    """Monthly simple returns for the requested long-history asset classes.

    classes: names from ASSET_CLASSES (all when None). Earliest data varies:
    equities 1926, treasuries ~1953 (GS20/GS1) , cash 1934, gold 1968.
    """
    classes = classes or ASSET_CLASSES
    unknown = set(classes) - set(ASSET_CLASSES)
    if unknown:
        raise ValueError(f"unknown asset classes {sorted(unknown)}; "
                         f"choose from {ASSET_CLASSES}")
    cols = {}
    if "US Stock Market" in classes:
        cols["US Stock Market"] = _french_market(start)
    sv_wanted = [c for c in classes if c in EQUITY_CLASSES and c != "US Stock Market"]
    if sv_wanted:
        sv = _french_size_value(start)
        for c in sv_wanted:
            cols[c] = sv[c]
    for name in _TREASURY_SPECS:
        if name in classes:
            cols[name] = _treasury(name, start)
    if "Cash (T-Bills)" in classes:
        cols["Cash (T-Bills)"] = _cash(start)
    if "Gold" in classes:
        cols["Gold"] = _gold(start)
    out = pd.DataFrame(cols)
    out = out.loc[out.index >= pd.Timestamp(start)]
    return out[[c for c in classes]]
