"""Free macro data: CPI and T-bill rates from FRED (no API key via
pandas_datareader), Shiller CAPE from the Yale dataset."""

from __future__ import annotations

import pandas as pd

from . import cache

SHILLER_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"


def get_fred(series: str, start: str = "1950-01-01",
             refresh: bool = False) -> pd.Series:
    from pandas_datareader.data import DataReader
    key = cache.cache_key("fred", series, start)
    if not refresh:
        hit = cache.load(key)
        if hit is not None:
            return hit.iloc[:, 0]
    df = DataReader(series, "fred", start=start)
    cache.save(key, df)
    return df.iloc[:, 0]


def get_cpi(start: str = "1950-01-01") -> pd.Series:
    """CPI-U index (CPIAUCSL) — used for real returns and inflation-adjusted
    cashflows. Falls back to the
    Shiller GitHub mirror when FRED is unreachable."""
    try:
        return get_fred("CPIAUCSL", start)
    except Exception:
        from .github_mirrors import cpi_series
        cpi = cpi_series()
        return cpi.loc[cpi.index >= pd.Timestamp(start)]


def get_tbill_rate(start: str = "1954-01-01") -> pd.Series:
    """3-month T-bill secondary-market rate (TB3MS), annual percent -> decimal."""
    return get_fred("TB3MS", start) / 100.0


def _cape_from_multpl() -> pd.Series:
    """Current Shiller CAPE from multpl.com (monthly, 1871+)."""
    import io
    import urllib.request

    req = urllib.request.Request(
        "https://www.multpl.com/shiller-pe/table/by-month",
        headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=30).read().decode()
    t = pd.read_html(io.StringIO(html))[0]
    t.columns = ["date", "cape"]
    dates = pd.to_datetime(t["date"].str.replace(r"^(\w{3}) \d{1,2}, ",
                                                r"\1 1, ", regex=True))
    s = pd.Series(pd.to_numeric(t["cape"], errors="coerce").values,
                  index=dates).dropna()
    s = s[s > 0].sort_index()
    s.index = s.index.to_period("M").to_timestamp()
    return s[~s.index.duplicated(keep="last")].rename("CAPE")


def get_shiller_cape(refresh: bool = False) -> pd.Series:
    """Monthly Shiller CAPE (cyclically adjusted P/E).

    Primary: multpl.com (current). Fallbacks: Yale xls, then the bundled
    GitHub mirror. Requires xlrd for the xls path.
    """
    key = cache.cache_key("shiller", "cape")
    if not refresh:
        hit = cache.load(key)
        if hit is not None:
            return hit.iloc[:, 0]
    try:
        s = _cape_from_multpl()
    except Exception:
        try:
            raw = pd.read_excel(SHILLER_URL, sheet_name="Data", skiprows=7)
        except Exception:
            from .github_mirrors import shiller_cape
            s = shiller_cape()
            cache.save(key, s.to_frame())
            return s
        raw = raw.rename(columns={raw.columns[0]: "Date", "CAPE": "CAPE"})
        dates = pd.to_datetime(
            raw["Date"].astype(str).str.replace(r"\.1$", ".10", regex=True),
            format="%Y.%m", errors="coerce")
        cape = pd.to_numeric(raw.get("CAPE", raw.iloc[:, 12]), errors="coerce")
        s = pd.Series(cape.values, index=dates).dropna()
        # Yale pre-fills future months with CAPE = 0; CAPE is always positive.
        s = s[s > 0]
        s.name = "CAPE"
    cache.save(key, s.to_frame())
    return s
