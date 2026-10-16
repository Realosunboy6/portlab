"""Sector grouping for ticker portfolios via Yahoo Finance metadata.

One HTTP call per ticker, cached on disk as JSON, so repeated use is free.
"""

from __future__ import annotations

import json

import pandas as pd

from .cache import cache_root


def _meta_path():
    return cache_root() / "ticker_metadata.json"


def ticker_info(ticker: str) -> dict:
    """Cached Yahoo Finance metadata: sector, industry, name."""
    t = ticker.strip().upper()
    p = _meta_path()
    meta = json.loads(p.read_text()) if p.exists() else {}
    if t in meta:
        return meta[t]
    import yfinance as yf

    info = yf.Ticker(t).info or {}
    rec = {"sector": info.get("sector") or "Unknown",
           "industry": info.get("industry") or "Unknown",
           "name": info.get("longName") or info.get("shortName") or t}
    meta[t] = rec
    try:
        p.write_text(json.dumps(meta))
    except Exception:
        pass
    return rec


def sector_of(ticker: str) -> str:
    """Sector for a ticker: static universe map first, Yahoo metadata fallback."""
    t = ticker.strip().upper()
    try:
        from .universe import SECTOR_MAP
        if t in SECTOR_MAP:
            return SECTOR_MAP[t]
    except Exception:
        pass
    return ticker_info(t)["sector"]


def sector_weights(weights: dict[str, float] | pd.Series) -> pd.Series:
    """Aggregate portfolio weights by sector, descending.

    Tickers whose metadata can't be fetched land in "Unknown" rather than
    failing the whole call.
    """
    w = pd.Series(weights, dtype=float)

    def safe_sector(t: str) -> str:
        try:
            return sector_of(t)
        except Exception:
            return "Unknown"

    by_sector = w.groupby(w.index.map(safe_sector)).sum()
    return by_sector.sort_values(ascending=False).rename("weight")
