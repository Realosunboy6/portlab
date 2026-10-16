"""Famous "lazy" portfolios, pre-loaded with standard ETF proxies.

Sources: Bogleheads wiki (Lazy portfolios), portfoliocharts.com (Golden
Butterfly, Pinwheel), optimizedportfolio.com, lazyportfolioetf.com.

Every allocation sums to 1. The limiting ETF inception year is noted so
backtests know the earliest common start; for longer history map the tickers
to `portlab.data.asset_classes` proxies.
"""

from __future__ import annotations

LAZY_PORTFOLIOS: dict[str, dict[str, float]] = {
    "Classic 60/40":          {"VTI": 0.60, "BND": 0.40},
    "Bogleheads Three-Fund":  {"VTI": 0.40, "VXUS": 0.20, "BND": 0.40},
    "All Weather (Dalio)":    {"VTI": 0.30, "TLT": 0.40, "IEF": 0.15,
                               "GLD": 0.075, "DBC": 0.075},
    "Golden Butterfly":       {"VTI": 0.20, "VBR": 0.20, "TLT": 0.20,
                               "SHY": 0.20, "GLD": 0.20},
    "Permanent (Browne)":     {"VTI": 0.25, "TLT": 0.25, "BIL": 0.25, "GLD": 0.25},
    "Ivy 5 (Faber)":          {"VTI": 0.20, "VEU": 0.20, "IEF": 0.20,
                               "DBC": 0.20, "VNQ": 0.20},
    "Swensen Yale":           {"VTI": 0.30, "VEA": 0.15, "EEM": 0.05, "VNQ": 0.20,
                               "IEF": 0.15, "TIP": 0.15},
    "Coffeehouse":            {"VOO": 0.10, "VTV": 0.10, "VB": 0.10, "VBR": 0.10,
                               "VEA": 0.10, "VNQ": 0.10, "BND": 0.40},
    "No-Brainer (Bernstein)": {"VOO": 0.25, "VB": 0.25, "VEA": 0.25, "BSV": 0.25},
    "Pinwheel":               {"VTI": 0.15, "VBR": 0.10, "VEA": 0.15, "EEM": 0.10,
                               "IEF": 0.15, "BIL": 0.10, "VNQ": 0.15, "GLD": 0.10},
    "Rick Ferri Core Four":   {"VTI": 0.48, "VXUS": 0.24, "VNQ": 0.08, "BND": 0.20},
    "Merriman Ultimate (equity)": {"VOO": 0.10, "VTV": 0.10, "VB": 0.10, "VBR": 0.10,
                                   "VNQ": 0.10, "EFA": 0.10, "EFV": 0.10,
                                   "SCZ": 0.10, "DLS": 0.10, "EEM": 0.10},
    "Larry Portfolio":        {"VBR": 0.15, "DLS": 0.075, "DGS": 0.075, "IEF": 0.70},
    "Couch Potato (Burns)":   {"VTI": 0.50, "TIP": 0.50},
    "Merriman 4-Fund Combo":  {"VOO": 0.25, "RPV": 0.25, "IJR": 0.25, "IJS": 0.25},
    "Talmud":                 {"VTI": 1 / 3, "VNQ": 1 / 3, "BND": 1 / 3},
    "Desert":                 {"VTI": 0.30, "IEF": 0.60, "GLD": 0.10},
}

# Earliest common backtest start (limiting ETF inception year).
INCEPTION: dict[str, int] = {
    "Classic 60/40": 2007,            # BND
    "Bogleheads Three-Fund": 2011,    # VXUS
    "All Weather (Dalio)": 2006,      # DBC
    "Golden Butterfly": 2004,         # GLD
    "Permanent (Browne)": 2007,       # BIL
    "Ivy 5 (Faber)": 2007,            # VEU
    "Swensen Yale": 2007,             # VEA
    "Coffeehouse": 2010,              # VOO
    "No-Brainer (Bernstein)": 2010,   # VOO
    "Pinwheel": 2007,
    "Rick Ferri Core Four": 2011,     # VXUS
    "Merriman Ultimate (equity)": 2007,  # DLS/SCZ
    "Larry Portfolio": 2007,          # DGS
    "Couch Potato (Burns)": 2003,     # TIP
    "Merriman 4-Fund Combo": 2010,    # VOO
    "Talmud": 2007,                   # BND
    "Desert": 2004,                   # GLD
}


def get_lazy_portfolio(name: str) -> dict[str, float]:
    """Case/space-insensitive lookup of a lazy portfolio allocation."""
    key = name.strip().lower()
    for nm, alloc in LAZY_PORTFOLIOS.items():
        if nm.lower() == key or nm.lower().startswith(key):
            return dict(alloc)
    raise KeyError(f"unknown lazy portfolio {name!r}; "
                   f"choose from {list(LAZY_PORTFOLIOS)}")


def lazy_tickers() -> list[str]:
    """Every ticker used across the library (for a single batched download)."""
    return sorted({t for alloc in LAZY_PORTFOLIOS.values() for t in alloc})
