"""Global defaults shared across portlab modules.

Every function that uses one of these values also accepts it as an explicit
parameter, so notebooks can override without touching module state.
"""

TRADING_DAYS = 252
MONTHS_PER_YEAR = 12

# Annual risk-free rate used when no T-bill series is supplied.
DEFAULT_RF = 0.03

# Confidence level for VaR / CVaR calculations.
DEFAULT_ALPHA = 0.95


def periods_per_year(freq: str) -> int:
    """Map a pandas-style frequency label to annualization periods."""
    freq = freq.upper()
    table = {"D": TRADING_DAYS, "B": TRADING_DAYS, "W": 52, "M": 12, "ME": 12,
             "Q": 4, "QE": 4, "A": 1, "Y": 1, "YE": 1}
    if freq not in table:
        raise ValueError(f"Unknown frequency {freq!r}; expected one of {sorted(table)}")
    return table[freq]
