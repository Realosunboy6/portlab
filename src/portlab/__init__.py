"""portlab — portfolio risk and optimization library.

Public core: performance/risk metrics, VaR/ES estimation with VaR
backtesting (Kupiec, Christoffersen), tail diagnostics, regime
detection, portfolio optimization (mean-variance, CVaR, Black-
Litterman, HRP, risk parity, robust), portfolio backtesting, and
Brinson attribution. See VALIDATION.md for the verification record.

Conventions: inputs are *simple returns* (positive = gain); VaR/ES
are reported as *positive numbers = losses*. In ``portlab.risk``,
``alpha`` is the tail probability (0.05 = 95% VaR); in
``portlab.metrics`` and the CVaR optimizer, ``alpha`` is the
confidence level (0.95). Each module's docstring states which.
"""

__version__ = "0.1.1"

from . import (attribution, backtest, covariance, metrics, optimize,
               rebalance, returns, risk)

__all__ = [
    "attribution", "backtest", "covariance", "metrics", "optimize",
    "rebalance", "returns", "risk", "__version__",
]
