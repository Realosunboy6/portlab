"""RiskLens — market-risk monitoring for portlab.

Submodules:

- ``var``: historical / parametric VaR and Expected Shortfall on return series.
- ``backtest``: VaR backtesting — Kupiec POF, Christoffersen independence,
  joint test, Basel traffic light.
- ``tail``: tail diagnostics — Jarque–Bera, Hill tail index, GPD extreme
  VaR/TVaR.
- ``rebalance``: drift tables, rebalance signals (threshold / calendar), and
  the trades that restore target weights.
- ``monitor``: the daily watcher — fills + prices → risk report.

Convention shared by every function: inputs are *simple returns* (positive =
gain); VaR/ES are reported as *positive numbers = losses*; ``alpha`` is the
tail probability (0.05 = 95% VaR). Note this differs from
``portlab.metrics.var_historical``, where alpha is the confidence level —
the delegation in ``risk.var`` converts.
"""

from . import backtest, monitor, rebalance, regime, tail, var
from .backtest import (basel_traffic_light, christoffersen_ind, exceptions,
                       joint_test, kupiec_pof)
from .monitor import (daily_risk_report, portfolio_returns_from_fills,
                      write_markdown_report)
from .rebalance import (drift_table, drift_trades, rebalance_signal,
                        rebalance_trades)  # noqa: F401  (deprecated alias)
from .regime import detect_regime, ewma_vol, regime_var
from .tail import (gpd_var_tvar, hill_estimator, hill_plot_data, jarque_bera,
                   tail_summary)
from .var import expected_shortfall, historical_var, parametric_var

__all__ = [
    "var", "backtest", "tail", "rebalance", "monitor", "regime",
    "historical_var", "parametric_var", "expected_shortfall",
    "detect_regime", "ewma_vol", "regime_var",
    "exceptions", "kupiec_pof", "christoffersen_ind", "joint_test",
    "basel_traffic_light",
    "jarque_bera", "hill_estimator", "hill_plot_data", "gpd_var_tvar",
    "tail_summary",
    "drift_table", "rebalance_signal", "drift_trades",
    "portfolio_returns_from_fills", "daily_risk_report",
    "write_markdown_report",
]
