# portlab

A portfolio risk and optimization library for Python, built around one
principle: **a risk number you cannot backtest is an opinion.** Every
estimator here is cross-checked against an independent implementation
and ships with the tests that pin its conventions down — see
[VALIDATION.md](VALIDATION.md) for the verification record, including
the defects that process found and fixed.

## What's inside

| Area | Modules | Highlights |
|---|---|---|
| Metrics | `portlab.metrics`, `portlab.returns` | CAGR, vol, Sharpe/Sortino/Calmar, drawdowns, rolling stats |
| VaR / ES | `portlab.risk.var` | Historical, parametric (normal / Student-t / Cornish–Fisher), EWMA, GARCH(1,1)-filtered VaR and Expected Shortfall, with explicit mean conventions (`mean="zero" \| "constant"`) |
| VaR backtesting | `portlab.risk.backtest` | Exception series with a nullable-boolean contract (unscored days are `<NA>`, never silently passed), Kupiec POF, Christoffersen independence (gap-aware: transitions never join across data holes), joint conditional-coverage test, Basel traffic light |
| Tail diagnostics | `portlab.risk.tail` | Jarque–Bera, Hill tail index, GPD extreme VaR/TVaR |
| Regime | `portlab.risk.regime` | HMM-style regime detection, EWMA vol, regime-conditional VaR |
| Optimization | `portlab.optimize` | Max-Sharpe / GMV / target-return frontiers, min-CVaR and min-CDaR (LP), Black–Litterman, HRP, risk parity, resampled efficiency, robust (box-uncertainty) weights, constraint attribution |
| Covariance | `portlab.covariance` | Sample, Ledoit–Wolf shrinkage, semicovariance |
| Backtesting | `portlab.backtest` | Portfolio backtests with rebalancing, rolling backtests, historical-episode stress |
| Attribution | `portlab.attribution` | Brinson allocation/selection/interaction |

## Conventions (read this before trusting a number)

- Inputs are **simple returns** (positive = gain). VaR/ES are reported
  as **positive numbers = losses**. An exception is `returns < -VaR`.
- **`alpha` means two things, by module family** — in `portlab.risk.*`
  it is the *tail probability* (`0.05` = 95% VaR); in
  `portlab.metrics.*` and the CVaR optimizer it is the *confidence
  level* (`0.95`). Every docstring states which.
- Rolling estimators use **forecast alignment**: the value stamped at
  day *t* is built only from data strictly before *t*. Backtests score
  exactly the stamped days — no look-ahead, no warmup padding.
- `exceptions()` returns a nullable-boolean Series. Days that cannot
  be scored (VaR warmup, missing returns, gaps) are `<NA>`; Kupiec
  counts only scored days and Christoffersen counts only transitions
  between genuinely adjacent scored days.

## Install

```bash
pip install git+https://github.com/Realosunboy6/portlab.git
```

Core dependencies: numpy, pandas, scipy, scikit-learn. The CVaR/CDaR
LP solvers use scipy; optional solver backends (cvxpy, OSQP, Gurobi)
are imported lazily when requested.

## Quick start

```python
import pandas as pd
from portlab.risk import var, backtest

rets = pd.Series(...)                      # daily simple returns
v = var.historical_var(rets, alpha=0.05, window=250)
exc = backtest.exceptions(rets, v)         # nullable boolean Series
print(backtest.kupiec_pof(exc.dropna().astype(bool), alpha=0.05))
print(backtest.christoffersen_ind(exc))    # gap-aware; do not dropna
```

## Scope

This is the public core of a larger private toolkit. Not included:
live portfolio monitoring, broker integrations, tax-lot optimization,
and forecasting/indicator tooling. The algorithms here are standard
textbook methods (Kupiec 1995; Christoffersen 1998; Rockafellar &
Uryasev 2000; Black & Litterman 1992) — the contribution is the
implementation discipline and the validation suite.

## License

MIT — see [LICENSE](LICENSE).

## Author

Ibrahim Oyeyinka — PhD candidate, Industrial & Systems Engineering,
Northern Illinois University.
