# portlab

[![PyPI version](https://img.shields.io/pypi/v/portlab.svg)](https://pypi.org/project/portlab/)
[![Python versions](https://img.shields.io/pypi/pyversions/portlab.svg)](https://pypi.org/project/portlab/)
[![License: MIT](https://img.shields.io/pypi/l/portlab.svg)](LICENSE)

A portfolio risk and optimization library for Python, built around one
principle: **a risk number you cannot backtest is an opinion.** Every
estimator here is cross-checked against an independent implementation
and ships with the tests that pin its conventions down — see
[VALIDATION.md](VALIDATION.md) for the verification record, including
the defects that process found and fixed.

## Install

```bash
pip install portlab
```

Requires Python 3.10+. Core dependencies: numpy, pandas, scipy,
scikit-learn, statsmodels, plotly. The CVaR/CDaR LP solvers use
scipy; optional solver backends (cvxpy, OSQP, Gurobi) are imported
lazily when requested. Market-data loaders are also lazy: install
`portlab[data]` for the yfinance / pandas-datareader / pyarrow
stack used by `portlab.data`.

From source instead:

```bash
pip install git+https://github.com/Realosunboy6/portlab.git
```

## Quick start

```python
import pandas as pd
from portlab.risk.var import historical_var
from portlab.risk.backtest import exceptions, kupiec_pof, christoffersen_ind

rets = pd.Series(...)                       # daily simple returns
var = historical_var(rets, alpha=0.05, window=250)
exc = exceptions(rets, var)                 # nullable boolean Series
print(kupiec_pof(exc, 0.05))                # coverage test
print(christoffersen_ind(exc))              # gap-aware independence test
```

### Reproduce the published validation anchor

```python
import pandas as pd
from portlab.risk.var import historical_var
from portlab.risk.backtest import exceptions, kupiec_pof, christoffersen_ind

px = pd.read_csv("https://stooq.com/q/d/l/?s=spy.us&i=d",
                 parse_dates=["Date"], index_col="Date")["Close"].sort_index()
rets = px.loc["2019-01-02":].pct_change().dropna()

exc = exceptions(rets, historical_var(rets, alpha=0.05, window=250))
print(kupiec_pof(exc, 0.05))
print(christoffersen_ind(exc))
```

Recorded on the validation dataset (through 2026-10-09): n=1,703,
86 exceptions, Kupiec LR = 0.0089, Christoffersen LR_ind = 10.589773.
Live data adds recent days, so counts grow slightly — the LR values
should still land on the anchors to ~3 decimals.

SPY's VaR passes the coverage test but fails independence — the
exceptions bunch in Feb–Apr 2020. That distinction is the point of the
library: cite conditional coverage, not Kupiec alone.

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
| Monte Carlo & goals | `portlab.montecarlo` | Seeded simulation (bootstrap, block bootstrap, normal, Student-t, multi-asset statistical), withdrawal rules (fixed, percentage, guardrails, VPW, RMD), glide paths, goal probabilities |
| Monitoring | `portlab.risk.monitor`, `portlab.risk.rebalance` | Fills + prices → daily risk report (the RiskLens watcher); drift tables and rebalance signals |
| Factor models | `portlab.factor`, `portlab.optimize.riskfactor`, `portlab.data.factors` | Fama–French / Carhart regressions (Newey–West), rolling exposures, factor risk decomposition |
| Tactical & indicators | `portlab.tactical`, `portlab.indicators` | MA timing, relative strength, dual momentum, target volatility; RSI, MACD, Bollinger, ATR, stochastic |
| Derivatives | `portlab.derivatives` | Black–Scholes prices/Greeks, implied volatility, CRR binomial trees (American early exercise) |
| ALM | `portlab.alm` | Liability schedules, present value, funded status, surplus optimization, hedge portfolios |
| Tax | `portlab.taxlots`, `portlab.taxopt` | Lot ledger (FIFO/LIFO/HIFO/min-tax relief, wash-sale window), tax-aware sale plans and tax-aware optimization (min-tax is a greedy relief heuristic, documented as such) |
| Multi-period optimization | `portlab.mpo` | Weight trajectories with L1 turnover (transaction-cost) penalties |
| Stress & scenarios | `portlab.stress`, `portlab.scenarios` | Historical-episode shocks, filtered historical simulation (GARCH-residual scenarios), scenario VaR/ES |
| Corporate actions & costs | `portlab.corpactions`, `portlab.costs` | Splits/dividends applied to positions with basis invariance; fixed/proportional/spread/impact cost models |
| Data | `portlab.data` | Price/factor/macro loaders with local caching (Yahoo, Ken French, FRED); network deps are lazy — install `portlab[data]` |
| Reports & charts | `portlab.report`, `portlab.plots` | HTML tear sheets and Plotly charts |
| Analytics | `portlab.analytics` | Correlations, cointegration, PCA, performance tables, screeners |
| Fill validation | `portlab.fillvalidate` | Pre-trade fill checks against prices and broker minimums |

## Conventions (read this before trusting a number)

- Inputs are **simple returns** (positive = gain). VaR/ES are reported
  as **positive numbers = losses**. An exception is `returns < -VaR`.
- **`alpha` means two things, by module family** — in `portlab.risk.*`
  it is the *tail probability* (`0.05` = 95% VaR); in `portlab.metrics.*`
  and the CVaR optimizer it is the *confidence level* (`0.95`). Every
  docstring states which.
- Rolling estimators use **forecast alignment**: the value stamped at
  day *t* is built only from data strictly before *t*. Backtests score
  exactly the stamped days — no look-ahead, no warmup padding.
- `exceptions()` returns a nullable-boolean Series. Days that cannot
  be scored (VaR warmup, missing returns, gaps) are `<NA>`; Kupiec
  counts only scored days and Christoffersen counts only transitions
  between genuinely adjacent scored days.

## Scope

As of 0.2.0 this is the complete toolkit — the earlier public core
plus the modules that were initially held back (monitoring, tax,
Monte Carlo, factors, tactical, ALM, derivatives, data, reporting),
each verified before release; see VALIDATION.md for the 0.2.0
verification record. What is not in this repo, by design: broker
credentials, live account data, and the Sentinel web application
that consumes this library. The algorithms are standard textbook
methods (Kupiec 1995; Christoffersen 1998; Rockafellar & Uryasev
2000; Black & Litterman 1992) — the contribution is the
implementation discipline and the validation suite.

## Citation

If you use portlab in research, please cite it via
[CITATION.cff](CITATION.cff) (GitHub's "Cite this repository" button
reads the same file).

## License

MIT — see [LICENSE](LICENSE).

## Author

Ibrahim Oyeyinka — PhD candidate, Industrial & Systems Engineering,
Northern Illinois University.
