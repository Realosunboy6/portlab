# portlab

[![PyPI version](https://img.shields.io/pypi/v/portlab.svg)](https://pypi.org/project/portlab/)
[![Python versions](https://img.shields.io/pypi/pyversions/portlab.svg)](https://pypi.org/project/portlab/)
[![License: MIT](https://img.shields.io/pypi/l/portlab.svg)](LICENSE)

A portfolio analysis toolkit for Python, built on one principle: **a
risk number you cannot backtest is an opinion.** Every estimator is
cross-checked against an independent implementation, and the tests
that pin its conventions down ship with the package — see
[VALIDATION.md](VALIDATION.md) for the record, including the defects
that process found and fixed.

## Install

```bash
pip install portlab            # core
pip install "portlab[data]"    # + market-data loaders (yfinance etc.)
pip install "portlab[opt]"     # + cvxpy solver backend
```

Requires Python 3.10+. From source:
`pip install git+https://github.com/Realosunboy6/portlab.git`

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

Validation anchor (SPY, through 2026-10-09): n=1,703, 86 exceptions,
Kupiec LR = 0.0089, Christoffersen LR_ind = 10.589773. SPY passes
coverage but fails independence — the exceptions bunch in Feb–Apr
2020. Cite conditional coverage, not Kupiec alone. Reproduction
code is in VALIDATION.md.

## What's inside

- **Risk** — `risk.var`, `risk.backtest`, `risk.tail`, `risk.regime`:
  VaR/ES (historical, parametric, EWMA, GARCH), Kupiec/Christoffersen
  backtests, Basel traffic light, tail diagnostics, regimes
- **Optimization** — `optimize`, `mpo`: max-Sharpe/GMV frontiers,
  min-CVaR/CDaR, Black–Litterman, HRP, risk parity, robust weights,
  multi-period optimization with transaction costs
- **Simulation** — `montecarlo`, `scenarios`, `stress`: seeded Monte
  Carlo with withdrawal rules, filtered historical simulation,
  historical-episode stress
- **Factors & signals** — `factor`, `analytics`, `forecast`,
  `indicators`, `tactical`: factor regressions, PCA, cointegration,
  ARIMA/ETS forecasts, technical indicators, tactical allocation
- **Portfolio operations** — `taxlots`, `taxopt`, `corpactions`,
  `costs`, `attribution`, `rebalance`: tax-lot ledger and tax-aware
  optimization, corporate actions, cost models, Brinson attribution
- **Monitoring & output** — `risk.monitor`, `data`, `derivatives`,
  `alm`, `report`, `plots`: fills-to-risk-report daily watcher
  (RiskLens), data loaders, options pricing, liability-driven
  investing, tear sheets and charts

## Conventions

- Inputs are **simple returns**; VaR/ES are **positive = losses**.
  An exception is `returns < -VaR`.
- `alpha` is the *tail probability* in `portlab.risk.*` (0.05 = 95%
  VaR) and the *confidence level* in `portlab.metrics` and the CVaR
  optimizer (0.95). Each docstring states which.
- Rolling estimators are forecast-aligned: the value at day *t*
  uses only data strictly before *t*. Unscored days in
  `exceptions()` are `<NA>`, never silently passed.

## Citation

Cite via [CITATION.cff](CITATION.cff) (GitHub's "Cite this
repository" button reads the same file).

## License

MIT — see [LICENSE](LICENSE).

## Author

Ibrahim Oyeyinka — PhD candidate, Industrial & Systems Engineering,
Northern Illinois University.
