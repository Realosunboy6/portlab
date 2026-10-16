# Validation record

How the numbers in this library were checked, what the checking found,
and what remains known-but-unfixed. All results below are from an
audit run in October 2026 on real market data (12 tickers, 1,954
trading days, 2019-01-02 → 2026-10-09), followed by five rounds of
independent adversarial verification of the VaR backtest stack.

## Method

- **Reference implementations:** PyPortfolioOpt and Riskfolio-Lib
  (optimization, CVaR), `arch` (GARCH), statsmodels (ARIMA, factor
  regressions), scikit-learn (Ledoit–Wolf), hmmlearn (regimes).
- **Independent reimplementation:** the backtest statistics were
  re-derived from the primary sources (Kupiec 1995; Christoffersen
  1998) and compared numerically, including an exhaustive sweep and
  randomized differential testing (below).
- **Tests in this repo are synthetic by design** (seeded, no network).
  The real-data anchors in this document are the regression targets
  the synthetic suite's conventions were validated against.

## Defects found by verification — and fixed

1. **Kupiec warmup padding.** Exception counting once treated
   NaN-warmup days (before the first VaR exists) as non-exceptions,
   inflating *n* and biasing the test toward rejection: SPY LR 2.104
   instead of 0.018. Measured exactly, the false-rejection rate at a
   nominal 5% level was **26.9% padded vs 5.2% fixed**.
2. **One-observation look-ahead in rolling VaR.** The rolling helper
   stamped each window's VaR on the window's last day, putting day
   *t*'s own return inside its estimation window. Rolling estimators
   now stamp forecasts: the value at *t* uses only data before *t*.
3. **Wrong restricted MLE in Christoffersen independence.** The
   restricted π was computed as the exception rate over days instead
   of over transition pairs, (n01 + n11) / total_pairs. On SPY this
   moved LR_ind from 10.589804 to the correct **10.589773** — small
   there, wrong in general.
4. **Silent passes on empty input.** Kupiec/Christoffersen once
   returned LR = 0, p = 1.0 ("perfect pass") for a backtest with zero
   scored days. They now raise (Kupiec) or return NaN with an
   explanatory note (independence/joint, when exceptions exist but
   their transitions are unidentified).
5. **Transitions joined across data holes.** Christoffersen counting
   treated scored days separated by unscored gaps as adjacent. The
   exception series is now nullable-boolean end-to-end; transitions
   are counted only between genuinely adjacent scored pairs.

## Differential-testing evidence

- **Exhaustive sweep:** all 29,523 sequences over {0, 1, NA} of length
  ≤ 9 were run through both this implementation and an independent
  one: **0 mismatches** in transition counts or in the
  defined/undefined classification; maximum LR difference 1.6e-11.
  The undefined-independence condition removes exactly the 3,008
  zero-information sequences (all had LR exactly 0 under the old
  formula).
- **Randomized check:** the pair-based restricted MLE matches
  independent numerical maximization of the restricted likelihood to
  9.1e-10 over 2,000 gapped random series.
- **Edge values:** Kupiec LR at n = 100: x = 0 → 10.2587;
  x = 100 → 599.1465 (reference values, matched to 4 dp).

## Real-data anchors (SPY, rolling-250 historical VaR, α = 0.05)

| Quantity | Value |
|---|---|
| Scored days | 1,703 |
| Exceptions | 86 |
| Kupiec LR (p) | 0.0089 (passes) |
| Christoffersen counts (n00, n01, n10, n11) | 1542, 74, 74, 12 |
| Independence LR (p) | 10.589773 (0.00114 — **fails**) |
| Joint LR, df 2 | 10.5987 |

The failure is reported, not tuned away: SPY's exceptions cluster in
Feb–Apr 2020, so the model passes coverage and fails independence.
"Passes the backtest" claims for this configuration should cite
conditional coverage, not Kupiec alone.

## Parity results (audit)

- Max-Sharpe weights vs PyPortfolioOpt: agreement to 5.7e-5.
- GARCH(1,1) one-step σ forecasts vs `arch` (zero-mean, variance-
  targeted): within the test tolerance across the audit set; a
  residual VaR gap vs `arch`'s constant-mean default decomposed as
  ≈79% the mean term in the reference VaR, ≈15% a one-day dating
  offset in the comparison harness, ≈6% fit-mean specification,
  ≈0% variance targeting/initialization. Same-window estimators
  differ by 0.0002 pp — no estimation defect. This is why
  `garch_var` and `parametric_var` now expose an explicit
  `mean="zero" | "constant"` argument instead of disagreeing silently.

## Known limitations

- **GARCH with normal innovations under-covers.** On SPY (rolling
  250-day windows, 1,703 forecasts), exception rates were ≈6.0%
  (zero-mean) and ≈6.4% (mean-relative) against a 5% target. Prefer
  `dist="t"`.
- **250-day historical VaR is slow to adapt** — see the independence
  failure above; it is a property of the estimator, surfaced by the
  backtest rather than hidden by it.
- Public tests use seeded synthetic data. Figures quoted here come
  from the October 2026 real-data audit; the harness is reproducible
  from the anchors and conventions documented above.

## The 0.2.0 full release (2026-10-16)

0.1.x was a curated core. 0.2.0 publishes the complete toolkit: the
modules initially held back (`risk.monitor`, `risk.rebalance`,
`taxlots`, `taxopt`, `mpo`, `factor`, `forecast`, `indicators`,
`tactical`, `derivatives`, `alm`, `montecarlo`, `scenarios`,
`stress`, `corpactions`, `costs`, `analytics`, `fillvalidate`,
`data`, `plots`, `report`, `optimize.riskfactor`) are now public,
each verified before release by the same standard as the 0.1 core:
an independent-implementation check, separate from the module's own
tests. The private suite stood at 383 passed (1 live-network test
deselected) on release day; the public suite below reproduces that
count. Results of the independent pass:

- **Derivatives** — Black–Scholes prices match an independent scipy
  implementation to <1e-10 on every case; put–call parity exact;
  gamma matches the closed form; implied vol round-trips the input
  vol to <1e-6; the CRR binomial price converges to Black–Scholes.
  Convention pinned: `bs_greeks` reports vega per 1% volatility
  point.
- **Indicators** — RSI and MACD are numerically identical to
  reference Wilder/EMA implementations; Bollinger bands use the
  sample standard deviation (ddof=1).
- **ALM** — liability present value matches direct discounting to
  1e-8; funded-status arithmetic exact.
- **Factor** — Fama–French 3-factor loadings, alpha, t-stats and
  R² are identical to a direct statsmodels OLS fit. **Forecast** —
  ARIMA forecasts are identical to the statsmodels fit they wrap.
- **Monte Carlo** — the 0.1.0 hold ("release when seeded-tested")
  is closed: the same seed reproduces paths bit-for-bit; with the
  normal model the simulated mean terminal value matches the exact
  identity E[V_T] = V_0 (1 + μ/12)^(12T) within Monte Carlo error;
  bootstrapping a constant return series reproduces the exact
  compounded value to the cent.
- **Tax** — `tax_aware_sale_plan` matches hand computation on a
  constructed ledger, including loss-lot-first ordering under
  `min_tax` and the long-term boundary (held > 365 days). `min_tax`
  remains what it always was: a greedy relief heuristic, not an
  optimizer guarantee — the docstrings say so. The `taxopt`
  cost-segment marginals match hand marginals at the default rates
  (ST 37%, LT 20%).
- **Multi-period optimization** — with T=1 and zero transaction
  cost, `multiperiod_weights` equals the KKT closed-form
  mean–variance solution; with a prohibitive cost it correctly
  does not trade.
- **Monitoring** — `portfolio_returns_from_fills` reproduces a
  hand-computed value series and time-weighted returns exactly,
  including the contribution day (external flows are excluded from
  returns, as TWR requires).
- **Scenarios / stress** — scenario VaR/ES recomputed by hand from
  the scenario cube; historical-episode shocks and scenario P&L
  attribution match manual arithmetic. **Corporate actions** — a
  2:1 split preserves total cost basis exactly; a cash dividend
  leaves basis untouched. **Costs** — `turnover` is the gross
  convention, Σ|trades| / portfolio value.

Public-tree hygiene: the test fixtures in this repo are synthetic.
(The private suite's fill-validation fixtures use a real
portfolio's fills; in this tree they are replaced with synthetic
values, and broker names in docstrings are generalized.) The
release was scanned for credentials, account identifiers, local
paths, and licensed-material markers: none present. The author's
university affiliation appears only as author information in the
README and citation file.

Public suite on release day: **383 passed, 1 skipped** — the skip
is the suite's only live-network test (`test_tbill_download`,
declared in `tests/conftest.py`). It is opt-in rather than removed:
with the data extra installed and internet access, anyone can run
it with `PORTLAB_LIVE=1 pytest tests/test_riskfree.py`. Verified
in that mode on release day: the test passes against the live FRED
service (full suite in live mode: 384 passed, 0 skipped). The
feature it tests, `portlab.data.riskfree.get_tbill`, is ordinary
library API, usable directly in user code.
