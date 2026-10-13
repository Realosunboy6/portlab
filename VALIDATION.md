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

## Extraction notes (what is not in this repo, and why)

This package is a curated snapshot of a larger private toolkit,
published with fresh history under MIT. Held back, with their tests:
live monitoring (`risk.monitor`, `risk.rebalance` — the daily watcher
over a real portfolio), tax-lot optimization, multi-period
optimization (`mpo`), factor/indicator/forecast tooling, tear-sheet
reporting (`report`), and market data access. Two tests are skipped
in this repo because they exercise held-back modules (`mpo`,
`report`); the skips and their reasons are declared in
`tests/conftest.py`, and the
VaR/backtest tests were extracted verbatim from the private suite's
risk test file (the monitor/rebalance tests stayed private with
their modules).
