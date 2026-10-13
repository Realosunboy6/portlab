"""Portfolio backtesting engine.

Supports fixed target weights, periodic contributions/withdrawals (optionally
inflation-adjusted), periodic or tolerance-band rebalancing, benchmark
comparison, CPI-deflated (real) results, short-cash leverage with an optional
maintenance-margin rule, and after-tax results (dividend and capital-gains
taxes).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import DEFAULT_RF, TRADING_DAYS
from .. import metrics as M

REBALANCE_MONTHS = {"never": None, "monthly": 1, "quarterly": 3,
                    "semiannually": 6, "annually": 12}


@dataclass
class BacktestResult:
    balance: pd.Series                 # end-of-period equity value
    returns: pd.Series                 # per-period time-weighted returns
    weights: pd.DataFrame              # start-of-period effective weights
    cashflows: pd.Series               # external flows (+contribution/-withdrawal)
    benchmark_returns: pd.Series | None = None
    real_balance: pd.Series | None = None
    periods: int = TRADING_DAYS
    rf: float | pd.Series = DEFAULT_RF
    name: str = "Portfolio"
    rebalance_dates: list = field(default_factory=list)
    taxes_paid: pd.Series | None = None      # per-period taxes deducted
    margin_calls: list = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        s = M.summary(self.returns, rf=self.rf, periods=self.periods,
                      bench=self.benchmark_returns, name=self.name)
        s["Ending Balance"] = float(self.balance.iloc[-1])
        if self.real_balance is not None:
            s["Ending Balance (real)"] = float(self.real_balance.iloc[-1])
        if self.taxes_paid is not None and self.taxes_paid.sum() > 0:
            s["Total Taxes Paid"] = float(self.taxes_paid.sum())
        initial = self.balance.iloc[0] / (1 + self.returns.iloc[0])
        flows = pd.Series(
            [-initial, *(-self.cashflows.iloc[1:])],
            index=self.balance.index)
        flows.iloc[-1] += self.balance.iloc[-1]
        s["Money-Weighted Return (IRR)"] = M.money_weighted_return(flows)
        cols = [s.rename(self.name)]
        if self.benchmark_returns is not None:
            cols.append(M.summary(self.benchmark_returns, rf=self.rf,
                                  periods=self.periods, name="Benchmark"))
        return pd.concat(cols, axis=1)

    def drawdowns(self, top: int = 10) -> pd.DataFrame:
        return M.drawdown_table(self.returns, top)

    def tear_sheet(self, output: str | None = "tear_sheet.html",
                   title: str | None = None) -> str:
        """One-line shareable HTML report for this backtest."""
        from ..report import tear_sheet
        return tear_sheet(self.returns, benchmark=self.benchmark_returns,
                          rf=self.rf, periods=self.periods,
                          title=title or f"{self.name} — Tear Sheet",
                          output=output)


def _is_period_boundary(prev: pd.Timestamp, cur: pd.Timestamp, months: int) -> bool:
    """True when `cur` starts a new k-month block (calendar-aligned)."""
    if prev.year != cur.year or prev.month != cur.month:
        return ((cur.month - 1) % months) == 0 or months == 1
    return False


def backtest_portfolio(
    returns: pd.DataFrame,
    weights: dict[str, float] | pd.Series,
    initial: float = 10_000.0,
    cashflow: float = 0.0,
    cashflow_freq: str = "monthly",
    cashflow_inflation_adjusted: bool = False,
    rebalance: str = "annually",
    rebalance_band: float | None = None,
    benchmark: pd.Series | None = None,
    cpi: pd.Series | None = None,
    rf: float | pd.Series = DEFAULT_RF,
    periods: int = TRADING_DAYS,
    name: str = "Portfolio",
    leverage: float = 1.0,
    debt_rate: float | pd.Series = 0.0,
    maintenance_margin: float | None = None,
    tax_dividend: float = 0.0,
    tax_capgains: float = 0.0,
    div_yield: dict[str, float] | pd.Series | None = None,
    carry_losses: bool = True,
) -> BacktestResult:
    """Simulate a target-weight portfolio through historical returns.

    returns: per-period simple returns per asset (daily or monthly).
    weights: target allocation, will be normalized to sum to 1.
    cashflow: amount added (+) or withdrawn (-) at each cashflow boundary.
    cashflow_freq: 'monthly' | 'quarterly' | 'annually'.
    cashflow_inflation_adjusted: grow the cashflow with CPI (requires cpi).
    rebalance: 'never' | 'monthly' | 'quarterly' | 'semiannually' | 'annually'
        | 'bands' (use rebalance_band as absolute weight deviation trigger).
    cpi: monthly CPI index series; enables real (inflation-adjusted) balances.

    Leverage (short cash):
      leverage: gross exposure / equity, reset at every rebalance (1.0 = none).
      debt_rate: annual borrow rate; a pd.Series (e.g. T-bill history) is
        looked up per date (e.g. the historical 3-month T-bill series).
      maintenance_margin: e.g. 0.25 — when equity/gross falls below this,
        positions are sold to restore the target leverage and the date is
        recorded in `margin_calls`.

    Taxes:
      tax_dividend: rate applied to dividend income each period. Because
        prices are total-return (dividends reinvested), the dividend stream is
        approximated from `div_yield` (annual yield per asset, e.g.
        {"VTI": 0.015}); without div_yield no dividend tax is charged.
      tax_capgains: rate applied to net realized gains on rebalancing sales
        and withdrawals (average-cost basis).
      carry_losses: net realized losses offset later gains.
    """
    w_target = pd.Series(weights, dtype=float)
    w_target = w_target / w_target.sum()
    rets = returns[list(w_target.index)].dropna(how="any")
    if rets.empty:
        raise ValueError("no overlapping return history for the requested assets")
    if leverage < 1.0:
        raise ValueError("leverage must be >= 1.0 (use cash in the allocation "
                         "for sub-1x exposure)")

    cf_months = {"monthly": 1, "quarterly": 3, "annually": 12}[cashflow_freq]
    if rebalance != "bands":
        if rebalance not in REBALANCE_MONTHS:
            raise ValueError(f"rebalance must be one of {[*REBALANCE_MONTHS, 'bands']}")
        reb_months = REBALANCE_MONTHS[rebalance]
    else:
        reb_months = None
        if rebalance_band is None:
            raise ValueError("rebalance='bands' requires rebalance_band")

    # Inflation factor for growing cashflows.
    infl_factor = None
    if cashflow_inflation_adjusted and cpi is not None:
        cpi_m = cpi.reindex(rets.index, method="ffill")
        infl_factor = (cpi_m / cpi_m.iloc[0]).fillna(1.0)

    # Per-period dividend yields per asset (for the dividend-tax haircut).
    dy_p = None
    if tax_dividend > 0 and div_yield is not None:
        dy = pd.Series(div_yield, dtype=float).reindex(w_target.index).fillna(0.0)
        dy_p = dy.values / periods

    def rate_p(date) -> float:
        if isinstance(debt_rate, pd.Series):
            r_ann = debt_rate.asof(date)
            r_ann = 0.0 if pd.isna(r_ann) else float(r_ann)
        else:
            r_ann = float(debt_rate)
        return (1 + r_ann) ** (1 / periods) - 1

    def realize(sells: np.ndarray, values: np.ndarray, basis: np.ndarray,
                loss_carry: float) -> tuple[float, float, np.ndarray]:
        """Tax on pro-rata sales with average-cost basis. Returns
        (tax, new_loss_carry, new_basis)."""
        safe = np.maximum(values, 1e-12)
        gains = np.where(values > 0, sells * (1 - basis / safe), 0.0)
        net = gains.sum() + loss_carry
        tax = tax_capgains * max(net, 0.0)
        new_carry = min(net, 0.0) if carry_losses else 0.0
        new_basis = np.where(values > 0, basis * (1 - sells / safe), basis)
        return tax, new_carry, new_basis

    equity = initial
    gross0 = equity * leverage
    debt = gross0 - equity
    holdings = gross0 * w_target.values           # dollar value per asset
    basis = holdings.copy()
    loss_carry = 0.0
    bal, twr, wts, flows, taxes = [], [], [], [], []
    reb_dates, mcall_dates = [], []
    prev_date = rets.index[0]

    for i, (date, r) in enumerate(rets.iterrows()):
        period_tax = 0.0
        # --- external cashflow at period start (skip the very first period)
        flow = 0.0
        if i > 0 and cashflow != 0.0 and _is_period_boundary(prev_date, date, cf_months):
            flow = cashflow * (float(infl_factor.loc[date]) if infl_factor is not None else 1.0)
            total = holdings.sum()
            if total - debt + flow <= 0:          # ruined by withdrawal
                holdings = np.zeros_like(holdings)
                debt = 0.0
            elif flow > 0 and total > 0:
                added = holdings * (flow / total)
                holdings = holdings + added
                basis = basis + added
            elif flow < 0 and total > 0:
                sells = holdings * (-flow / total)
                t, loss_carry, basis = realize(sells, holdings, basis, loss_carry)
                holdings = holdings - sells
                if t > 0 and holdings.sum() > 0:  # pay tax from the portfolio
                    frac = t / holdings.sum()
                    holdings = holdings * (1 - frac)
                    basis = basis * (1 - frac)
                    period_tax += t

        # --- rebalance check at period start
        total = holdings.sum()
        equity = total - debt
        if equity > 0 and i > 0:
            cur_w = holdings / total
            do_reb = False
            if reb_months is not None and _is_period_boundary(prev_date, date, reb_months):
                do_reb = True
            elif rebalance == "bands" and np.abs(cur_w - w_target.values).max() > rebalance_band:
                do_reb = True
            if do_reb:
                target_gross = equity * leverage
                new_holdings = target_gross * w_target.values
                sells = np.maximum(holdings - new_holdings, 0.0)
                t, loss_carry, basis = realize(sells, holdings, basis, loss_carry)
                buys = np.maximum(new_holdings - holdings, 0.0)
                basis = basis + buys
                holdings = new_holdings
                debt = target_gross - equity
                if t > 0 and holdings.sum() > 0:
                    frac = t / holdings.sum()
                    holdings = holdings * (1 - frac)
                    basis = basis * (1 - frac)
                    debt = debt          # tax paid by shrinking positions
                    period_tax += t
                reb_dates.append(date)

        gross_start = holdings.sum()
        equity_start = gross_start - debt
        wts.append(holdings / gross_start if gross_start > 0 else w_target.values)

        # --- apply market returns (with optional dividend-tax haircut)
        pre = holdings
        growth = 1 + r.values
        if dy_p is not None:
            growth = growth - dy_p * tax_dividend
            period_tax += float((pre * dy_p * tax_dividend).sum())
        holdings = pre * growth

        # --- debt accrual and equity
        if debt > 0:
            debt = debt * (1 + rate_p(date))
        equity_end = holdings.sum() - debt

        if equity_end <= 0 and (leverage > 1.0 or debt > 0):
            holdings = np.zeros_like(holdings)   # levered ruin
            debt = 0.0
            equity_end = 0.0

        # --- maintenance margin
        if (maintenance_margin is not None and equity_end > 0
                and holdings.sum() > 0
                and equity_end / holdings.sum() < maintenance_margin):
            target_gross = equity_end * leverage
            sell_total = holdings.sum() - target_gross
            if sell_total > 0:
                sells = holdings * (sell_total / holdings.sum())
                t, loss_carry, basis = realize(sells, holdings, basis, loss_carry)
                holdings = holdings - sells
                debt = max(debt - (sell_total - t), 0.0)
                period_tax += t
                equity_end = holdings.sum() - debt
                mcall_dates.append(date)

        twr.append(equity_end / equity_start - 1 if equity_start > 0 else 0.0)
        bal.append(equity_end)
        flows.append(flow)
        taxes.append(period_tax)
        prev_date = date

    idx = rets.index
    balance = pd.Series(bal, index=idx, name=name)
    port_rets = pd.Series(twr, index=idx, name=name)
    weights_df = pd.DataFrame(wts, index=idx, columns=w_target.index)
    cashflows = pd.Series(flows, index=idx)
    taxes_paid = pd.Series(taxes, index=idx)

    bench = None
    if benchmark is not None:
        bench = benchmark.reindex(idx).dropna()

    real_balance = None
    if cpi is not None:
        cpi_m = cpi.reindex(idx, method="ffill")
        real_balance = balance * (cpi_m.iloc[0] / cpi_m)

    return BacktestResult(balance=balance, returns=port_rets, weights=weights_df,
                          cashflows=cashflows, benchmark_returns=bench,
                          real_balance=real_balance, periods=periods, rf=rf,
                          name=name, rebalance_dates=reb_dates,
                          taxes_paid=taxes_paid, margin_calls=mcall_dates)


def compare_portfolios(
    returns: pd.DataFrame,
    allocations: dict[str, dict[str, float]],
    benchmark: pd.Series | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, dict[str, BacktestResult]]:
    """Backtest any number of allocations side by side."""
    results = {nm: backtest_portfolio(returns, w, benchmark=benchmark, name=nm, **kwargs)
               for nm, w in allocations.items()}
    table = pd.concat([r.summary().iloc[:, 0] for r in results.values()], axis=1)
    return table, results


def backtest_dynamic(
    returns: pd.DataFrame,
    schedule: dict,
    initial: float = 10_000.0,
    rf: float | pd.Series = DEFAULT_RF,
    periods: int = TRADING_DAYS,
    name: str = "Dynamic Portfolio",
) -> BacktestResult:
    """Dynamic-allocation backtest: weights change on given dates.

    schedule: {date_like: {ticker: weight}} — on each date the portfolio is
    rebalanced to the new targets; between dates weights drift with returns.
    The union of all tickers across the schedule defines the asset set.
    """
    sched = {pd.Timestamp(d): pd.Series(w, dtype=float) / sum(w.values())
             for d, w in schedule.items()}
    dates = sorted(sched)
    assets = sorted({a for w in sched.values() for a in w.index})
    rets = returns[assets].dropna(how="any")
    rets = rets.loc[rets.index >= dates[0]]
    if rets.empty:
        raise ValueError("no return history after the first schedule date")

    holdings = None
    bal, twr, wts, reb_dates = [], [], [], []
    next_i = 0
    for date, r in rets.iterrows():
        while next_i < len(dates) and dates[next_i] <= date:
            target = sched[dates[next_i]].reindex(assets).fillna(0.0)
            total = holdings.sum() if holdings is not None else initial
            holdings = total * target.values
            reb_dates.append(date)
            next_i += 1
        start_total = holdings.sum()
        wts.append(holdings / start_total if start_total > 0 else np.zeros(len(assets)))
        holdings = holdings * (1 + r.values)
        end_total = holdings.sum()
        twr.append(end_total / start_total - 1 if start_total > 0 else 0.0)
        bal.append(end_total)

    idx = rets.index
    return BacktestResult(
        balance=pd.Series(bal, index=idx, name=name),
        returns=pd.Series(twr, index=idx, name=name),
        weights=pd.DataFrame(wts, index=idx, columns=assets),
        cashflows=pd.Series(0.0, index=idx), periods=periods, rf=rf,
        name=name, rebalance_dates=reb_dates)


def glide_path(
    start_weights: dict[str, float],
    end_weights: dict[str, float],
    start: str,
    end: str,
    steps: int = 12,
) -> dict[pd.Timestamp, dict[str, float]]:
    """Linear glide-path schedule for `backtest_dynamic`: transition from a
    career-stage to a retirement-stage portfolio.

    steps: number of allocation change points between start and end inclusive.
    """
    if steps < 2:
        raise ValueError("steps must be >= 2")
    assets = sorted(set(start_weights) | set(end_weights))
    w0 = np.array([start_weights.get(a, 0.0) for a in assets], dtype=float)
    w1 = np.array([end_weights.get(a, 0.0) for a in assets], dtype=float)
    w0, w1 = w0 / w0.sum(), w1 / w1.sum()
    dates = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), periods=steps)
    sched = {}
    for k, d in enumerate(dates):
        frac = k / (steps - 1)
        w = (1 - frac) * w0 + frac * w1
        sched[d] = {a: float(x) for a, x in zip(assets, w) if x > 1e-12}
    return sched
