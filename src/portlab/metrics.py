"""Performance and risk metrics for return series.

All functions take *simple* per-period returns (a pd.Series) unless noted.
`periods` is the number of return periods per year (252 daily, 12 monthly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize as sciopt

from .config import DEFAULT_ALPHA, DEFAULT_RF, TRADING_DAYS


def _excess(rets: pd.Series, rf: float | pd.Series, periods: int) -> pd.Series:
    """Excess returns. `rf` is an annualized scalar (PV: use a T-bill Series
    from `portlab.data.riskfree` for their exact convention) or a Series of
    per-period rates aligned to `rets`."""
    if isinstance(rf, pd.Series):
        return rets - rf.reindex(rets.index).ffill().fillna(0.0)
    return rets - rf / periods


def _rf_annual(rf: float | pd.Series, periods: int) -> float:
    """Annualized scalar from a scalar or per-period Series rf."""
    if isinstance(rf, pd.Series):
        return float(rf.mean() * periods)
    return float(rf)


# ---------------------------------------------------------------- core stats

def cagr(rets: pd.Series, periods: int = TRADING_DAYS) -> float:
    growth = float((1 + rets).prod())
    if growth <= 0:
        return -1.0
    return growth ** (periods / len(rets)) - 1


def ann_vol(rets: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(rets.std(ddof=1) * np.sqrt(periods))


def sharpe(rets: pd.Series, rf: float | pd.Series = DEFAULT_RF,
           periods: int = TRADING_DAYS) -> float:
    excess = _excess(rets, rf, periods)
    sd = excess.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods))


def sortino(rets: pd.Series, rf: float | pd.Series = DEFAULT_RF,
            periods: int = TRADING_DAYS) -> float:
    excess = _excess(rets, rf, periods)
    downside = excess[excess < 0]
    dd = np.sqrt((downside ** 2).sum() / len(rets))
    if dd == 0:
        return np.inf if excess.mean() > 0 else 0.0
    return float(excess.mean() / dd * np.sqrt(periods))


def omega(rets: pd.Series, threshold_annual: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Omega ratio: sum of gains above threshold / sum of losses below it."""
    thr = threshold_annual / periods
    diff = rets - thr
    losses = -diff[diff < 0].sum()
    gains = diff[diff > 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else 1.0
    return float(gains / losses)


def calmar(rets: pd.Series, periods: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(rets)
    if mdd == 0:
        return np.inf
    return float(cagr(rets, periods) / abs(mdd))


# ---------------------------------------------------------------- drawdowns

def drawdown_series(rets: pd.Series, geometric: bool = True) -> pd.Series:
    """Drawdowns from peak; geometric=False uses additive (cumsum) wealth.

    Initial wealth (1.0) counts as the first peak, so losses taken before any
    new high are drawdowns (the standard convention)."""
    wealth = (1 + rets).cumprod() if geometric else 1 + rets.cumsum()
    peak = np.maximum(wealth.cummax(), 1.0)
    return wealth / peak - 1


def max_drawdown(rets: pd.Series) -> float:
    return float(drawdown_series(rets).min())


def drawdown_table(rets: pd.Series, top: int = 10) -> pd.DataFrame:
    """Worst drawdown episodes: start, trough, recovery, depth, durations."""
    dd = drawdown_series(rets)
    episodes = []
    in_dd = False
    start = trough = None
    trough_val = 0.0
    for date, val in dd.items():
        if not in_dd and val < 0:
            in_dd, start, trough, trough_val = True, date, date, val
        elif in_dd:
            if val < trough_val:
                trough, trough_val = date, val
            if val == 0:
                episodes.append((start, trough, date, trough_val))
                in_dd = False
    if in_dd:
        episodes.append((start, trough, pd.NaT, trough_val))
    rows = [
        {
            "Start": s,
            "Trough": t,
            "Recovery": r,
            "Depth": v,
            "Length (periods)": (dd.index.get_loc(r if pd.notna(r) else dd.index[-1])
                                 - dd.index.get_loc(s) + 1),
        }
        for s, t, r, v in episodes
    ]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Depth").head(top).reset_index(drop=True)


def ulcer_index(rets: pd.Series) -> float:
    dd = drawdown_series(rets)
    return float(np.sqrt((dd ** 2).mean()))


# ---------------------------------------------------------------- tail risk

def var_historical(rets: pd.Series, alpha: float = DEFAULT_ALPHA) -> float:
    """Historical Value-at-Risk (positive number = loss)."""
    return float(-np.quantile(rets, 1 - alpha))


def cvar_historical(rets: pd.Series, alpha: float = DEFAULT_ALPHA) -> float:
    """Expected shortfall beyond VaR (positive number = loss)."""
    cutoff = np.quantile(rets, 1 - alpha)
    tail = rets[rets <= cutoff]
    if len(tail) == 0:
        return var_historical(rets, alpha)
    return float(-tail.mean())


# ---------------------------------------------------------------- benchmark-relative

def beta_alpha(rets: pd.Series, bench: pd.Series, rf: float | pd.Series = DEFAULT_RF,
               periods: int = TRADING_DAYS) -> tuple[float, float]:
    """CAPM beta and annualized alpha vs a benchmark."""
    df = pd.concat([rets, bench], axis=1, join="inner").dropna()
    r, b = _excess(df.iloc[:, 0], rf, periods), _excess(df.iloc[:, 1], rf, periods)
    cov = np.cov(r, b)
    beta = cov[0, 1] / cov[1, 1]
    alpha = (r.mean() - beta * b.mean()) * periods
    return float(beta), float(alpha)


def tracking_error(rets: pd.Series, bench: pd.Series, periods: int = TRADING_DAYS) -> float:
    df = pd.concat([rets, bench], axis=1, join="inner").dropna()
    return float((df.iloc[:, 0] - df.iloc[:, 1]).std(ddof=1) * np.sqrt(periods))


def information_ratio(rets: pd.Series, bench: pd.Series, periods: int = TRADING_DAYS) -> float:
    df = pd.concat([rets, bench], axis=1, join="inner").dropna()
    active = df.iloc[:, 0] - df.iloc[:, 1]
    sd = active.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(active.mean() / sd * np.sqrt(periods))


def capture_ratios(rets: pd.Series, bench: pd.Series) -> tuple[float, float]:
    """(upside capture, downside capture) vs benchmark, geometric."""
    df = pd.concat([rets, bench], axis=1, join="inner").dropna()
    r, b = df.iloc[:, 0], df.iloc[:, 1]

    def _capture(mask):
        if mask.sum() == 0:
            return np.nan
        rp = (1 + r[mask]).prod() ** (1 / mask.sum()) - 1
        rb = (1 + b[mask]).prod() ** (1 / mask.sum()) - 1
        return np.nan if rb == 0 else rp / rb

    return float(_capture(b > 0)), float(_capture(b < 0))


# ---------------------------------------------------------------- cashflow-aware

def money_weighted_return(cashflows: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Annualized IRR of dated cashflows (contributions negative, final value positive).

    Index must be a DatetimeIndex. Solves NPV(rate)=0 with rate compounded annually.
    """
    t0 = cashflows.index[0]
    years = np.array([(d - t0).days / 365.25 for d in cashflows.index])
    amounts = cashflows.values.astype(float)

    def npv(rate):
        return float(np.sum(amounts / (1 + rate) ** years))

    try:
        return float(sciopt.brentq(npv, -0.9999, 10.0, maxiter=200))
    except ValueError:
        return np.nan


# ---------------------------------------------------------------- tables

def rolling_returns(rets: pd.Series, window_years: int, periods: int = TRADING_DAYS) -> pd.Series:
    """Rolling annualized return over a window of `window_years`."""
    w = window_years * periods
    return ((1 + rets).rolling(w).apply(np.prod, raw=True)) ** (1 / window_years) - 1


def annual_returns(rets: pd.Series) -> pd.Series:
    return (1 + rets).groupby(rets.index.year).prod() - 1


def monthly_return_table(rets: pd.Series) -> pd.DataFrame:
    """Year x Month table of compounded returns."""
    m = (1 + rets).groupby([rets.index.year, rets.index.month]).prod() - 1
    tbl = m.unstack()
    tbl.columns = [pd.Timestamp(2000, c, 1).strftime("%b") for c in tbl.columns]
    return tbl


def summary(rets: pd.Series, rf: float | pd.Series = DEFAULT_RF, periods: int = TRADING_DAYS,
            bench: pd.Series | None = None, name: str = "Portfolio") -> pd.Series:
    """Full metric block for one return series."""
    yr = annual_returns(rets)
    out = {
        "CAGR": cagr(rets, periods),
        "Annualized Volatility": ann_vol(rets, periods),
        "Sharpe Ratio": sharpe(rets, rf, periods),
        "Sortino Ratio": sortino(rets, rf, periods),
        "Calmar Ratio": calmar(rets, periods),
        "Omega Ratio": omega(rets, _rf_annual(rf, periods), periods),
        "Max Drawdown": max_drawdown(rets),
        "Ulcer Index": ulcer_index(rets),
        "Hist. VaR 95% (per period)": var_historical(rets),
        "Hist. CVaR 95% (per period)": cvar_historical(rets),
        "Best Year": yr.max() if len(yr) else np.nan,
        "Worst Year": yr.min() if len(yr) else np.nan,
        "Skewness": float(rets.skew()),
        "Excess Kurtosis": float(rets.kurtosis()),
    }
    if bench is not None:
        b, a = beta_alpha(rets, bench, rf, periods)
        up, down = capture_ratios(rets, bench)
        out.update({
            "Beta": b,
            "Annualized Alpha": a,
            "Tracking Error": tracking_error(rets, bench, periods),
            "Information Ratio": information_ratio(rets, bench, periods),
            "Upside Capture": up,
            "Downside Capture": down,
        })
    return pd.Series(out, name=name)


# ------------------------------------------------- parametric tail risk

def var_parametric(rets: pd.Series, alpha: float = DEFAULT_ALPHA,
                   method: str = "normal") -> float:
    """Parametric Value-at-Risk (positive = loss).

    method='normal' uses the Gaussian quantile; 'cornish_fisher' adjusts the
    quantile for the sample's skewness and excess kurtosis (modified VaR).
    The CF expansion is reliable for moderate skew/kurtosis only (|skew| ~< 2);
    for extreme distributions prefer var_historical.
    """
    from scipy import stats as _st
    mu, sd = float(rets.mean()), float(rets.std(ddof=1))
    z = float(_st.norm.ppf(1 - alpha))          # lower-tail quantile (negative)
    if method == "cornish_fisher":
        s_, k_ = float(rets.skew()), float(rets.kurtosis())
        z = (z + (z ** 2 - 1) * s_ / 6 + (z ** 3 - 3 * z) * k_ / 24
             - (2 * z ** 3 - 5 * z) * s_ ** 2 / 36)
    elif method != "normal":
        raise ValueError("method must be 'normal' or 'cornish_fisher'")
    return float(-(mu + sd * z))


def cvar_parametric(rets: pd.Series, alpha: float = DEFAULT_ALPHA) -> float:
    """Closed-form Gaussian expected shortfall (positive = loss)."""
    from scipy import stats as _st
    mu, sd = float(rets.mean()), float(rets.std(ddof=1))
    z = float(_st.norm.ppf(alpha))
    return float(sd * _st.norm.pdf(z) / (1 - alpha) - mu)


def geometric_mean(rets: pd.Series) -> float:
    """Per-period geometric mean return: (prod(1+r))^(1/n) - 1."""
    growth = float((1 + rets).prod())
    if growth <= 0:
        return -1.0
    return growth ** (1 / len(rets)) - 1


def cdar(rets: pd.Series, alpha: float = DEFAULT_ALPHA) -> float:
    """Conditional Drawdown at Risk: mean of the worst (1-alpha) fraction of
    drawdown observations (positive number). Uses a k-worst selection rather
    than a quantile cutoff so the point mass of zero-drawdown periods (every
    new high) cannot dilute the tail."""
    dd = np.sort(-drawdown_series(rets).values)[::-1]      # worst first
    k = max(int(np.ceil(round((1 - alpha) * len(dd), 9))), 1)
    return float(dd[:k].mean())


# ------------------------------------------------- benchmark & trade-style stats

def treynor(rets: pd.Series, bench: pd.Series, rf: float | pd.Series = DEFAULT_RF,
            periods: int = TRADING_DAYS) -> float:
    b, _ = beta_alpha(rets, bench, rf, periods)
    if b == 0:
        return np.nan
    return float((cagr(rets, periods) - _rf_annual(rf, periods)) / b)


def m_squared(rets: pd.Series, bench: pd.Series, rf: float | pd.Series = DEFAULT_RF,
              periods: int = TRADING_DAYS) -> float:
    """Modigliani M²: portfolio return scaled to benchmark volatility."""
    return float(_rf_annual(rf, periods) + sharpe(rets, rf, periods) * ann_vol(bench.dropna(), periods))


def r_squared(rets: pd.Series, bench: pd.Series) -> float:
    df = pd.concat([rets, bench], axis=1, join="inner").dropna()
    c = df.iloc[:, 0].corr(df.iloc[:, 1])
    return float(c ** 2)


def tail_ratio(rets: pd.Series, q: float = 0.95) -> float:
    """|right tail| / |left tail| at quantile q — >1 means fatter gains."""
    left = abs(np.quantile(rets, 1 - q))
    if left == 0:
        return np.inf
    return float(abs(np.quantile(rets, q)) / left)


def win_rate(rets: pd.Series) -> float:
    nz = rets[rets != 0]
    return float((nz > 0).mean()) if len(nz) else np.nan


def payoff_ratio(rets: pd.Series) -> float:
    wins, losses = rets[rets > 0], rets[rets < 0]
    if len(losses) == 0 or losses.mean() == 0:
        return np.inf
    return float(wins.mean() / abs(losses.mean()))


def profit_factor(rets: pd.Series) -> float:
    losses = rets[rets < 0].sum()
    if losses == 0:
        return np.inf
    return float(rets[rets > 0].sum() / abs(losses))


def gain_to_pain(rets: pd.Series) -> float:
    losses = abs(rets[rets < 0].sum())
    if losses == 0:
        return np.inf
    return float(rets.sum() / losses)


def kelly_fraction(rets: pd.Series) -> float:
    """Optimal fraction of capital per Kelly (mean/variance approximation)."""
    var = rets.var(ddof=1)
    if var == 0:
        return np.nan
    return float(rets.mean() / var)


def best_worst(rets: pd.Series) -> pd.DataFrame:
    """Best and worst compounded period returns by day/month/quarter/year."""
    rows = {}
    for label, freq in [("Period (native)", None), ("Month", "ME"),
                        ("Quarter", "QE"), ("Year", "YE")]:
        r = rets if freq is None else (1 + rets).resample(freq).prod() - 1
        rows[label] = {"Best": float(r.max()), "Worst": float(r.min())}
    return pd.DataFrame(rows).T


# ------------------------------------------------- drawdown-family ratios

def pain_index(rets: pd.Series) -> float:
    """Mean absolute drawdown over the whole history."""
    return float(-drawdown_series(rets).mean())


def pain_ratio(rets: pd.Series, rf: float | pd.Series = DEFAULT_RF,
               periods: int = TRADING_DAYS) -> float:
    pi = pain_index(rets)
    if pi == 0:
        return np.inf
    return float((cagr(rets, periods) - _rf_annual(rf, periods)) / pi)


def sterling_ratio(rets: pd.Series, periods: int = TRADING_DAYS,
                   n_worst: int = 10) -> float:
    """CAGR / |mean of the n worst drawdown depths|."""
    tbl = drawdown_table(rets, top=n_worst)
    if tbl.empty:
        return np.inf
    denom = abs(tbl["Depth"].mean())
    return float(cagr(rets, periods) / denom) if denom else np.inf


def burke_ratio(rets: pd.Series, rf: float | pd.Series = DEFAULT_RF,
                periods: int = TRADING_DAYS, n_worst: int = 10) -> float:
    """Excess CAGR / sqrt(sum of squared drawdown depths)."""
    tbl = drawdown_table(rets, top=n_worst)
    if tbl.empty:
        return np.inf
    denom = float(np.sqrt((tbl["Depth"] ** 2).sum()))
    return float((cagr(rets, periods) - _rf_annual(rf, periods)) / denom) if denom else np.inf


def recovery_factor(rets: pd.Series) -> float:
    mdd = abs(max_drawdown(rets))
    if mdd == 0:
        return np.inf
    total = float((1 + rets).prod() - 1)
    return float(total / mdd)


# ------------------------------------------------- rolling helpers

def rolling_sharpe(rets: pd.Series, window: int, rf: float | pd.Series = DEFAULT_RF,
                   periods: int = TRADING_DAYS) -> pd.Series:
    excess = _excess(rets, rf, periods)
    return (excess.rolling(window).mean()
            / excess.rolling(window).std(ddof=1) * np.sqrt(periods))


def rolling_vol(rets: pd.Series, window: int,
                periods: int = TRADING_DAYS) -> pd.Series:
    return rets.rolling(window).std(ddof=1) * np.sqrt(periods)


def rolling_beta(rets: pd.Series, bench: pd.Series, window: int) -> pd.Series:
    df = pd.concat([rets, bench], axis=1, join="inner").dropna()
    r, b = df.iloc[:, 0], df.iloc[:, 1]
    cov = r.rolling(window).cov(b)
    var = b.rolling(window).var()
    return cov / var
