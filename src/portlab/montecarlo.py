"""Monte Carlo simulation of portfolio growth and retirement withdrawals.

Return models:
  - 'bootstrap'        IID resampling of historical portfolio returns
  - 'block_bootstrap'  circular block resampling (preserves autocorrelation)
  - 'normal'           parameterized normal from given mean/vol
  - 'student_t'        parameterized fat-tailed Student-t
  - 'statistical'      multivariate normal from asset mu/cov + weights

The parameterized models accept per-year sequences for mean/vol, and the
statistical model accepts a year-by-year `weights_schedule` — together with
`glide_weights()` this simulates life-cycle glide paths.

Withdrawal rules:
  - none, fixed real amount, fixed percentage, smoothed percentage
    (rolling average balance), RMD-style 1/remaining-life-expectancy,
    custom per-period schedule,
  - 'vpw'        Bogleheads Variable Percentage Withdrawal (annuity-based)
  - 'guardrails' Guyton-Klinger decision rules
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MONTHS_PER_YEAR


@dataclass
class MonteCarloResult:
    balances: np.ndarray          # (n_sims, T+1) nominal balances
    withdrawals: np.ndarray       # (n_sims, T) per-period withdrawals taken
    freq: int = MONTHS_PER_YEAR

    @property
    def success_rate(self) -> float:
        """Fraction of paths that never hit zero."""
        return float((self.balances.min(axis=1) > 0).mean())

    def percentiles(self, qs=(10, 25, 50, 75, 90)) -> pd.DataFrame:
        years = np.arange(self.balances.shape[1]) / self.freq
        data = {f"p{q}": np.percentile(self.balances, q, axis=0) for q in qs}
        return pd.DataFrame(data, index=pd.Index(years, name="year"))

    def ending_stats(self) -> pd.Series:
        end = self.balances[:, -1]
        return pd.Series({
            "Success Rate": self.success_rate,
            "Median Ending Balance": float(np.median(end)),
            "Mean Ending Balance": float(end.mean()),
            "10th Percentile": float(np.percentile(end, 10)),
            "90th Percentile": float(np.percentile(end, 90)),
            "Prob. Ending < Initial": float((end < self.balances[0, 0]).mean()),
        })

    def income_stats(self) -> pd.Series:
        """Withdrawal-income adequacy (the right lens for VPW-style rules)."""
        annual = self.withdrawals.reshape(self.withdrawals.shape[0], -1, self.freq).sum(axis=2)
        return pd.Series({
            "Median Annual Withdrawal": float(np.median(annual)),
            "P10 Annual Withdrawal": float(np.percentile(annual, 10)),
            "Minimum Annual Withdrawal": float(annual.min()),
        })


def _per_period_params(values, years: int, freq: int, transform) -> np.ndarray | None:
    """Broadcast a scalar or per-year sequence to a length-T per-period array."""
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = np.full(years, float(arr))
    if len(arr) != years:
        raise ValueError(f"per-year parameter needs length {years}, got {len(arr)}")
    return np.repeat(transform(arr), freq)


def glide_weights(start_weights: pd.Series | dict, end_weights: pd.Series | dict,
                  years: int) -> pd.DataFrame:
    """Year-by-year linear glide path for the 'statistical' model's
    `weights_schedule` (rows = year 0..years-1)."""
    s, e = pd.Series(start_weights, dtype=float), pd.Series(end_weights, dtype=float)
    assets = sorted(set(s.index) | set(e.index))
    s = s.reindex(assets).fillna(0.0); s = s / s.sum()
    e = e.reindex(assets).fillna(0.0); e = e / e.sum()
    fracs = np.linspace(0, 1, years)
    return pd.DataFrame([(1 - f) * s.values + f * e.values for f in fracs],
                        columns=assets, index=pd.Index(range(years), name="year"))


def _simulate_returns(
    model: str, n_sims: int, T: int, rng: np.random.Generator,
    hist_returns: pd.Series | None = None,
    mean_annual=0.07, vol_annual=0.15, t_dof: float = 5.0,
    asset_mu: pd.Series | None = None, asset_cov: pd.DataFrame | None = None,
    weights: pd.Series | None = None,
    weights_schedule: pd.DataFrame | None = None,
    freq: int = MONTHS_PER_YEAR, block: int = 6,
) -> np.ndarray:
    """(n_sims, T) matrix of per-period simple portfolio returns."""
    years = T // freq
    if model in ("bootstrap", "block_bootstrap"):
        if hist_returns is None or len(hist_returns) < 12:
            raise ValueError("bootstrap models need a historical return series")
        h = hist_returns.values
        if model == "bootstrap":
            idx = rng.integers(0, len(h), size=(n_sims, T))
            return h[idx]
        n_blocks = int(np.ceil(T / block))
        starts = rng.integers(0, len(h), size=(n_sims, n_blocks))
        offsets = np.arange(block)
        idx = (starts[:, :, None] + offsets[None, None, :]) % len(h)
        return h[idx].reshape(n_sims, -1)[:, :T]

    if model == "statistical":
        if asset_mu is None or asset_cov is None:
            raise ValueError("statistical model needs asset_mu and asset_cov")
        if weights_schedule is not None:
            ws = weights_schedule.reindex(columns=asset_mu.index).fillna(0.0)
            if len(ws) != years:
                raise ValueError(f"weights_schedule needs {years} rows (one per year)")
            mu_y = ws.values @ asset_mu.values
            var_y = np.einsum("ij,jk,ik->i", ws.values, asset_cov.values, ws.values)
            mu_p = np.repeat(mu_y / freq, freq)
            sd_p = np.repeat(np.sqrt(var_y / freq), freq)
        else:
            if weights is None:
                raise ValueError("statistical model needs weights or weights_schedule")
            w = weights.reindex(asset_mu.index).fillna(0.0)
            w = (w / w.sum()).values
            mu_p = np.full(T, float(w @ asset_mu.values) / freq)
            sd_p = np.full(T, np.sqrt(float(w @ asset_cov.values @ w) / freq))
        return rng.normal(mu_p, sd_p, size=(n_sims, T))

    mu_p = _per_period_params(mean_annual, years, freq, lambda a: a / freq)
    sd_p = _per_period_params(vol_annual, years, freq, lambda a: a / np.sqrt(freq))
    if model == "normal":
        return rng.normal(mu_p, sd_p, size=(n_sims, T))
    if model == "student_t":
        raw = rng.standard_t(t_dof, size=(n_sims, T))
        scale = sd_p / np.sqrt(t_dof / (t_dof - 2))
        return mu_p + raw * scale
    raise ValueError(f"unknown model {model!r}")


def _vpw_pct(age: float, depletion_age: float, r: float, cap: float) -> float:
    """Bogleheads VPW: annuity-due payment rate for the remaining horizon."""
    N = max(int(round(depletion_age - age)), 1)
    if abs(r) < 1e-9:
        pct = 1.0 / N
    else:
        pct = r / ((1 - (1 + r) ** -N) * (1 + r))
    return min(pct, cap)


def monte_carlo(
    initial: float = 1_000_000.0,
    years: int = 30,
    n_sims: int = 5_000,
    model: str = "bootstrap",
    hist_returns: pd.Series | None = None,
    mean_annual=0.07,
    vol_annual=0.15,
    t_dof: float = 5.0,
    asset_mu: pd.Series | None = None,
    asset_cov: pd.DataFrame | None = None,
    weights: pd.Series | None = None,
    weights_schedule: pd.DataFrame | None = None,
    freq: int = MONTHS_PER_YEAR,
    withdrawal: str = "none",
    withdrawal_amount: float = 0.0,       # annual, today's dollars (fixed rule)
    withdrawal_pct: float = 0.04,          # annual (percentage/guardrail rules)
    smoothing_periods: int = 12,           # smoothed-percentage rule
    life_expectancy: float = 90.0,         # RMD rule: age-based horizon
    current_age: float = 60.0,
    custom_schedule: pd.Series | None = None,  # per-period amounts (+contribution/-withdrawal)
    contribution_annual: float = 0.0,      # annual contribution (accumulation phase)
    inflation_annual: float = 0.025,
    # --- VPW rule (Bogleheads worksheet defaults) ---
    vpw_depletion_age: float = 100.0,
    vpw_stock_pct: float = 0.60,
    vpw_stock_return: float = 0.05,        # expected real return, global stocks
    vpw_bond_return: float = 0.019,        # expected real return, global bonds
    vpw_cap: float = 0.10,
    # --- Guyton-Klinger guardrails ---
    gk_guardrail: float = 0.20,            # +/- band around the initial rate
    gk_adjust: float = 0.10,               # cut/raise size
    gk_inflation_cap: float = 0.06,        # max annual inflation increase
    seed: int = 42,
) -> MonteCarloResult:
    """Simulate portfolio survival under a withdrawal/contribution policy."""
    rng = np.random.default_rng(seed)
    T = years * freq
    R = _simulate_returns(model, n_sims, T, rng, hist_returns, mean_annual,
                          vol_annual, t_dof, asset_mu, asset_cov, weights,
                          weights_schedule, freq)

    infl_p = (1 + inflation_annual) ** (1 / freq) - 1
    balances = np.empty((n_sims, T + 1))
    balances[:, 0] = initial
    withdrawals = np.zeros((n_sims, T))
    rolling = np.full((n_sims, max(smoothing_periods, 1)), initial)

    vpw_r = vpw_stock_pct * vpw_stock_return + (1 - vpw_stock_pct) * vpw_bond_return
    gk_initial_rate = withdrawal_pct
    gk_annual_wd = np.full(n_sims, initial * gk_initial_rate)

    for t in range(T):
        bal = balances[:, t]
        cpi = (1 + infl_p) ** t

        if withdrawal == "none":
            wd = np.zeros(n_sims)
        elif withdrawal == "fixed":
            wd = np.full(n_sims, withdrawal_amount / freq * cpi)
        elif withdrawal == "fixed_pct":
            wd = bal * (withdrawal_pct / freq)
        elif withdrawal == "smoothed_pct":
            wd = rolling.mean(axis=1) * (withdrawal_pct / freq)
        elif withdrawal == "rmd":
            age = current_age + t / freq
            remaining = max(life_expectancy - age, 1.0)
            wd = bal / remaining / freq
        elif withdrawal == "vpw":
            age = current_age + t / freq
            pct = _vpw_pct(age, vpw_depletion_age, vpw_r, vpw_cap)
            wd = bal * (pct / freq)
        elif withdrawal == "guardrails":
            if t > 0 and t % freq == 0:      # annual decision point
                yr_ret = np.prod(1 + R[:, t - freq:t], axis=1) - 1
                safe_bal = np.maximum(bal, 1e-9)
                rate = gk_annual_wd / safe_bal
                # Inflation Rule (capped) + Withdrawal Rule (skip after a loss
                # in overspent states)
                inc = 1 + min(inflation_annual, gk_inflation_cap)
                skip = (yr_ret < 0) & (rate > gk_initial_rate)
                gk_annual_wd = np.where(skip, gk_annual_wd, gk_annual_wd * inc)
                rate = gk_annual_wd / safe_bal
                # Capital Preservation Rule (not in the final 15 years)
                if years - t / freq > 15:
                    cut = rate > gk_initial_rate * (1 + gk_guardrail)
                    gk_annual_wd = np.where(cut, gk_annual_wd * (1 - gk_adjust),
                                            gk_annual_wd)
                # Prosperity Rule
                rate = gk_annual_wd / safe_bal
                raise_ = rate < gk_initial_rate * (1 - gk_guardrail)
                gk_annual_wd = np.where(raise_, gk_annual_wd * (1 + gk_adjust),
                                        gk_annual_wd)
            wd = np.where(bal > 0, gk_annual_wd / freq, 0.0)
        elif withdrawal == "custom":
            if custom_schedule is None:
                raise ValueError("withdrawal='custom' needs custom_schedule")
            amt = float(custom_schedule.iloc[t]) if t < len(custom_schedule) else 0.0
            wd = np.full(n_sims, max(-amt, 0.0) * cpi)
        else:
            raise ValueError(f"unknown withdrawal rule {withdrawal!r}")

        contrib = contribution_annual / freq * cpi
        if withdrawal == "custom" and custom_schedule is not None \
                and t < len(custom_schedule) and float(custom_schedule.iloc[t]) > 0:
            contrib += float(custom_schedule.iloc[t]) * cpi

        wd = np.minimum(wd, np.maximum(bal + contrib, 0.0))
        new_bal = (bal + contrib - wd) * (1 + R[:, t])
        new_bal = np.maximum(new_bal, 0.0)
        balances[:, t + 1] = new_bal
        withdrawals[:, t] = wd
        rolling = np.roll(rolling, 1, axis=1)
        rolling[:, 0] = new_bal

    return MonteCarloResult(balances=balances, withdrawals=withdrawals, freq=freq)


def goal_probability(result: "MonteCarloResult", target: float) -> dict:
    """Probability of reaching a wealth goal — PV Financial Goals.

    result: MonteCarloResult from `monte_carlo` (typically withdrawal="none"
        with `contribution_annual` set for the accumulation phase).
    target: wealth target in the same dollars as the simulation.
    Returns probability plus median / 10th-percentile ending balances.
    """
    end = result.balances[:, -1]
    return {
        "probability": float((end >= target).mean()),
        "median_ending": float(np.median(end)),
        "p10_ending": float(np.percentile(end, 10)),
        "target": float(target),
        "n_sims": int(end.shape[0]),
    }
