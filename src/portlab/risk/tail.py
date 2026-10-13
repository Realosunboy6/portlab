"""Tail diagnostics: is the normal model lying about the tails? (Hubbert Ch. 19/21)

Three layers, from cheap to serious:

1. ``jarque_bera`` — the 30-second normality kill-switch (chi2_2 test on
   skewness/kurtosis).
2. ``hill_estimator`` / ``hill_plot_data`` — non-parametric tail-index
   estimate: how heavy is the tail, really?
3. ``gpd_var_tvar`` — peaks-over-threshold GPD fit for extreme VaR/TVaR
   beyond what the data directly shows.

CONVENTION: functions take *simple returns* (positive = gain), matching the
rest of ``portlab.risk``. Internally the tail machinery works on *losses*
(positive = loss); the conversion is handled inside each function and noted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "jarque_bera",
    "hill_estimator",
    "hill_plot_data",
    "gpd_var_tvar",
    "tail_summary",
]


def jarque_bera(returns: pd.Series) -> dict:
    """Jarque–Bera normality test.

    JB = n * (S^2/6 + (K-3)^2/24) ~ chi2(2) under normality, where S is sample
    skewness and K sample kurtosis. Reject normality at 5% if JB > 5.99.
    Equity returns routinely produce JB in the hundreds or thousands.
    """
    r = pd.Series(returns, dtype=float).dropna()
    n = len(r)
    if n < 8:
        raise ValueError("jarque_bera needs at least 8 observations")
    s = float(stats.skew(r, bias=False))
    k = float(stats.kurtosis(r, fisher=False, bias=False))  # Pearson: 3 = normal
    jb = n * (s ** 2 / 6.0 + (k - 3.0) ** 2 / 24.0)
    p_value = float(stats.chi2.sf(jb, df=2))
    return {
        "n": n,
        "skew": s,
        "kurtosis": k,
        "jb_stat": float(jb),
        "p_value": p_value,
        "reject": bool(p_value < 0.05),
        "note": "H0: returns are normal. Reject at 5% if p < 0.05 (JB > 5.99).",
    }


def hill_estimator(losses: pd.Series | np.ndarray, k: int) -> dict:
    """Hill (1975) tail-index estimator on the ``k`` largest losses.

    ``losses`` must already be in loss convention (positive = loss); pass
    ``-returns``. With order statistics L_(1) >= L_(2) >= ... >= L_(k):

        xi_hat = (1/k) * sum_{i=1..k} log(L_(i) / L_(k))

    ``xi`` is the tail index: xi -> 0 is a light (exponential-like) tail,
    larger xi is heavier. Moment rule: xi >= 1/2 implies infinite variance,
    xi >= 1 implies no finite mean. Returns xi, its asymptotic standard
    error (xi / sqrt(k)), and k.
    """
    L = np.sort(np.asarray(losses, dtype=float).ravel())[::-1]  # descending
    L = L[np.isfinite(L)]
    if k < 2 or k > len(L):
        raise ValueError(f"k must satisfy 2 <= k <= n_losses ({len(L)})")
    if L[k - 1] <= 0:
        raise ValueError("threshold loss L_(k) must be positive")
    top = L[:k]
    xi = float(np.mean(np.log(top / L[k - 1])))
    return {
        "xi": xi,
        "se": float(xi / np.sqrt(k)),
        "k": k,
        "threshold": float(L[k - 1]),
        "note": ("xi>=0.5: infinite variance (variance-based risk figures "
                 "unreliable); xi>=1: no finite mean. Asymptotic se = xi/sqrt(k)."),
    }


def hill_plot_data(losses: pd.Series | np.ndarray,
                   k_range: range | None = None) -> pd.DataFrame:
    """Hill estimates over a range of ``k`` for the Hill plot.

    The Hill estimator is sensitive to the choice of k (bias-variance
    tradeoff: small k is noisy, large k drags in the body of the
    distribution). The standard practice is to plot xi_hat(k) and look for a
    stable plateau. Returns a DataFrame with columns k, xi, se.
    """
    L = np.asarray(losses, dtype=float).ravel()
    L = L[np.isfinite(L)]
    n = len(L)
    if k_range is None:
        k_range = range(5, max(6, n // 4))
    rows = []
    for k in k_range:
        if k < 2 or k > n:
            continue
        h = hill_estimator(L, k)
        rows.append({"k": k, "xi": h["xi"], "se": h["se"]})
    return pd.DataFrame(rows, columns=["k", "xi", "se"])


def gpd_var_tvar(losses: pd.Series | np.ndarray, alpha: float,
                 threshold: float) -> dict:
    """Peaks-over-threshold GPD fit → extreme VaR and TVaR.

    Exceedances Y = losses - threshold (for losses > threshold) are modelled
    as Generalized Pareto GPD(xi, beta) via MLE (``scipy.stats.genpareto``,
    location fixed at 0). Then, with n exceedances out of N observations:

        VaR_alpha = u + (beta/xi) * [ ((N/n)*alpha)^{-xi} - 1 ]
        TVaR_alpha = VaR/(1-xi) + (beta - u*xi)/(1-xi),   for xi < 1

    ``alpha`` is the tail probability (e.g. 0.01 for 99% VaR); ``losses``
    must be in loss convention (positive = loss); pass ``-returns``.

    Caveat (Hubbert Ch. 19 vs 21): the GPD derivation assumes iid exceedances,
    but financial losses cluster (volatility clustering). The modern remedy is
    to fit GARCH first and apply EVT to the standardized residuals; this
    function does the plain version and flags the caveat.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    L = np.asarray(losses, dtype=float).ravel()
    L = L[np.isfinite(L)]
    N = len(L)
    exc = L[L > threshold] - threshold
    n = len(exc)
    if n < 10:
        raise ValueError(
            f"only {n} exceedances above threshold {threshold}; need >= 10 "
            "for a usable GPD fit (lower the threshold)")
    xi, loc, beta = stats.genpareto.fit(exc, floc=0)
    if xi <= 0:
        # Light-tail fit: fall back to the empirical quantile rather than
        # extrapolating a bounded-tail model.
        var = float(np.quantile(L, 1 - alpha))
        tvar = float(L[L >= var].mean()) if (L >= var).any() else var
        return {
            "xi": float(xi), "beta": float(beta), "threshold": float(threshold),
            "n_exceedances": n, "n_total": N,
            "var": var, "tvar": tvar,
            "note": ("GPD fit gave xi<=0 (light tail); fell back to empirical "
                     "quantile. No extrapolation performed."),
        }
    ratio = (N / n) * alpha
    var = threshold + (beta / xi) * (ratio ** (-xi) - 1.0)
    if xi >= 1:
        tvar = float("inf")
        note = ("xi>=1: TVaR is infinite (no finite mean). VaR reported; "
                "TVaR undefined.")
    else:
        tvar = var / (1 - xi) + (beta - threshold * xi) / (1 - xi)
        note = ("Peaks-over-threshold GPD extrapolation. Assumes iid "
                "exceedances; volatility clustering violates this — "
                "prefer GARCH-filtered residuals for production use.")
    return {
        "xi": float(xi),
        "beta": float(beta),
        "threshold": float(threshold),
        "n_exceedances": n,
        "n_total": N,
        "var": float(var),
        "tvar": float(tvar),
        "note": note,
    }


def tail_summary(returns: pd.Series) -> dict:
    """One-row tail diagnostic summary for a return series.

    Returns dict: n, mean, std, skew, kurtosis, min, max, jb_stat, jb_p,
    jb_reject, hill_xi (k = max(10, n//20) largest losses), hill_se.
    The Hill index is computed on losses (``-returns``); only positive losses
    enter the tail, so series with no losses report xi = NaN.
    """
    r = pd.Series(returns, dtype=float).dropna()
    n = len(r)
    if n < 8:
        # Not enough history for tail diagnostics (a portfolio's first week
        # of life hits this). Report NaNs with an explanatory note instead
        # of raising — the daily monitor must never crash on young data.
        nan = float("nan")
        return {
            "n": n,
            "mean": float(r.mean()) if n else nan,
            "std": float(r.std(ddof=1)) if n > 1 else nan,
            "skew": nan,
            "kurtosis": nan,
            "min": float(r.min()) if n else nan,
            "max": float(r.max()) if n else nan,
            "jb_stat": nan,
            "jb_p": nan,
            "jb_reject": False,
            "hill_xi": nan,
            "hill_se": nan,
            "hill_k": 0,
            "note": (f"n={n}: fewer than 8 observations — Jarque-Bera, Hill "
                     f"tail index and related diagnostics are unavailable."),
        }
    jb = jarque_bera(r)
    losses = -r
    pos = losses[losses > 0]
    k = max(10, n // 20)
    if len(pos) >= k:
        hill = hill_estimator(pos.to_numpy(), k)
        xi, se = hill["xi"], hill["se"]
    else:
        xi, se = float("nan"), float("nan")
    return {
        "n": n,
        "mean": float(r.mean()),
        "std": float(r.std(ddof=1)),
        "skew": jb["skew"],
        "kurtosis": jb["kurtosis"],
        "min": float(r.min()),
        "max": float(r.max()),
        "jb_stat": jb["jb_stat"],
        "jb_p": jb["p_value"],
        "jb_reject": jb["reject"],
        "hill_xi": xi,
        "hill_se": se,
        "hill_k": k,
    }
