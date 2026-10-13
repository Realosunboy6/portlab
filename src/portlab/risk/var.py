"""Value-at-Risk and Expected Shortfall estimation on return series.

CONVENTION (shared by every module in ``portlab.risk``):

- All functions take *simple per-period returns* as a ``pd.Series`` (positive
  numbers = gains), exactly like ``portlab.metrics``.
- VaR and ES are reported as *positive numbers = losses*, matching
  ``portlab.metrics.var_historical`` / ``cvar_historical``.
- ``alpha`` is the tail probability, so ``alpha=0.05`` means the 95% VaR.
  NOTE: this differs from ``portlab.metrics``, where ``alpha`` is the
  *confidence level* (``DEFAULT_ALPHA = 0.95``). Delegation converts via
  ``alpha_metrics = 1 - alpha`` at the call site — see the comments below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..metrics import var_historical, cvar_historical

__all__ = ["historical_var", "parametric_var", "expected_shortfall",
           "parametric_var_t", "expected_shortfall_t",
           "cornish_fisher_var", "garch_var"]


def _rolling(returns: pd.Series, window: int | None,
             fn) -> pd.Series:
    """Apply ``fn`` (a scalar VaR/ES estimator) on a rolling or expanding window.

    Alignment contract (both branches): the value at t is a *forecast for
    day t*, computed only from returns strictly before t — day t's own
    return is never in its own estimation window. The expanding branch
    leaves only entry 0 as NaN; the rolling branch leaves the first
    ``window`` entries NaN, because a forecast for day t needs ``window``
    prior days. ``fn`` receives the raw window values and returns a
    positive-loss number.

    (Pre-fix, the rolling branch stamped each window's value on the
    window's last day — pandas-rolling style — so the scored day was
    inside its own window. An external verification audit caught the
    resulting off-by-one in VaR backtest counts; see the alignment
    regression tests in tests/test_risklens.py.)
    """
    returns = pd.Series(returns, dtype=float)
    if window is None:
        # Expanding: VaR at t uses all data strictly before t (no look-ahead).
        out = pd.Series(np.nan, index=returns.index, dtype=float)
        vals = returns.to_numpy()
        for i in range(1, len(vals)):
            out.iloc[i] = fn(vals[:i])
        return out
    if window < 2:
        raise ValueError("window must be >= 2 or None")
    out = pd.Series(np.nan, index=returns.index, dtype=float)
    vals = returns.to_numpy()
    for i in range(window, len(vals)):
        out.iloc[i] = fn(vals[i - window:i])
    return out


def historical_var(returns: pd.Series, alpha: float = 0.05,
                   window: int | None = None) -> pd.Series:
    """Rolling/expanding historical-simulation VaR (positive number = loss).

    At each date the VaR is the ``(1 - alpha)``-quantile of past *losses*,
    i.e. ``-quantile(returns, alpha)`` with ``alpha`` the tail probability:
    the loss exceeded with probability ``alpha``. Delegates the window
    statistic to ``portlab.metrics.var_historical``; this function only adds
    the no-look-ahead rolling/expanding alignment. ``window=None`` gives the
    expanding estimator (all history before t).
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    # metrics.var_historical takes alpha as the CONFIDENCE level; convert.
    return _rolling(returns, window, lambda w: var_historical(
        pd.Series(w), alpha=1 - alpha))


def parametric_var(returns: pd.Series, alpha: float = 0.05,
                   window: int | None = None,
                   mean: str = "constant") -> pd.Series:
    """Rolling/expanding normal (variance-covariance) VaR (positive = loss).

    ``mean`` selects the location convention, shared with ``garch_var``
    so the two cannot disagree silently:

    - ``"constant"`` (default): VaR = -(mu - z_{1-alpha} * sigma), with
      sample mean ``mu`` and sample standard deviation ``sigma``
      (ddof=1) of the window — a mean-relative VaR.
    - ``"zero"``: VaR = z_{1-alpha} * sigma_0, with ``sigma_0`` the
      root-mean-square of the window about zero — an absolute VaR that
      treats the expected return as zero.

    Backtests must use the matching return basis: ``exceptions()`` on
    raw returns pairs with ``"zero"``; for ``"constant"``, score
    demeaned returns. Fast and smooth, but the normal model inherits
    fat-tail weakness — it is systematically too small in the tails.
    Compare against ``historical_var``; a large gap is itself a
    diagnostic (see ``portlab.risk.tail.jarque_bera``).
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if mean not in ("zero", "constant"):
        raise ValueError("mean must be 'zero' or 'constant'")
    z = stats.norm.ppf(1 - alpha)

    def _pvar(w: np.ndarray) -> float:
        w = np.asarray(w, dtype=float)
        if len(w) < 2:
            return np.nan
        if mean == "zero":
            sd0 = float(np.sqrt(np.mean(w * w)))
            return float(z * sd0)
        mu = w.mean()
        sd = w.std(ddof=1)
        return float(-(mu - z * sd))

    return _rolling(returns, window, _pvar)


def expected_shortfall(returns: pd.Series, alpha: float = 0.05,
                       window: int | None = None) -> pd.Series:
    """Rolling/expanding historical Expected Shortfall (positive = loss).

    ES is the mean loss conditional on the loss exceeding VaR — the coherent
    risk measure (sub-additive) that VaR is not. Delegates the window statistic
    to ``portlab.metrics.cvar_historical``.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    # metrics.cvar_historical takes alpha as the CONFIDENCE level; convert.
    return _rolling(returns, window, lambda w: cvar_historical(
        pd.Series(w), alpha=1 - alpha))


def _fit_t(df: float | None, w: np.ndarray) -> tuple[float, float, float]:
    """Fit (nu, loc, scale) of a Student-t to window ``w``.

    ``df=None`` -> MLE via ``scipy.stats.t.fit``; otherwise the given
    degrees of freedom with loc/scale from sample mean/std adjusted for
    the t variance (var = nu/(nu-2) * scale^2, needs nu > 2).
    """
    w = np.asarray(w, dtype=float)
    if df is not None:
        if df <= 2:
            raise ValueError("Student-t VaR needs df > 2 for a finite variance")
        mu = w.mean()
        sd = w.std(ddof=1)
        scale = sd * np.sqrt((df - 2.0) / df)
        return float(df), float(mu), float(scale)
    nu, loc, scale = stats.t.fit(w)
    return float(nu), float(loc), float(scale)


def parametric_var_t(returns: pd.Series, alpha: float = 0.05,
                     window: int | None = None,
                     df: float | None = None) -> pd.Series:
    """Rolling/expanding Student-t parametric VaR (positive number = loss).

    VaR = -(loc + scale * t_ppf(alpha, nu)). ``df=None`` fits nu by MLE on
    each window; pass ``df`` to fix it (e.g. df=5 is a common choice for
    equity tails). As nu -> infinity this converges to ``parametric_var``.

    Subtlety: variance-matched, a fat-tailed t can give a *smaller* VaR than
    normal at 95% (t(4): -1.508 sd vs -1.645 sd) — the extra tail mass sits
    *beyond* the 5% point. At 99% the ordering flips (-2.649 vs -2.326 sd).
    Report both levels; that is what RiskLens does.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if df is not None and df <= 2:
        raise ValueError("Student-t VaR needs df > 2 for a finite variance")

    def _tvar(w: np.ndarray) -> float:
        w = np.asarray(w, dtype=float)
        if len(w) < 8:
            return np.nan
        try:
            nu, loc, scale = _fit_t(df, w)
        except Exception:
            return np.nan
        if not np.isfinite(scale) or scale <= 0:
            return np.nan
        return float(-(loc + scale * stats.t.ppf(alpha, nu)))

    return _rolling(returns, window, _tvar)


def expected_shortfall_t(returns: pd.Series, alpha: float = 0.05,
                         window: int | None = None,
                         df: float | None = None) -> pd.Series:
    """Rolling/expanding Student-t Expected Shortfall (positive = loss).

    Closed form for X ~ t_nu(loc, scale):

        ES = -loc + scale * g(q)/alpha * (nu + q^2)/(nu - 1)

    with q = t_ppf(alpha, nu) and g the t density. (E[Z | Z <= q] is
    negative, so the scale term *adds* to the loss: ES >= VaR always.)
    Needs nu > 1 (finite mean); windows that fit nu <= 1 return NaN rather
    than a fiction.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if df is not None and df <= 2:
        raise ValueError("Student-t ES needs df > 2 for a finite variance")

    def _tes(w: np.ndarray) -> float:
        w = np.asarray(w, dtype=float)
        if len(w) < 8:
            return np.nan
        try:
            nu, loc, scale = _fit_t(df, w)
        except Exception:
            return np.nan
        if nu <= 1 or not np.isfinite(scale) or scale <= 0:
            return np.nan
        q = stats.t.ppf(alpha, nu)
        g = stats.t.pdf(q, nu)
        return float(-loc + scale * (g / alpha) * (nu + q ** 2) / (nu - 1))

    return _rolling(returns, window, _tes)


def cornish_fisher_var(returns: pd.Series, alpha: float = 0.05,
                       window: int | None = None) -> pd.Series:
    """Rolling/expanding Cornish-Fisher VaR (positive number = loss).

    Adjusts the normal quantile for sample skewness S and excess kurtosis K:

        z_cf = z + (z^2-1)*S/6 + (z^3-3z)*K/24 - (2z^3-5z)*S^2/36
        VaR  = -(mu + sigma * z_cf),   z = norm.ppf(alpha)

    (Zangari 1996 / RiskMetrics.) With S = K = 0 it reduces exactly to
    ``parametric_var``. A cheap way to let skew/kurtosis into VaR without a
    full distribution fit — but it is an *expansion*, not a model: with
    extreme tails it can misbehave, so compare against ``historical_var``.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    z = stats.norm.ppf(alpha)

    def _cfvar(w: np.ndarray) -> float:
        w = np.asarray(w, dtype=float)
        if len(w) < 8:
            return np.nan
        mu = w.mean()
        sd = w.std(ddof=1)
        if sd <= 0 or not np.isfinite(sd):
            return np.nan
        s = stats.skew(w, bias=False)
        k = stats.kurtosis(w, bias=False)  # excess kurtosis; 0 = normal
        if not np.isfinite(s) or not np.isfinite(k):
            return np.nan
        z_cf = (z + (z ** 2 - 1) * s / 6.0
                + (z ** 3 - 3 * z) * k / 24.0
                - (2 * z ** 3 - 5 * z) * s ** 2 / 36.0)
        return float(-(mu + sd * z_cf))

    return _rolling(returns, window, _cfvar)


def _garch11_fit(w: np.ndarray) -> tuple[float, float, float, float]:
    """Variance-targeted GARCH(1,1) MLE. Returns (omega, alpha, beta,
    var_target). Raises on non-convergence."""
    from scipy.optimize import minimize

    w = np.asarray(w, dtype=float)
    var_target = w.var()
    if not np.isfinite(var_target) or var_target <= 0:
        raise ValueError("non-positive variance target")

    def nll(theta):
        a, b = theta
        if a < 0 or b < 0 or a + b >= 0.999:
            return np.inf
        omega = var_target * (1 - a - b)
        # sigma2[0] = unconditional variance; r[t] ~ N(0, sigma2[t]), then
        # sigma2[t+1] = omega + a*r[t]^2 + b*sigma2[t]  (no look-ahead: the
        # return never enters its own variance)
        s2 = var_target
        ll = 0.0
        for r in w:
            if s2 <= 0:
                return np.inf
            ll += 0.5 * (np.log(s2) + r * r / s2)
            s2 = omega + a * r * r + b * s2
        return ll

    res = minimize(nll, x0=np.array([0.08, 0.88]), method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-8})
    if not res.success:
        raise ValueError("GARCH(1,1) MLE did not converge")
    a, b = (float(v) for v in res.x)
    return var_target * (1 - a - b), a, b, float(var_target)


def _garch11_volpath(w: np.ndarray, omega: float, a: float, b: float,
                     var_target: float) -> np.ndarray:
    """Conditional volatilities sigma_t for each observation."""
    w = np.asarray(w, dtype=float)
    s2 = var_target
    out = np.empty(len(w))
    for t, r in enumerate(w):
        out[t] = np.sqrt(s2)
        s2 = omega + a * r * r + b * s2
    return out


def _garch11_forecast(w: np.ndarray) -> float:
    """One-step-ahead GARCH(1,1) variance forecast, variance-targeted MLE.

    sigma2_{t+1} = omega + a * r_t^2 + b * sigma2_t, with
    omega = var_target * (1 - a - b)  (variance targeting: 2 free params).
    Returns the forecast *standard deviation*. Raises on non-convergence.
    """
    omega, a, b, var_target = _garch11_fit(w)
    w = np.asarray(w, dtype=float)
    s2 = var_target
    for r in w:
        s2 = omega + a * r * r + b * s2
    return float(np.sqrt(s2))


def garch_var(returns: pd.Series, alpha: float = 0.05,
              window: int = 250, dist: str = "normal",
              mean: str = "zero") -> pd.Series:
    """Rolling GARCH(1,1)-filtered VaR (positive number = loss).

    At each date, fits GARCH(1,1) on the trailing ``window`` returns and
    sets VaR_{t+1} = -(mu + sigma_forecast * q_alpha), with ``q_alpha``
    the normal quantile (``dist="normal"``) or the standardized-t
    quantile (``dist="t"``, df fit by MLE on the standardized
    residuals).

    ``mean`` selects the location convention, shared with
    ``parametric_var`` so the two cannot disagree silently:

    - ``"zero"`` (default): mu = 0 — the fit is a zero-mean GARCH on
      raw returns and the VaR is absolute. This is the conservative
      1-day default (a 250-day sample mean is mostly noise, and a
      positive mu would shrink the VaR), and it is the convention
      ``exceptions()`` backtests against raw returns.
    - ``"constant"``: the fit runs on demeaned returns and the window
      mean is added back into the quantile — a mean-relative VaR,
      matching the ``arch`` package's default specification. For a
      like-for-like backtest, score demeaned returns in
      ``exceptions()``.

    Unlike the unconditional VaR estimators, this reacts to volatility
    clustering: VaR widens after volatile days and shrinks in calm ones.
    Expensive (one MLE per date) — prefer a modest history or subsample the
    evaluation dates.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if dist not in ("normal", "t"):
        raise ValueError("dist must be 'normal' or 't'")
    if mean not in ("zero", "constant"):
        raise ValueError("mean must be 'zero' or 'constant'")
    if window < 20:
        raise ValueError("window must be >= 20 for GARCH estimation")

    returns = pd.Series(returns, dtype=float)
    vals = returns.to_numpy()
    out = pd.Series(np.nan, index=returns.index, dtype=float)
    q_norm = stats.norm.ppf(alpha)
    for i in range(window, len(vals)):
        w = vals[i - window:i]
        mu = float(w.mean()) if mean == "constant" else 0.0
        wf = w - mu if mean == "constant" else w
        try:
            sig = _garch11_forecast(wf)
        except Exception:
            continue
        if dist == "t":
            # standardized residuals -> t df, then rescale the quantile so
            # the t has unit variance: q = t_ppf(alpha, nu) * sqrt((nu-2)/nu)
            try:
                # quick recursion for residuals with the fitted params is
                # overkill here; use demeaned/scaled window as the proxy
                resid = (wf - wf.mean()) / wf.std(ddof=1)
                nu = float(stats.t.fit(resid)[0])
                if nu <= 2:
                    continue
                q = stats.t.ppf(alpha, nu) * np.sqrt((nu - 2) / nu)
            except Exception:
                continue
        else:
            q = q_norm
        out.iloc[i] = float(-(mu + sig * q))  # q < 0 -> positive loss number
    return out
