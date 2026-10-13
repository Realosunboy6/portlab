"""VaR backtesting: did the VaR model tell the truth? (Hubbert Ch. 24)

A VaR *exception* (or "failure") occurs on any day the realized loss exceeds
the VaR forecast. If the VaR level is ``alpha`` (e.g. 5%), exceptions should
arrive independently with probability ``alpha`` — a Bernoulli(alpha) sequence.
The tests below check the two halves of that claim separately:

- *Unconditional coverage* (Kupiec POF): is the exception *rate* right?
- *Independence* (Christoffersen): are exceptions *clustered* in time?

Both likelihood-ratio statistics are asymptotic chi-square; they are only
trustworthy for T >= ~250 observations. Every test here warns below that.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "exceptions",
    "kupiec_pof",
    "christoffersen_ind",
    "joint_test",
    "basel_traffic_light",
    "MIN_BACKTEST_OBS",
]

#: Minimum sample size for the asymptotic chi-square approximations to be
#: credible (Hubbert Ch. 24; the Basel framework uses T = 250 trading days).
MIN_BACKTEST_OBS = 250


def _check_n(n: int, name: str) -> None:
    if n < MIN_BACKTEST_OBS:
        warnings.warn(
            f"{name}: only {n} observations (< {MIN_BACKTEST_OBS}); the "
            "chi-square p-value is asymptotic and should not overrule "
            "judgment on short samples.",
            UserWarning, stacklevel=3)


def exceptions(returns: pd.Series, var_series: pd.Series) -> pd.Series:
    """Boolean exception indicator: True where the loss exceeds the VaR.

    ``returns`` are simple returns (positive = gain) and ``var_series`` is a
    VaR series in the ``portlab.risk`` convention (positive number = loss),
    aligned on the same index. An exception is ``-returns > var``, i.e.
    ``returns < -var``.

    The result is a full-length nullable-boolean Series aligned to
    ``returns.index``: True/False on scored days, ``<NA>`` on unscored
    ones. A day is unscored when its VaR is NaN (no forecast was issued —
    typically the warmup of a rolling/expanding estimator) or its return
    is NaN (no outcome to score). Unscored days must not be scored as
    non-exceptions: padding them False would inflate the
    Kupiec/Christoffersen denominators with days the model never
    predicted (or that never happened), deflating the exception rate and
    biasing the tests. They are marked, not dropped, because *where* the
    holes are matters: ``christoffersen_ind`` counts transitions only
    between genuinely adjacent scored days, so a hole must stay visible
    rather than silently joining its neighbors into a fake pair.
    Callers that need a plain display series (plots, report tables)
    should ``.fillna(False)`` on the result — the padding then stays a
    display choice, never a test input. Do not ``dropna()`` the result
    before handing it to ``christoffersen_ind``: dropping the holes
    re-joins their neighbors into fake adjacent pairs.

    Alignment is strict, because silent misalignment is the worst
    failure mode here (it turns a backtest into an empty sample that
    "passes"): a non-Series ``var_series`` is aligned positionally and
    must have exactly ``len(returns)`` values, and a Series whose index
    shares no labels with ``returns`` raises instead of scoring zero
    days.
    """
    returns = pd.Series(returns, dtype=float)
    if isinstance(var_series, pd.Series):
        var_series = pd.Series(var_series, dtype=float)
        # Raise only on truly disjoint indexes. A returns slice inside
        # the VaR warmup shares every label but has NaN VaR throughout:
        # that is a legitimate empty sample (kupiec_pof will raise its
        # n=0 error), not a misalignment.
        if var_series.notna().any() and returns.notna().any() and (
                not returns.index.isin(var_series.index).any()):
            raise ValueError(
                "var_series and returns share no common index labels; "
                "align them before calling exceptions().")
        var_series = var_series.reindex(returns.index)
    else:
        var_arr = np.asarray(var_series, dtype=float).ravel()
        if len(var_arr) != len(returns):
            raise ValueError(
                f"var_series has {len(var_arr)} values but returns has "
                f"{len(returns)}; pass a Series aligned to returns or an "
                "array of equal length.")
        var_series = pd.Series(var_arr, index=returns.index)
    scored = var_series.notna() & returns.notna()
    exc = pd.Series(pd.NA, index=returns.index, dtype="boolean")
    exc[scored] = returns[scored].lt(-var_series[scored])
    return exc


def _exc_floats(exc) -> np.ndarray:
    """Coerce an exceptions series to a float array (NaN = unscored).

    Handles plain bool / 0-1 input and the nullable-boolean output of
    ``exceptions()``; an object-dtype Series holding ``pd.NA`` falls
    back through the boolean dtype, since ``astype(float)`` rejects it.
    """
    s = pd.Series(exc)
    try:
        arr = s.astype(float).to_numpy().ravel()
    except (TypeError, ValueError):
        try:
            arr = s.astype("boolean").astype(float).to_numpy().ravel()
        except (TypeError, ValueError) as err:
            # Normalize every coercion failure to ValueError, so
            # callers validating input catch one exception type.
            raise ValueError(
                "exceptions series is not coercible to boolean / 0-1 "
                f"values: {err}") from err
    vals = arr[~np.isnan(arr)]
    if vals.size and not np.isin(vals, (0.0, 1.0)).all():
        raise ValueError(
            "exceptions series must be boolean or 0/1 valued (NaN/<NA> "
            f"marks unscored days); got values {np.unique(vals)[:5]}. "
            "Pass the output of exceptions(), not raw returns.")
    return arr


def kupiec_pof(exc: pd.Series, alpha: float) -> dict:
    """Kupiec (1995) proportion-of-failures test for unconditional coverage.

    H0: exceptions are Bernoulli(alpha), i.e. the exception rate equals the
    nominal VaR level. LR = -2 * log( L(alpha) / L(p_hat) ) ~ chi2(1), where
    p_hat = (# exceptions) / T is the observed rate.

    Returns dict with n, n_exceptions, expected, lr_stat, p_value, reject
    (at 5%), and a note. ``exc`` may be boolean or 0/1, including the
    nullable-boolean output of ``exceptions()``: ``<NA>`` (unscored)
    entries are dropped before counting, so n is scored days only.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    x = _exc_floats(exc)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        raise ValueError(
            "kupiec_pof: no observations (n=0). A coverage test needs "
            "scored forecast days; an empty exceptions series is a "
            "caller error, not a pass.")
    _check_n(n, "kupiec_pof")
    t1 = float(x.sum())          # exceptions
    t0 = float(n - t1)           # non-exceptions
    p_hat = t1 / n if n else 0.0

    # Log-likelihoods; guard the boundary cases p_hat in {0, 1}.
    def _ll(p: float) -> float:
        p = min(max(p, 1e-12), 1 - 1e-12)
        return t0 * np.log(1 - p) + t1 * np.log(p)

    lr = -2.0 * (_ll(alpha) - _ll(p_hat))
    lr = max(lr, 0.0)
    p_value = float(stats.chi2.sf(lr, df=1))
    return {
        "n": n,
        "n_exceptions": int(t1),
        "expected_exceptions": n * alpha,
        "exception_rate": p_hat,
        "lr_stat": float(lr),
        "p_value": p_value,
        "reject": bool(p_value < 0.05),
        "note": ("H0: exception rate == alpha. Reject at 5% if p < 0.05. "
                 "Asymptotic: credible for n >= 250."),
    }


def christoffersen_ind(exc: pd.Series) -> dict:
    """Christoffersen (1998) independence test for exception clustering.

    H0: exceptions are serially independent (first-order Markov chain with
    equal transition probabilities). Builds the 2x2 contingency table of
    consecutive pairs (n00, n01, n10, n11) and compares the unrestricted
    Markov likelihood against the restricted (independence) likelihood.
    LR_ind ~ chi2(1).

    ``exc`` may be the nullable-boolean output of ``exceptions()``:
    transitions count only between *genuinely adjacent scored days* — a
    pair straddling an unscored day (warmup head, interior NaN gap) is
    not a consecutive pair and is skipped, never joined across the hole.
    ``n`` in the result is the number of scored days. Adjacency is
    positional on the given series: pass the full ``exceptions()``
    output — ``dropna()``-ing it first would re-join holes invisibly,
    and rows missing from the index entirely cannot be detected here.

    The restricted (independence) likelihood uses the pair-based MLE
    pi = (n01 + n11) / (total pairs) — the maximizer of the
    transition-count likelihood under p01 = p11 — not the day-count
    exception mean.

    With zero exceptions the table degenerates and the test is undefined;
    returns NaN statistics with an explanatory note instead of raising.
    The same applies when exceptions exist but the adjacent pairs cannot
    support the test: no pairs at all, no pair starting in one of the
    two states, or no pair ending in one of the two states (which puts
    the restricted MLE on the boundary of the parameter space).
    """
    xf = _exc_floats(exc)
    scored = ~np.isnan(xf)
    n = int(scored.sum())
    if n == 0:
        raise ValueError("christoffersen_ind: no observations (n=0).")
    _check_n(n, "christoffersen_ind")
    x = xf[scored].astype(bool)
    pair_ok = scored[:-1] & scored[1:]
    prev = xf[:-1][pair_ok].astype(bool)
    curr = xf[1:][pair_ok].astype(bool)
    if x.sum() == 0:
        return {
            "n": n,
            "lr_stat": float("nan"),
            "p_value": float("nan"),
            "reject": False,
            "n00": int(pair_ok.sum()), "n01": 0, "n10": 0, "n11": 0,
            "note": ("No exceptions observed: the independence test is "
                     "undefined (no transitions to estimate). Coverage is "
                     "trivially fine; independence is vacuous."),
        }
    n00 = int((~prev & ~curr).sum())
    n01 = int((~prev & curr).sum())
    n10 = int((prev & ~curr).sum())
    n11 = int((prev & curr).sum())
    total_pairs = n00 + n01 + n10 + n11
    if (total_pairs == 0 or (n00 + n01) == 0 or (n10 + n11) == 0
            or (n01 + n11) == 0 or (n00 + n10) == 0):
        # Exceptions exist, but the adjacent scored pairs cannot
        # identify both transition probabilities (e.g. every exception
        # is isolated by holes, or the only exception is the final
        # day). Returning LR 0 / p 1 here would fabricate a pass.
        return {
            "n": n,
            "lr_stat": float("nan"),
            "p_value": float("nan"),
            "reject": False,
            "n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "note": ("Independence test undefined: adjacent scored pairs "
                     f"total {total_pairs} (out of state 0: {n00 + n01}, "
                     f"out of state 1: {n10 + n11}; ending in state 0: "
                     f"{n00 + n10}, ending in state 1: {n01 + n11}); both "
                     "transition probabilities must be identified and the "
                     "restricted MLE interior to form the LR."),
        }

    def _ll(p01: float, p11: float) -> float:
        p01 = min(max(p01, 1e-12), 1 - 1e-12)
        p11 = min(max(p11, 1e-12), 1 - 1e-12)
        return (n00 * np.log(1 - p01) + n01 * np.log(p01)
                + n10 * np.log(1 - p11) + n11 * np.log(p11))

    p = (n01 + n11) / total_pairs  # restricted pair-based MLE
    lr = -2.0 * (_ll(p, p) - _ll(n01 / (n00 + n01),
                                n11 / (n10 + n11)))
    lr = max(lr, 0.0) + 0.0  # (+0.0 normalizes a possible -0.0)
    p_value = float(stats.chi2.sf(lr, df=1))
    return {
        "n": n,
        "lr_stat": float(lr),
        "p_value": p_value,
        "reject": bool(p_value < 0.05),
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "note": ("H0: exceptions serially independent. Reject at 5% if "
                 "p < 0.05 (exceptions cluster). Asymptotic: n >= 250."),
    }


def joint_test(exc: pd.Series, alpha: float) -> dict:
    """Christoffersen joint test of correct conditional coverage.

    LR_cc = LR_pof + LR_ind ~ chi2(2): a single test that the exceptions are
    both at the right rate *and* independent. Reject at 5% if p < 0.05
    (chi2_2 critical value 5.99).
    """
    kup = kupiec_pof(exc, alpha)
    ind = christoffersen_ind(exc)
    lr_pof = kup["lr_stat"]
    lr_ind = ind["lr_stat"]
    if np.isnan(lr_ind):
        # Independence undefined — either no exceptions at all, or
        # exceptions whose transitions the adjacent pairs cannot
        # identify. The joint test reduces to the coverage test (df 1);
        # the note says which case fired, since reject alone cannot.
        if kup["n_exceptions"]:
            note = ("Independence component undefined (exceptions present "
                    "but untestable from adjacent scored pairs); joint "
                    "test falls back to Kupiec POF.")
        else:
            note = ("No exceptions: independence component undefined; "
                    "joint test falls back to Kupiec POF.")
        return {
            "n": kup["n"],
            "lr_stat": float(lr_pof),
            "p_value": kup["p_value"],
            "reject": kup["reject"],
            "df": 1,
            "note": note,
        }
    lr = float(lr_pof + lr_ind)
    p_value = float(stats.chi2.sf(lr, df=2))
    return {
        "n": kup["n"],
        "lr_stat": lr,
        "p_value": p_value,
        "reject": bool(p_value < 0.05),
        "df": 2,
        "note": "H0: correct conditional coverage (rate + independence). "
                "Reject at 5% if p < 0.05. Asymptotic: n >= 250.",
    }


def basel_traffic_light(n_exceptions: int, n_obs: int = 250,
                        alpha: float = 0.01) -> dict:
    """Basel (1996) traffic-light zones for a 99% VaR backtest over 250 days.

    Green:  0–4 exceptions   (model accepted)
    Yellow: 5–9 exceptions   (supervisory attention, capital multiplier rises)
    Red:    10+ exceptions   (model rejected)

    The counts are calibrated to n=250, alpha=0.01. For other sample sizes
    the function scales the zone boundaries proportionally and says so in
    the note — but the honest guidance is: only use the textbook zones at
    (250, 0.01), and treat anything else as indicative.
    """
    if n_obs != 250 or alpha != 0.01:
        scale = (n_obs / 250) * (alpha / 0.01)
        g = max(int(round(4 * scale)), 0)
        y = max(int(round(9 * scale)), g + 1)
        note = (f"Boundaries scaled by {scale:.2f} for n={n_obs}, "
                f"alpha={alpha}: textbook Basel zones apply to (250, 0.01).")
    else:
        g, y = 4, 9
        note = "Basel 1996 zones for 99% VaR over 250 trading days."
    zone = "green" if n_exceptions <= g else ("yellow" if n_exceptions <= y
                                             else "red")
    return {
        "n_exceptions": int(n_exceptions),
        "n_obs": n_obs,
        "alpha": alpha,
        "zone": zone,
        "green_max": g,
        "yellow_max": y,
        "note": note,
    }
