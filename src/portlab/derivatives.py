"""Option pricing: Black-Scholes (European), Greeks, implied volatility,
and Cox-Ross-Rubinstein binomial trees (European or American).

Conventions: S spot, K strike, T years to expiry, r annual risk-free rate,
sigma annual volatility, q annual dividend yield. Theta is per calendar day,
vega per 1 vol point (i.e. per 0.01 sigma, divide by 100), rho per 1% rate
move (divide by 100).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             kind: str = "call", q: float = 0.0) -> float:
    """Black-Scholes price of a European call or put."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_q, disc_r = np.exp(-q * T), np.exp(-r * T)
    if kind == "call":
        return float(S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2))
    if kind == "put":
        return float(K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1))
    raise ValueError("kind must be 'call' or 'put'")


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float,
              kind: str = "call", q: float = 0.0) -> dict:
    """Delta, gamma, theta (per day), vega (per vol point), rho (per 1% rate)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_q, disc_r = np.exp(-q * T), np.exp(-r * T)
    pdf_d1 = norm.pdf(d1)
    gamma = disc_q * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * disc_q * pdf_d1 * np.sqrt(T) / 100.0
    if kind == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (-(S * pdf_d1 * sigma * disc_q) / (2 * np.sqrt(T))
                 - r * K * disc_r * norm.cdf(d2)
                 + q * S * disc_q * norm.cdf(d1)) / 365.0
        rho = K * T * disc_r * norm.cdf(d2) / 100.0
    elif kind == "put":
        delta = disc_q * (norm.cdf(d1) - 1.0)
        theta = (-(S * pdf_d1 * sigma * disc_q) / (2 * np.sqrt(T))
                 + r * K * disc_r * norm.cdf(-d2)
                 - q * S * disc_q * norm.cdf(-d1)) / 365.0
        rho = -K * T * disc_r * norm.cdf(-d2) / 100.0
    else:
        raise ValueError("kind must be 'call' or 'put'")
    return {"delta": float(delta), "gamma": float(gamma),
            "theta": float(theta), "vega": float(vega), "rho": float(rho)}


def implied_vol(price: float, S: float, K: float, T: float, r: float,
                kind: str = "call", q: float = 0.0) -> float:
    """Implied volatility via Brent's method."""
    f = lambda s: bs_price(S, K, T, r, s, kind, q) - price  # noqa: E731
    lo, hi = 1e-6, 5.0
    if f(lo) * f(hi) > 0:
        raise ValueError("price outside attainable range for these parameters")
    return float(brentq(f, lo, hi, xtol=1e-8))


def binomial_price(S: float, K: float, T: float, r: float, sigma: float,
                   n: int = 200, kind: str = "call",
                   american: bool = False, q: float = 0.0) -> float:
    """Cox-Ross-Rubinstein binomial tree price (vectorized)."""
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    j = np.arange(n + 1)
    spots = S * u ** j * d ** (n - j)
    vals = np.maximum(spots - K, 0.0) if kind == "call" \
        else np.maximum(K - spots, 0.0)
    for _ in range(n, 0, -1):
        vals = disc * (p * vals[1:] + (1 - p) * vals[:-1])
        if american:
            j = np.arange(len(vals))
            spots = S * u ** j * d ** (len(vals) - 1 - j)
            exercise = np.maximum(spots - K, 0.0) if kind == "call" \
                else np.maximum(K - spots, 0.0)
            vals = np.maximum(vals, exercise)
    return float(vals[0])
