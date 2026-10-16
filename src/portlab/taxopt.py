"""Tax-aware portfolio optimization.

Fuses lot-level tax accounting (:mod:`portlab.taxlots`) with the optimizers:
the tax cost of realizing gains — computed lot by lot under ``min_tax``
selection — enters the objective as a piecewise-linear convex cost on sales,
the way Barra-style tax-aware optimizers model it.

Single-period::

    min  (γ/2) w'Σw − μ'w + κ_tc·||w − w0||_1 + Σ_i T_i(sell_i)
    s.t. Σw = 1, w ≥ 0,  sell_i = max(w0_i − w_i, 0)

``T_i`` is the lot-implied tax cost curve: under ``min_tax`` ordering the
marginal tax rate per dollar sold is non-decreasing (harvestable losses
first at *negative* cost, then long-term gains, then short-term gains), so
``T_i`` is convex piecewise-linear and each segment becomes a bounded
variable in the QP. All quantities are in return units (tax drag per dollar
sold), so the objective stays consistent.

Multi-period (:func:`tax_aware_mpo`): one set of segment variables per asset
is shared across the horizon — lots deplete against *cumulative* sales.
Two planning approximations, both documented: the portfolio value ``V`` is
held constant over the horizon, and the lot ordering uses current prices.

Assumptions worth knowing:
- The tax cost is levied on *gross* sales (lot relief is real: selling lots
  realizes their gains/losses even if the proceeds are reinvested in the
  same asset). Consequently the optimizer may harvest losses while holding
  weights steady — sell the loser, rebuy it — when the tax benefit exceeds
  transaction costs. That is the economically correct answer under these
  assumptions, but in practice repurchasing the same security within 30
  days triggers the wash-sale rule and disallows the loss: execute harvests
  via a correlated substitute, a 31-day gap, or run
  ``LotLedger.wash_sale_report`` on the executed trades. The result reports
  ``gross_sales`` per asset so the harvest is visible separately from net
  ``trades``.
- Shares without lot history are assumed to have zero unrealized gain
  (pass ``uncovered_basis`` to override).
- Loss benefits are taken in full (the $3k/yr capital-loss limitation and
  state taxes are not modeled).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .optimize.auglag import auglag_qp
from .taxlots import LotLedger, _term, _EPS

__all__ = ["tax_cost_segments", "tax_aware_weights", "tax_aware_mpo"]

DEFAULT_ST_RATE = 0.37
DEFAULT_LT_RATE = 0.20


def tax_cost_segments(ledger: LotLedger, symbol: str, price: float,
                      st_rate: float = DEFAULT_ST_RATE,
                      lt_rate: float = DEFAULT_LT_RATE,
                      uncovered_basis: float | None = None) -> pd.DataFrame:
    """Piecewise-linear tax cost of selling ``symbol`` at ``price``.

    Walks the open lots in ``min_tax`` order; each lot is one segment with
    capacity ``remaining * price`` dollars and marginal tax drag
    ``rate * gain_per_share / price`` per dollar sold (negative for losses:
    harvesting is a benefit). Returns a DataFrame with
    ``cap_dollars`` / ``marginal_rate``, in sale order.
    """
    as_of = pd.Timestamp.now()
    lots = [l for l in ledger.lots
            if l.symbol == symbol and l.remaining > _EPS]
    if not lots and uncovered_basis is None:
        return pd.DataFrame(columns=["cap_dollars", "marginal_rate",
                                     "term", "gain_per_share"])
    # min_tax order: losses (most negative first), LT gains, ST gains
    def key(l):
        gps = price - l.basis_per_share
        bucket = 0 if gps < 0 else (1 if _term(l.date, as_of) == "LT" else 2)
        return (bucket, gps)
    segs = []
    for lot in sorted(lots, key=key):
        gps = price - lot.basis_per_share
        rate = st_rate if _term(lot.date, as_of) == "ST" else lt_rate
        segs.append({"cap_dollars": lot.remaining * price,
                     "marginal_rate": rate * gps / price,
                     "term": _term(lot.date, as_of),
                     "gain_per_share": gps})
    if uncovered_basis is not None:
        gps = price - uncovered_basis
        # uncovered shares: term unknown — treat as long-term (documented)
        segs.append({"cap_dollars": float("inf"),
                     "marginal_rate": lt_rate * gps / price,
                     "term": "LT?", "gain_per_share": gps})
    df = pd.DataFrame(segs)
    # convexity check: marginal rates must be non-decreasing in sale order
    if len(df) and bool((df["marginal_rate"].diff().dropna() < -1e-12).any()):
        raise ValueError("tax curve is not convex — check lot ordering")
    return df


def _build_qp(w0, mus, covs, ledger, prices, V, risk_aversion, tc_bps,
              st_rate, lt_rate, uncovered_basis):
    """Shared QP construction. Returns (C, c, A, b, meta)."""
    T, n = mus.shape
    assets = list(mus.columns)
    px = pd.Series(prices, dtype=float).reindex(assets)
    # per-asset segments in WEIGHT units: cap^w = cap_dollars / V
    seg_caps, seg_rates = [], []
    for a in assets:
        segs = tax_cost_segments(ledger, a, float(px[a]), st_rate, lt_rate,
                                 uncovered_basis.get(a) if uncovered_basis
                                 else None)
        # the uncovered fallback has infinite cap — replace with a cap far
        # above any feasible sale (sales can always exceed lot coverage;
        # they just use the fallback rate)
        caps = segs["cap_dollars"].replace([np.inf], 1e9) / V
        seg_caps.append(caps.to_numpy())
        seg_rates.append(segs["marginal_rate"].to_numpy())
    nseg = [len(c) for c in seg_caps]
    S = sum(nseg)
    # variable order: [w (T*n) | zp (T*n) | zm (T*n) | x (S)]
    N = 3 * n * T + S
    x_off = 3 * n * T
    seg_of = []  # per asset: (start, stop) in x-block
    s = x_off
    for k in nseg:
        seg_of.append((s, s + k))
        s += k

    kappa = tc_bps / 1e4
    C = np.zeros((N, N))
    for t in range(T):
        C[t * n:(t + 1) * n, t * n:(t + 1) * n] = \
            0.5 * risk_aversion * np.asarray(covs[t], dtype=float)
    c = np.zeros(N)
    c[:T * n] = -np.asarray(mus, dtype=float).ravel()
    c[T * n:3 * T * n] = kappa
    for i, (a0, a1) in enumerate(seg_of):
        c[a0:a1] = seg_rates[i]

    # equalities: budget (T), w_t − zp_t + zm_t = w_{t−1} (nT),
    #             zm_{t,i} summed over t = Σ_k x_{ik} per asset with lots (n')
    n_linked = sum(1 for k in nseg if k > 0)
    n_eq = T + n * T + n_linked
    A = np.zeros((n_eq, N))
    b = np.zeros(n_eq)
    for t in range(T):
        A[t, t * n:(t + 1) * n] = 1.0
        b[t] = 1.0
    for t in range(T):
        rows = slice(T + t * n, T + (t + 1) * n)
        A[rows, t * n:(t + 1) * n] = np.eye(n)
        A[rows, T * n + t * n:T * n + (t + 1) * n] = -np.eye(n)   # zp
        A[rows, 2 * T * n + t * n:2 * T * n + (t + 1) * n] = np.eye(n)  # zm
        if t == 0:
            b[rows] = np.asarray(w0, dtype=float)
        else:
            A[rows, (t - 1) * n:t * n] = -np.eye(n)
    r = T + n * T
    for i, (a0, a1) in enumerate(seg_of):
        if a1 == a0:
            continue
        for t in range(T):  # cumulative sales of asset i = segment fill
            A[r, 2 * T * n + t * n + i] = 1.0
        A[r, a0:a1] = -1.0
        r += 1
    # segment bounds via y ≥ 0 plus upper bounds: handled by clipping in the
    # solver call below (auglag only supports y ≥ 0, so we split each x into
    # x and (cap − x) with an equality — see _solve_with_caps)
    meta = {"assets": assets, "T": T, "n": n, "x_off": x_off,
            "seg_of": seg_of, "seg_caps": seg_caps, "S": S, "N": N,
            "n_eq_base": T + n * T}
    return C, c, A, b, meta


def _solve_with_caps(C, c, A, b, meta):
    """auglag_qp only handles y ≥ 0; segment caps x ≤ cap become slacks
    s = cap − x ≥ 0 with an extra equality row each."""
    N, S = meta["N"], meta["S"]
    x_off = meta["x_off"]
    caps = np.concatenate([np.concatenate(
        [np.full(1, v) for v in c_]) if len(c_) else np.empty(0)
        for c_ in meta["seg_caps"]])
    M = N + S
    C2 = np.zeros((M, M))
    C2[:N, :N] = C
    c2 = np.concatenate([c, np.zeros(S)])
    n_extra = S
    A2 = np.zeros((A.shape[0] + n_extra, M))
    A2[:A.shape[0], :N] = A
    b2 = np.concatenate([b, caps])
    for j in range(S):
        A2[A.shape[0] + j, x_off + j] = 1.0      # x_j
        A2[A.shape[0] + j, N + j] = 1.0          # + slack_j = cap_j
    y0 = np.zeros(M)
    # warm start: equal weights, no trades
    n, T = meta["n"], meta["T"]
    y0[:T * n] = np.full(T * n, 1.0 / n)
    y = auglag_qp(C2, A2, b2, y0=y0, c=c2)
    return y[:N]


def tax_aware_weights(w0, mu, cov, ledger, prices, portfolio_value,
                      risk_aversion: float = 5.0, tc_bps: float = 10.0,
                      st_rate: float = DEFAULT_ST_RATE,
                      lt_rate: float = DEFAULT_LT_RATE,
                      uncovered_basis: dict | None = None) -> dict:
    """Single-period tax-aware mean-variance weights.

    Minimizes (γ/2)w'Σw − μ'w + turnover penalty + lot-implied tax cost.
    ``w0``: current weights (Series, sums to 1); ``portfolio_value``: dollars
    (converts weight sales to dollars against the lot caps).
    Returns dict with ``weights``, ``trades``, ``tax_cost`` (dollars of tax),
    ``tax_cost_by_asset`` and ``objective``.
    """
    w0s = pd.Series(w0, dtype=float)
    mu1 = pd.DataFrame([pd.Series(mu, dtype=float)])
    C, c, A, b, meta = _build_qp(w0s.to_numpy(), mu1,
                                 [np.asarray(cov, dtype=float)],
                                 ledger, prices, portfolio_value,
                                 risk_aversion, tc_bps, st_rate, lt_rate,
                                 uncovered_basis or {})
    y = _solve_with_caps(C, c, A, b, meta)
    return _result(y, C, c, meta, w0s, portfolio_value)


def tax_aware_mpo(w0, mus, covs, ledger, prices, portfolio_value,
                  risk_aversion: float = 5.0, tc_bps: float = 10.0,
                  st_rate: float = DEFAULT_ST_RATE,
                  lt_rate: float = DEFAULT_LT_RATE,
                  uncovered_basis: dict | None = None) -> dict:
    """Multi-period tax-aware weights (segment variables shared across the
    horizon; lots deplete against cumulative sales)."""
    w0s = pd.Series(w0, dtype=float)
    mu = pd.DataFrame(mus, dtype=float)
    C, c, A, b, meta = _build_qp(w0s.to_numpy(), mu,
                                 [np.asarray(x, dtype=float) for x in covs],
                                 ledger, prices, portfolio_value,
                                 risk_aversion, tc_bps, st_rate, lt_rate,
                                 uncovered_basis or {})
    y = _solve_with_caps(C, c, A, b, meta)
    return _result(y, C, c, meta, w0s, portfolio_value)


def _result(y, C, c, meta, w0, V):
    n, T, assets = meta["n"], meta["T"], meta["assets"]
    wmat = y[:T * n].reshape(T, n)
    weights = pd.DataFrame(wmat, columns=assets)
    x = y[meta["x_off"]:meta["x_off"] + meta["S"]]
    zm = y[2 * T * n:3 * T * n].reshape(T, n)
    tax_by_asset, tax_total = {}, 0.0
    for i, (a0, a1) in enumerate(meta["seg_of"]):
        if a1 > a0:
            t = float(c[a0:a1] @ x[a0 - meta["x_off"]:a1 - meta["x_off"]])
        else:
            t = 0.0
        tax_by_asset[assets[i]] = t * V  # return units -> dollars
        tax_total += t * V
    trades = weights.diff()
    trades.iloc[0] = weights.iloc[0] - w0.to_numpy()
    gross_sales = pd.Series(zm.sum(axis=0) * V, index=assets,
                            name="gross_sales_dollars")
    return {"weights": weights, "trades": trades,
            "gross_sales": gross_sales,
            "tax_cost": tax_total, "tax_cost_by_asset": tax_by_asset,
            "objective": float(y @ C @ y + c @ y)}
