"""Tax-aware lot-level accounting and rebalancing.

Tracks individual tax lots — each purchase's date, quantity and cost basis —
so sales can be planned lot by lot instead of on average cost. Provides:

- :class:`LotLedger`: buy/sell/split ledger with per-lot remaining quantities.
- Lot selection methods: ``fifo``, ``lifo``, ``hifo``, ``lofo``,
  ``min_tax`` (losses first, then long-term gains, then short-term gains —
  the standard tax-efficient order).
- Short/long-term classification on the US 1-year rule (gain is long-term
  only if held *more than* one year).
- Wash-sale handling per IRS Publication 550: a loss sale with a purchase
  of the same symbol within 30 days before *or* after has its loss
  disallowed (deferred, not forgiven) into the replacement shares' basis.
  :meth:`LotLedger.wash_sale_report` replays the full transaction history
  chronologically so the deferral lands *before* any later sale consumes
  the replacement lot; :meth:`LotLedger.realized_gains` reports
  wash-adjusted figures.
- :func:`tax_aware_sale_plan`: turn dollar sale targets into lot-level
  trades with estimated tax cost.

This is accounting, not tax advice: rates and rules change, and the module
implements the mechanics (what was realized, when, and whether a wash sale
defers it), not a tax return.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

import numpy as np
import pandas as pd

__all__ = ["TaxLot", "LotLedger", "tax_aware_sale_plan",
           "METHODS", "WASH_SALE_DAYS", "LT_DAYS"]

METHODS = ("fifo", "lifo", "hifo", "lofo", "min_tax")
WASH_SALE_DAYS = 30   # IRS Pub. 550: 30 days before or after the loss sale
LT_DAYS = 365         # > 1 year holding -> long-term

_lot_ids = count(1)
_EPS = 1e-12


@dataclass
class TaxLot:
    """One purchase lot. ``remaining`` mutates as sales consume it."""
    symbol: str
    date: pd.Timestamp
    qty: float
    basis_per_share: float
    lot_id: int = field(default_factory=lambda: next(_lot_ids))
    remaining: float = field(init=False)

    def __post_init__(self):
        self.date = pd.Timestamp(self.date)
        if self.qty <= 0:
            raise ValueError("lot qty must be positive")
        if self.basis_per_share < 0:
            raise ValueError("basis_per_share must be non-negative")
        self.remaining = float(self.qty)

    @property
    def remaining_basis(self) -> float:
        return self.remaining * self.basis_per_share


def _term(buy_date: pd.Timestamp, sell_date: pd.Timestamp) -> str:
    return "LT" if (sell_date - buy_date).days > LT_DAYS else "ST"


def _relieve(lots: list[TaxLot], symbol: str, date: pd.Timestamp,
             qty: float, price: float, method: str,
             fee: float = 0.0) -> pd.DataFrame:
    """Consume ``qty`` shares from ``lots`` (mutated), oldest-eligible first
    per ``method``. Only lots bought on or before ``date`` are eligible —
    you cannot sell shares you do not yet own. Returns per-lot rows."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    cands = [l for l in lots if l.symbol == symbol and l.date <= date
             and l.remaining > _EPS]
    avail = sum(l.remaining for l in cands)
    if qty > avail + 1e-9:
        raise ValueError(f"only {avail:.6f} shares available to sell, "
                         f"asked {qty}")
    if method == "fifo":
        ordered = sorted(cands, key=lambda l: (l.date, l.lot_id))
    elif method == "lifo":
        ordered = sorted(cands, key=lambda l: (l.date, l.lot_id), reverse=True)
    elif method == "hifo":
        ordered = sorted(cands, key=lambda l: l.basis_per_share, reverse=True)
    elif method == "lofo":
        ordered = sorted(cands, key=lambda l: l.basis_per_share)
    else:  # min_tax: losses first (most negative), then LT gains, then ST
        def key(l: TaxLot):
            gps = price - l.basis_per_share
            bucket = 0 if gps < 0 else (1 if _term(l.date, date) == "LT" else 2)
            return (bucket, gps)
        ordered = sorted(cands, key=key)
    rows, need = [], qty
    for lot in ordered:
        if need <= _EPS:
            break
        take = min(lot.remaining, need)
        proceeds = take * price
        basis = take * lot.basis_per_share
        rows.append({"lot_id": lot.lot_id, "lot_date": lot.date, "qty": take,
                     "proceeds": proceeds, "basis": basis,
                     "realized_gain": proceeds - basis,
                     "term": _term(lot.date, date)})
        lot.remaining -= take
        need -= take
    if fee:  # sale fee shrinks proceeds of the last lot relieved
        rows[-1]["proceeds"] -= fee
        rows[-1]["realized_gain"] -= fee
    return pd.DataFrame(rows)


class LotLedger:
    """Chronological lot ledger for one portfolio."""

    def __init__(self):
        self.lots: list[TaxLot] = []
        # full transaction log for the wash-sale replay
        self._buys: list[dict] = []    # date/symbol/qty/price/fee/lot_id
        self._splits: list[dict] = []  # date/symbol/ratio
        self._sales: list[dict] = []   # date/symbol/qty/price/method/fee

    # ---------------------------------------------------------- recording
    def buy(self, symbol: str, date, qty: float, price: float,
            fee: float = 0.0) -> TaxLot:
        """Record a purchase. ``fee`` is added to the lot's cost basis."""
        if qty <= 0:
            raise ValueError("buy qty must be positive")
        lot = TaxLot(symbol, date, qty, price + fee / qty)
        self.lots.append(lot)
        self._buys.append({"date": lot.date, "symbol": symbol,
                           "qty": float(qty), "price": float(price),
                           "fee": float(fee), "lot_id": lot.lot_id})
        return lot

    def apply_split(self, symbol: str, date, ratio: float) -> None:
        """Stock split / stock dividend: qty *= ratio, basis/share /= ratio.

        Same per-lot math as portlab.corpactions; applied to every open lot.
        """
        if ratio <= 0:
            raise ValueError("ratio must be positive")
        for lot in self.lots:
            if lot.symbol == symbol and lot.remaining > _EPS:
                lot.qty *= ratio
                lot.remaining *= ratio
                lot.basis_per_share /= ratio
        self._splits.append({"date": pd.Timestamp(date), "symbol": symbol,
                             "ratio": float(ratio)})

    def sell(self, symbol: str, date, qty: float, price: float,
             method: str = "min_tax", fee: float = 0.0) -> dict:
        """Sell ``qty`` shares, consuming lots per ``method``.

        Records the *gross* realized gain (wash-sale deferral is not applied
        here — see :meth:`wash_sale_report` and :meth:`realized_gains`).
        Returns the sale record with per-lot rows and the ST/LT split.
        """
        if qty <= 0:
            raise ValueError("sell qty must be positive")
        date = pd.Timestamp(date)
        rows = _relieve(self.lots, symbol, date, qty, price, method, fee)
        rec = {"symbol": symbol, "date": date, "qty": float(qty),
               "price": float(price), "method": method, "fee": float(fee),
               "lots": rows}
        rec["realized_gain"] = float(rows["realized_gain"].sum())
        rec["st_gain"] = float(rows.loc[rows["term"] == "ST",
                                        "realized_gain"].sum())
        rec["lt_gain"] = float(rows.loc[rows["term"] == "LT",
                                        "realized_gain"].sum())
        self._sales.append(rec)
        return rec

    # ---------------------------------------------------------- wash sales
    def _replay(self):
        """Replay the full history chronologically, applying wash-sale
        deferrals at the correct point: a disallowed loss bumps the
        replacement lot's basis *before* any later sale can consume it.

        Returns (wash_rows, adjusted_sales). Also patches the live lots'
        bases for replacement lots that are still open.
        """
        # fresh inventory carrying the original lot ids
        inv: dict[int, TaxLot] = {}
        for b in self._buys:
            lot = TaxLot(b["symbol"], b["date"], b["qty"],
                         b["price"] + b["fee"] / b["qty"], lot_id=b["lot_id"])
            inv[lot.lot_id] = lot
        # full buy list (any date) for the +-30d replacement search
        all_buys = self._buys
        pending_bump: dict[int, float] = {}  # lot_id -> dollars to add

        def apply_bump(lot_id: int, dollars: float):
            lot = inv[lot_id]
            base = lot.remaining if lot.remaining > _EPS else lot.qty
            lot.basis_per_share += dollars / base

        events = [("split", s["date"], s) for s in self._splits] + \
                 [("buy", b["date"], b) for b in self._buys] + \
                 [("sell", s["date"], s) for s in self._sales]
        # same-day order: splits, then buys, then sales
        events.sort(key=lambda e: (e[1], {"split": 0, "buy": 1, "sell": 2}[e[0]]))

        wash_rows, adj_sales = [], []
        for kind, _, e in events:
            if kind == "split":
                for lot in inv.values():
                    if (lot.symbol == e["symbol"] and lot.date <= e["date"]
                            and lot.remaining > _EPS):
                        lot.qty *= e["ratio"]
                        lot.remaining *= e["ratio"]
                        lot.basis_per_share /= e["ratio"]
            elif kind == "buy":
                if e["lot_id"] in pending_bump:
                    lot = inv[e["lot_id"]]
                    lot.basis_per_share += (pending_bump.pop(e["lot_id"])
                                            / lot.qty)
            else:  # sell
                rows = _relieve(list(inv.values()), e["symbol"], e["date"],
                                e["qty"], e["price"], e["method"], e["fee"])
                adj_st = float(rows.loc[rows["term"] == "ST",
                                        "realized_gain"].sum())
                adj_lt = float(rows.loc[rows["term"] == "LT",
                                        "realized_gain"].sum())
                consumed = set(rows["lot_id"])
                for _, row in rows.iterrows():
                    if row["realized_gain"] >= 0:
                        continue
                    lo = e["date"] - pd.Timedelta(days=WASH_SALE_DAYS)
                    hi = e["date"] + pd.Timedelta(days=WASH_SALE_DAYS)
                    repl = [b for b in all_buys
                            if b["symbol"] == e["symbol"]
                            and b["lot_id"] not in consumed
                            and lo <= b["date"] <= hi]
                    if not repl:
                        continue
                    repl_qty = sum(b["qty"] for b in repl)
                    frac = min(1.0, repl_qty / row["qty"])
                    disallowed = -row["realized_gain"] * frac
                    # bump the earliest replacement lot (spill to next if
                    # its remaining can't absorb — rare; keep it simple and
                    # documented: first replacement lot takes it)
                    target_id = sorted(repl, key=lambda b: b["date"])[0]["lot_id"]
                    if inv[target_id].date <= e["date"]:
                        apply_bump(target_id, disallowed)
                    else:
                        pending_bump[target_id] = \
                            pending_bump.get(target_id, 0.0) + disallowed
                    if row["term"] == "ST":
                        adj_st += disallowed
                    else:
                        adj_lt += disallowed
                    wash_rows.append(
                        {"symbol": e["symbol"], "sale_date": e["date"],
                         "lot_date": row["lot_date"], "sale_qty": row["qty"],
                         "loss": row["realized_gain"],
                         "disallowed_loss": disallowed, "term": row["term"],
                         "replacement_date":
                             sorted(repl, key=lambda b: b["date"])[0]["date"]})
                adj_sales.append({"symbol": e["symbol"], "date": e["date"],
                                  "qty": e["qty"],
                                  "realized_gain": adj_st + adj_lt,
                                  "st_gain": adj_st, "lt_gain": adj_lt,
                                  "method": e["method"]})
        # patch live open lots so unrealized() reflects statutory bases
        live = {l.lot_id: l for l in self.lots}
        for lid, lot in inv.items():
            if lid in live and live[lid].remaining > _EPS:
                live[lid].basis_per_share = lot.basis_per_share
        wash = pd.DataFrame(wash_rows,
                            columns=["symbol", "sale_date", "lot_date",
                                     "sale_qty", "loss", "disallowed_loss",
                                     "term", "replacement_date"])
        return wash, pd.DataFrame(adj_sales)

    def wash_sale_report(self) -> pd.DataFrame:
        """Loss sales with a same-symbol purchase within +-30 days (IRS Pub.
        550): the disallowed loss and the replacement purchase whose basis
        absorbs it. Empty when there are none."""
        wash, _ = self._replay()
        return wash

    def realized_gains(self, adjusted: bool = True) -> pd.DataFrame:
        """Realized gains per sale. ``adjusted=True`` (default) applies
        wash-sale deferrals via the chronological replay;
        ``adjusted=False`` gives the gross figures from :meth:`sell`."""
        if not self._sales:
            return pd.DataFrame(columns=["symbol", "date", "qty",
                                         "realized_gain", "st_gain",
                                         "lt_gain"])
        if not adjusted:
            return pd.DataFrame(
                [{"symbol": s["symbol"], "date": s["date"], "qty": s["qty"],
                  "realized_gain": s["realized_gain"], "st_gain": s["st_gain"],
                  "lt_gain": s["lt_gain"]} for s in self._sales])
        _, adj = self._replay()
        return adj

    # ---------------------------------------------------------- unrealized
    def _candidates(self, symbol: str) -> list[TaxLot]:
        return [l for l in self.lots
                if l.symbol == symbol and l.remaining > _EPS]

    def unrealized(self, prices: dict, as_of=None) -> pd.DataFrame:
        """Open lots with unrealized gain/loss and term at ``prices``.

        Call after :meth:`wash_sale_report`/:meth:`realized_gains` if wash
        sales may be present, so statutory basis adjustments are reflected.
        """
        as_of = (pd.Timestamp(as_of) if as_of is not None
                 else pd.Timestamp.now())
        rows = []
        for lot in self.lots:
            if lot.remaining <= _EPS or lot.symbol not in prices:
                continue
            px = prices[lot.symbol]
            rows.append({"symbol": lot.symbol, "lot_date": lot.date,
                         "qty": lot.remaining,
                         "basis_per_share": lot.basis_per_share,
                         "market_value": lot.remaining * px,
                         "unrealized_gain": (px - lot.basis_per_share)
                                            * lot.remaining,
                         "term": _term(lot.date, as_of)})
        return pd.DataFrame(rows)

    def harvest_candidates(self, prices: dict, as_of=None) -> pd.DataFrame:
        """Open lots sitting on losses, biggest loss first — the raw
        material for tax-loss harvesting."""
        u = self.unrealized(prices, as_of)
        if u.empty:
            return u
        return (u[u["unrealized_gain"] < 0]
                .sort_values("unrealized_gain")
                .reset_index(drop=True))

    def position(self, symbol: str) -> float:
        return sum(l.remaining for l in self._candidates(symbol))


def tax_aware_sale_plan(ledger: LotLedger, sale_targets: dict,
                       prices: dict, as_of=None,
                       method: str = "min_tax") -> pd.DataFrame:
    """Turn {symbol: dollars to raise} into lot-level trades.

    Selects lots per ``method`` (default min_tax: losses first) and reports
    per-symbol proceeds, estimated realized gain and its ST/LT split — the
    tax cost of the rebalance before it happens. Does not mutate the ledger;
    execute with :meth:`LotLedger.sell`.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    as_of = (pd.Timestamp(as_of) if as_of is not None
             else pd.Timestamp.now())
    rows = []
    for symbol, dollars in sale_targets.items():
        px = prices[symbol]
        qty = dollars / px
        cands = [l for l in ledger.lots
                 if l.symbol == symbol and l.date <= as_of
                 and l.remaining > _EPS]
        if not cands:
            raise ValueError(f"no open lots for {symbol}")
        # dry-run the relief on copies so the ledger is untouched
        sim = [TaxLot(l.symbol, l.date, l.qty, l.basis_per_share)
               for l in cands]
        for s, l in zip(sim, cands):
            s.remaining = l.remaining
        sim_rows = _relieve(sim, symbol, as_of,
                            min(qty, sum(l.remaining for l in cands)),
                            px, method)
        taken = float(sim_rows["qty"].sum())
        gain = float(sim_rows["realized_gain"].sum())
        st = float(sim_rows.loc[sim_rows["term"] == "ST",
                                "realized_gain"].sum())
        lt = float(sim_rows.loc[sim_rows["term"] == "LT",
                                "realized_gain"].sum())
        if taken < qty - 1e-9:
            raise ValueError(f"only {taken * px:.2f} raisable from {symbol}, "
                             f"needed {dollars:.2f}")
        rows.append({"symbol": symbol, "sell_qty": taken,
                     "proceeds": taken * px, "est_realized_gain": gain,
                     "est_st_gain": st, "est_lt_gain": lt, "method": method})
    return pd.DataFrame(rows).set_index("symbol")
