"""Corporate-action adjustments for fill-derived ledgers.

A stock split (or stock dividend) changes share counts but not economic value.
If the ledger ignores it, quantities and average costs silently corrupt the
moment it happens — Yahoo's *adjusted* price history does not fix your fills
or your brokerage cost basis.

Conventions (all ratios are "new shares per old share")::

    SPLIT 10:1        -> qty *= 10,  avg_cost /= 10   (cost basis unchanged)
    reverse split 1:10 -> qty /= 10, avg_cost *= 10   (cost basis unchanged)
    stock dividend     -> treated as a split with ratio (1 + rate)
    cash dividend      -> no ledger change; recorded as income in the log

Actions are dicts (or DataFrame rows) with keys::

    date, symbol, action, ratio
    action in {'SPLIT', 'STOCK_DIVIDEND', 'CASH_DIVIDEND'}
    ratio: new-per-old for splits; per-share cash amount for cash dividends

IMPORTANT: actions must be interleaved with fills in date order (an action on
the ex-date sorts *before* fills that day — fills on/after the ex-date already
trade at post-action prices). Applying a split to a ledger *after* later fills
were added scales shares that never existed pre-split. ``apply_actions`` takes
an explicit event stream so callers get the ordering right; see
``fill_ingest.py`` for the wiring.

Use :func:`fetch_actions` to pull the real corporate-action calendar from
Yahoo Finance (yfinance is imported lazily so the math core has no network
dependency).
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "SPLIT",
    "STOCK_DIVIDEND",
    "CASH_DIVIDEND",
    "ACTIONS",
    "apply_action",
    "apply_actions",
    "fetch_actions",
    "actions_from_yahoo_frame",
]

SPLIT = "SPLIT"
STOCK_DIVIDEND = "STOCK_DIVIDEND"
CASH_DIVIDEND = "CASH_DIVIDEND"
ACTIONS = frozenset({SPLIT, STOCK_DIVIDEND, CASH_DIVIDEND})


def _as_date(d) -> pd.Timestamp:
    return pd.Timestamp(d)


def apply_action(position: dict, action: dict) -> dict:
    """Apply one corporate action to a single position dict in place.

    ``position``: dict with keys ``qty``, ``avg_cost``, ``cost_basis``
    (cost_basis = qty * avg_cost is preserved across splits by construction).
    ``action``: dict with ``date``, ``symbol``, ``action``, ``ratio``.

    Returns a log-entry dict describing what happened (including no-ops and
    cash-dividend income). Raises ValueError on unknown actions or
    non-positive split ratios.
    """
    act = str(action["action"]).strip().upper()
    if act not in ACTIONS:
        raise ValueError(f"unknown corporate action: {action['action']!r}")
    ratio = float(action["ratio"])
    sym = str(action["symbol"]).strip().upper()
    date = _as_date(action["date"])
    qty = float(position.get("qty", 0.0))

    if act in (SPLIT, STOCK_DIVIDEND):
        if ratio <= 0:
            raise ValueError(f"{act} needs a positive ratio, got {ratio}")
        if qty == 0:
            return {"date": date, "symbol": sym, "action": act,
                    "ratio": ratio, "effect": "no-op (no position held)"}
        old_qty, old_avg = qty, float(position.get("avg_cost", 0.0))
        new_qty = old_qty * ratio
        new_avg = old_avg / ratio
        position["qty"] = new_qty
        position["avg_cost"] = new_avg
        # cost_basis is invariant; recompute from the rounded figures so the
        # three fields stay consistent instead of drifting apart.
        position["cost_basis"] = new_qty * new_avg
        return {"date": date, "symbol": sym, "action": act, "ratio": ratio,
                "effect": (f"qty {old_qty:.6f} -> {new_qty:.6f}, "
                           f"avg_cost {old_avg:.4f} -> {new_avg:.4f}")}
    # CASH_DIVIDEND: average-cost basis is untouched; the cash is income.
    amount = ratio  # per-share cash amount in the `ratio` field
    if amount < 0:
        raise ValueError(f"cash dividend needs a non-negative amount, got {amount}")
    income = qty * amount
    return {"date": date, "symbol": sym, "action": act, "ratio": amount,
            "effect": (f"cash income ${income:.2f} ({qty:.6f} sh @ "
                       f"${amount:.4f}/sh); cost basis unchanged")}


def apply_actions(ledger: dict, actions) -> list:
    """Apply an iterable of actions to a ledger dict in place.

    ``ledger``: dict symbol -> position dict (same shape as
    ``fill_ingest.build_ledger`` output: qty/avg_cost/cost_basis/...).
    ``actions``: iterable of dicts (or DataFrame rows) with
    date/symbol/action/ratio. Applied in (date, symbol) order.

    Returns the list of log entries. Prefer interleaving actions with fills in
    date order (see module docstring) over calling this on a finished ledger —
    this function is correct only when every action postdates every fill for
    that symbol.
    """
    if isinstance(actions, pd.DataFrame):
        rows = actions.to_dict("records")
    else:
        rows = list(actions)
    rows = sorted(rows, key=lambda a: (_as_date(a["date"]),
                                       str(a["symbol"]).strip().upper()))
    log = []
    for a in rows:
        sym = str(a["symbol"]).strip().upper()
        pos = ledger.setdefault(
            sym, {"qty": 0.0, "avg_cost": 0.0, "cost_basis": 0.0,
                  "fees_total": 0.0, "realized_pnl": 0.0})
        log.append(apply_action(pos, a))
    return log


def actions_from_yahoo_frame(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize one yfinance ``Ticker(...).actions`` frame to our schema.

    yfinance ``actions`` has a DatetimeIndex (ex-dates) and columns
    ``Dividends`` (per-share cash) and ``Stock Splits`` (new-per-old ratio,
    0.0 on non-split days). Returns a DataFrame with columns
    date/symbol/action/ratio, one row per nonzero event.
    """
    symbol = str(symbol).strip().upper()
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["date", "symbol", "action", "ratio"])
    rows = []
    for ts, row in frame.iterrows():
        div = float(row.get("Dividends", 0.0) or 0.0)
        split = float(row.get("Stock Splits", 0.0) or 0.0)
        if split and split != 1.0:
            rows.append({"date": pd.Timestamp(ts), "symbol": symbol,
                         "action": SPLIT, "ratio": split})
        if div:
            rows.append({"date": pd.Timestamp(ts), "symbol": symbol,
                         "action": CASH_DIVIDEND, "ratio": div})
    return pd.DataFrame(rows, columns=["date", "symbol", "action", "ratio"])


def fetch_actions(symbols, start=None, end=None) -> pd.DataFrame:
    """Pull corporate actions for ``symbols`` from Yahoo Finance.

    Returns one normalized DataFrame (date/symbol/action/ratio) across all
    symbols, sorted by (date, symbol). ``start``/``end`` bound the window;
    with no bounds the full Yahoo history is returned (can be slow — prefer
    passing a start). yfinance is imported lazily.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("fetch_actions needs yfinance installed") from exc
    if isinstance(symbols, str):
        symbols = [symbols]
    frames = []
    for s in symbols:
        t = yf.Ticker(s)
        try:
            raw = t.actions if start is None and end is None else t.history(
                start=start, end=end, actions=True)[["Dividends", "Stock Splits"]]
        except Exception:
            continue
        frames.append(actions_from_yahoo_frame(s, raw))
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "action", "ratio"])
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)
