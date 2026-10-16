"""Validation for hand-recorded fill files.

Fills are the one input the ledger cannot second-guess — a duplicated row, a
sell typed as a buy, or a price with a misplaced decimal corrupts cost basis
permanently. This module checks a batch of fills *before* ingestion and
reports every problem with a severity, so bad data is rejected or quarantined
instead of silently booked.

Checks
------
schema      required columns; side in {BUY, SELL}; numeric qty > 0 and
            price > 0; fee >= 0 when present; parseable, non-future dates
duplicates  byte-identical rows booked twice; reused order_id
chronology  sells that exceed the holdings replayed in date order
outliers    fill price far from the market price that day (needs ``prices``)
minimums    sub-$1.00 fills (the broker's fractional-order minimum)

``validate_fills`` returns a :class:`ValidationReport`; ``report.ok`` is True
only when there are no ERRORs. ``report.raise_on_error()`` raises on the
first error for scripts that must abort ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

__all__ = [
    "ERROR",
    "WARNING",
    "ValidationIssue",
    "ValidationReport",
    "validate_fills",
]

ERROR = "ERROR"
WARNING = "WARNING"

REQUIRED_COLUMNS = ("date", "symbol", "side", "qty", "price")
SIDES = frozenset({"BUY", "SELL"})

#: The broker's minimum notional for a fractional-share order (USD).
MIN_FRACTIONAL_NOTIONAL = 1.00

#: A fill whose price is this far (relative) from the market price is flagged.
DEFAULT_PRICE_TOLERANCE = 0.25


@dataclass
class ValidationIssue:
    severity: str          # ERROR or WARNING
    code: str              # machine-readable, e.g. 'DUPLICATE_ROW'
    message: str           # human-readable
    row: int | None = None      # index into the input fills, when applicable
    source: str | None = None   # originating file, when known


@dataclass
class ValidationReport:
    issues: list = field(default_factory=list)

    @property
    def errors(self) -> list:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> list:
        return [i for i in self.issues if i.severity == WARNING]

    @property
    def ok(self) -> bool:
        """True when nothing blocks ingestion (warnings are allowed)."""
        return not self.errors

    def raise_on_error(self) -> None:
        if self.errors:
            first = self.errors[0]
            raise ValueError(
                f"fill validation failed: [{first.code}] {first.message} "
                f"({len(self.errors)} error(s), {len(self.warnings)} warning(s))")

    def summary(self) -> str:
        lines = [f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"]
        for i in self.issues:
            where = f"row {i.row}" + (f" ({i.source})" if i.source else "")
            lines.append(f"[{i.severity}] {i.code} {where}: {i.message}")
        return "\n".join(lines)


def _parse_date(raw, idx, source, issues) -> pd.Timestamp | None:
    try:
        d = pd.Timestamp(raw)
    except (ValueError, TypeError):
        issues.append(ValidationIssue(ERROR, "BAD_DATE",
                                      f"unparseable date {raw!r}", idx, source))
        return None
    if pd.isna(d):
        issues.append(ValidationIssue(ERROR, "BAD_DATE",
                                      f"unparseable date {raw!r}", idx, source))
        return None
    if d.date() > pd.Timestamp.today().date():
        issues.append(ValidationIssue(ERROR, "FUTURE_DATE",
                                      f"fill date {d.date()} is in the future",
                                      idx, source))
        return None
    return d


def _as_price_frame(prices):
    """Normalize ``prices`` to a {(date, symbol): price} lookup.

    Accepts a DataFrame with columns date/symbol/price, a CSV path, or an
    already-built dict keyed by (date, symbol).
    """
    if prices is None:
        return {}
    if isinstance(prices, dict):
        return prices
    df = pd.read_csv(prices) if isinstance(prices, str) else prices.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    return {(r["date"], r["symbol"]): float(r["price"])
            for r in df.to_dict("records")}


def validate_fills(fills, prices=None,
                   price_tolerance: float = DEFAULT_PRICE_TOLERANCE
                   ) -> ValidationReport:
    """Validate a batch of fills; never raises on bad data, only reports.

    ``fills``: list of dicts (as produced by a fill loader) with at least
    date/symbol/side/qty/price; optional fee, order_id, _source.
    ``prices``: optional market-price lookup (see :func:`_as_price_frame`).
    """
    report = ValidationReport()
    issues = report.issues
    rows = list(fills)

    if not rows:
        issues.append(ValidationIssue(ERROR, "EMPTY", "no fills to validate"))
        return report

    cols = set(rows[0].keys())
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing:
        issues.append(ValidationIssue(
            ERROR, "MISSING_COLUMNS",
            f"required columns missing: {missing}; have {sorted(cols)}"))
        return report  # nothing else is meaningful without the schema

    seen_rows: dict[tuple, int] = {}
    seen_order_ids: dict[str, int] = {}
    holdings: dict[str, float] = {}
    price_of = _as_price_frame(prices)

    for idx, f in enumerate(rows):
        source = f.get("_source")
        sym = str(f.get("symbol", "")).strip().upper()
        if not sym:
            issues.append(ValidationIssue(ERROR, "BAD_SYMBOL",
                                          "empty symbol", idx, source))
            continue
        side = str(f.get("side", "")).strip().upper()
        if side not in SIDES:
            issues.append(ValidationIssue(
                ERROR, "BAD_SIDE",
                f"side {f.get('side')!r} not in {{BUY, SELL}}", idx, source))
            continue

        d = _parse_date(f.get("date"), idx, source, issues)

        try:
            qty = float(f["qty"])
        except (TypeError, ValueError):
            issues.append(ValidationIssue(ERROR, "BAD_QTY",
                                          f"qty {f.get('qty')!r} not numeric",
                                          idx, source))
            continue
        try:
            price = float(f["price"])
        except (TypeError, ValueError):
            issues.append(ValidationIssue(ERROR, "BAD_PRICE",
                                          f"price {f.get('price')!r} not numeric",
                                          idx, source))
            continue
        if not qty > 0:
            issues.append(ValidationIssue(ERROR, "BAD_QTY",
                                          f"qty must be > 0, got {qty}",
                                          idx, source))
            continue
        if not price > 0:
            issues.append(ValidationIssue(ERROR, "BAD_PRICE",
                                          f"price must be > 0, got {price}",
                                          idx, source))
            continue
        if "fee" in f and f["fee"] not in (None, ""):
            try:
                fee = float(f["fee"])
            except (TypeError, ValueError):
                issues.append(ValidationIssue(ERROR, "BAD_FEE",
                                              f"fee {f.get('fee')!r} not numeric",
                                              idx, source))
                continue
            if fee < 0:
                issues.append(ValidationIssue(ERROR, "BAD_FEE",
                                              f"fee must be >= 0, got {fee}",
                                              idx, source))
                continue

        # --- duplicates ---------------------------------------------------
        key = (str(d.date()) if d is not None else None, sym, side,
               round(qty, 8), round(price, 8))
        if key in seen_rows:
            issues.append(ValidationIssue(
                ERROR, "DUPLICATE_ROW",
                f"identical to row {seen_rows[key]} "
                f"({sym} {side} {qty} @ {price})", idx, source))
        else:
            seen_rows[key] = idx
        oid = f.get("order_id")
        if oid:
            oid = str(oid).strip()
            if oid in seen_order_ids:
                issues.append(ValidationIssue(
                    ERROR, "DUPLICATE_ORDER_ID",
                    f"order_id {oid!r} already used at row "
                    f"{seen_order_ids[oid]}", idx, source))
            else:
                seen_order_ids[oid] = idx

        # --- chronology: sells cannot exceed replayed holdings ------------
        held = holdings.get(sym, 0.0)
        if side == "BUY":
            holdings[sym] = held + qty
        else:
            if qty > held + 1e-9:
                issues.append(ValidationIssue(
                    ERROR, "OVERSELL",
                    f"sell {qty:.6f} {sym} exceeds replayed holdings "
                    f"{held:.6f}", idx, source))
            else:
                holdings[sym] = held - qty

        # --- minimums ------------------------------------------------------
        if qty * price < MIN_FRACTIONAL_NOTIONAL - 1e-9:
            issues.append(ValidationIssue(
                WARNING, "BELOW_MIN_ORDER",
                f"fill notional ${qty * price:.2f} is below the broker's "
                f"${MIN_FRACTIONAL_NOTIONAL:.2f} fractional minimum",
                idx, source))

        # --- price outliers -------------------------------------------------
        if d is not None and price_of:
            mkt = price_of.get((d.date(), sym))
            if mkt:
                dev = abs(price - mkt) / mkt
                if dev > price_tolerance:
                    issues.append(ValidationIssue(
                        WARNING, "PRICE_OUTLIER",
                        f"fill price ${price:.2f} is {dev:.1%} away from "
                        f"market ${mkt:.2f} on {d.date()}", idx, source))

    return report
