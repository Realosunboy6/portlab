"""Tests for portlab.fillvalidate.

ALL DATA HERE IS SYNTHETIC — these are validation-logic checks on crafted
fill batches, not market evidence.
"""

import pandas as pd
import pytest

from portlab import fillvalidate as FV


def _fill(date="2026-09-15", symbol="MCK", side="BUY", qty=0.041, price=912.35,
          fee=0.0, order_id="FIL-001", source="fills.csv"):
    return {"date": date, "symbol": symbol, "side": side, "qty": qty,
            "price": price, "fee": fee, "order_id": order_id,
            "_source": source}


def test_clean_batch_passes():
    fills = [_fill(order_id="A1"),
             _fill(symbol="LLY", qty=0.027, price=1088.40, order_id="A2")]
    r = FV.validate_fills(fills)
    assert r.ok and not r.errors and not r.warnings


def test_missing_columns_aborts():
    r = FV.validate_fills([{"date": "2026-09-15", "symbol": "MCK"}])
    assert not r.ok
    assert r.errors[0].code == "MISSING_COLUMNS"


def test_bad_side_qty_price_date():
    fills = [_fill(side="HOLD", order_id="B1"),
             _fill(qty=-1.0, order_id="B2"),
             _fill(price=0.0, order_id="B3"),
             _fill(date="not-a-date", order_id="B4"),
             _fill(date="2999-01-01", order_id="B5")]
    r = FV.validate_fills(fills)
    codes = {i.code for i in r.errors}
    assert {"BAD_SIDE", "BAD_QTY", "BAD_PRICE", "BAD_DATE",
            "FUTURE_DATE"} <= codes


def test_duplicate_row_and_order_id():
    a = _fill(order_id="C1")
    b = _fill(order_id="C1")  # identical row AND reused order_id
    r = FV.validate_fills([a, b])
    codes = {i.code for i in r.errors}
    assert "DUPLICATE_ROW" in codes and "DUPLICATE_ORDER_ID" in codes
    assert not r.ok


def test_oversell_detected_in_order():
    fills = [_fill(side="SELL", qty=1.0, order_id="D1"),   # nothing held yet
             _fill(side="BUY", qty=0.5, order_id="D2")]
    r = FV.validate_fills(fills)
    assert any(i.code == "OVERSELL" for i in r.errors)


def test_valid_sell_after_buy_passes():
    fills = [_fill(side="BUY", qty=1.0, order_id="E1"),
             _fill(side="SELL", qty=0.4, order_id="E2")]
    assert FV.validate_fills(fills).ok


def test_below_minimum_is_warning_not_error():
    r = FV.validate_fills([_fill(qty=0.0001, price=1.0, order_id="F1")])
    assert r.ok  # warnings don't block
    assert any(i.code == "BELOW_MIN_ORDER" for i in r.warnings)


def test_price_outlier_needs_prices_and_flags():
    fills = [_fill(price=912.35, order_id="G1"),
             _fill(price=5000.00, order_id="G2")]  # 5x the market
    prices = pd.DataFrame([{"date": "2026-09-15", "symbol": "MCK",
                            "price": 912.35}])
    r = FV.validate_fills(fills, prices=prices)
    assert r.ok
    out = [i for i in r.warnings if i.code == "PRICE_OUTLIER"]
    assert len(out) == 1 and out[0].row == 1


def test_no_prices_no_outlier_checks():
    r = FV.validate_fills([_fill(price=5000.00, order_id="H1")])
    assert not any(i.code == "PRICE_OUTLIER" for i in r.warnings)


def test_raise_on_error():
    r = FV.validate_fills([_fill(side="HOLD", order_id="I1")])
    with pytest.raises(ValueError, match="fill validation failed"):
        r.raise_on_error()


def test_empty_batch_errors():
    r = FV.validate_fills([])
    assert not r.ok and r.errors[0].code == "EMPTY"


def test_summary_lists_everything():
    r = FV.validate_fills([_fill(side="HOLD", order_id="J1"),
                           _fill(qty=0.0001, price=1.0, order_id="J2")])
    text = r.summary()
    assert "1 error(s), 1 warning(s)" in text
    assert "BAD_SIDE" in text and "BELOW_MIN_ORDER" in text
