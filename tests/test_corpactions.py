"""Tests for portlab.corpactions.

ALL DATA HERE IS SYNTHETIC except test_fetch_actions_nvda_split, which hits
the live Yahoo Finance API for NVDA's real 10:1 split (2024-06-10) — that one
is real market history, clearly labeled, and fails loudly if the network or
Yahoo's schema changes rather than silently.
"""

import pandas as pd
import pytest

from portlab import corpactions as CA


def _pos(qty=10.0, avg=100.0):
    return {"qty": qty, "avg_cost": avg, "cost_basis": qty * avg,
            "fees_total": 0.0, "realized_pnl": 0.0}


def _act(date, symbol, action, ratio):
    return {"date": date, "symbol": symbol, "action": action, "ratio": ratio}


# ------------------------------------------------------------------ splits
def test_split_scales_qty_and_cost():
    p = _pos(10.0, 100.0)
    log = CA.apply_action(p, _act("2024-06-10", "NVDA", "SPLIT", 10.0))
    assert p["qty"] == pytest.approx(100.0)
    assert p["avg_cost"] == pytest.approx(10.0)
    assert p["cost_basis"] == pytest.approx(1000.0)  # invariant
    assert "100.000000" in log["effect"]


def test_reverse_split():
    p = _pos(100.0, 10.0)
    CA.apply_action(p, _act("2024-06-10", "NVDA", "SPLIT", 0.1))
    assert p["qty"] == pytest.approx(10.0)
    assert p["avg_cost"] == pytest.approx(100.0)
    assert p["cost_basis"] == pytest.approx(1000.0)


def test_split_no_position_is_noop():
    p = _pos(0.0, 0.0)
    log = CA.apply_action(p, _act("2024-06-10", "NVDA", "SPLIT", 10.0))
    assert p["qty"] == 0.0
    assert "no-op" in log["effect"]


def test_split_bad_ratio_raises():
    with pytest.raises(ValueError):
        CA.apply_action(_pos(), _act("2024-06-10", "NVDA", "SPLIT", 0.0))
    with pytest.raises(ValueError):
        CA.apply_action(_pos(), _act("2024-06-10", "NVDA", "SPLIT", -2.0))


def test_unknown_action_raises():
    with pytest.raises(ValueError, match="unknown corporate action"):
        CA.apply_action(_pos(), _act("2024-06-10", "NVDA", "MERGER", 1.0))


# ---------------------------------------------------------------- dividends
def test_cash_dividend_leaves_ledger_alone_but_logs_income():
    p = _pos(10.0, 100.0)
    log = CA.apply_action(p, _act("2024-03-01", "KO", "CASH_DIVIDEND", 0.50))
    assert p["qty"] == 10.0 and p["avg_cost"] == 100.0
    assert p["cost_basis"] == 1000.0
    assert "$5.00" in log["effect"]


def test_stock_dividend_behaves_like_split():
    p = _pos(100.0, 50.0)
    CA.apply_action(p, _act("2024-03-01", "X", "STOCK_DIVIDEND", 1.05))
    assert p["qty"] == pytest.approx(105.0)
    assert p["avg_cost"] == pytest.approx(50.0 / 1.05)
    assert p["cost_basis"] == pytest.approx(5000.0)


# ---------------------------------------------------------------- batch
def test_apply_actions_sorts_and_skips_empty_symbols():
    ledger = {"AAA": _pos(10.0, 100.0)}
    log = CA.apply_actions(ledger, [
        _act("2024-06-10", "AAA", "SPLIT", 2.0),
        _act("2024-01-05", "ZZZ", "SPLIT", 3.0),  # no position -> no-op
    ])
    assert ledger["AAA"]["qty"] == pytest.approx(20.0)
    assert ledger["ZZZ"]["qty"] == 0.0
    assert [str(e["date"].date()) for e in log] == ["2024-01-05", "2024-06-10"]


def test_apply_actions_accepts_dataframe():
    ledger = {"AAA": _pos(4.0, 25.0)}
    df = pd.DataFrame([_act("2024-06-10", "AAA", "SPLIT", 5.0)])
    CA.apply_actions(ledger, df)
    assert ledger["AAA"]["qty"] == pytest.approx(20.0)
    assert ledger["AAA"]["avg_cost"] == pytest.approx(5.0)


# ------------------------------------------------------- yahoo normalization
def test_actions_from_yahoo_frame():
    frame = pd.DataFrame(
        {"Dividends": [0.0, 0.50], "Stock Splits": [10.0, 0.0]},
        index=pd.to_datetime(["2024-06-10", "2024-03-01"]))
    out = CA.actions_from_yahoo_frame("nvda", frame)
    assert list(out.columns) == ["date", "symbol", "action", "ratio"]
    assert set(out["action"]) == {"SPLIT", "CASH_DIVIDEND"}
    assert (out["symbol"] == "NVDA").all()
    split = out[out["action"] == "SPLIT"].iloc[0]
    assert split["ratio"] == 10.0 and str(split["date"].date()) == "2024-06-10"


def test_actions_from_yahoo_frame_empty():
    out = CA.actions_from_yahoo_frame("AAA", pd.DataFrame())
    assert out.empty and list(out.columns) == ["date", "symbol", "action", "ratio"]


# ------------------------------------------------------- live Yahoo check
def test_fetch_actions_nvda_split():
    """REAL DATA: NVDA's 10:1 split (ex-date 2024-06-10) must appear."""
    pytest.importorskip("yfinance")
    df = CA.fetch_actions("NVDA", start="2024-06-01", end="2024-06-20")
    splits = df[df["action"] == "SPLIT"]
    assert not splits.empty, "NVDA split missing from Yahoo actions"
    row = splits.iloc[0]
    assert row["ratio"] == pytest.approx(10.0)
    assert str(row["date"].date()) == "2024-06-10"
