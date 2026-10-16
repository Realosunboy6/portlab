"""Tests for portlab.taxlots.

ALL DATA HERE IS SYNTHETIC and hand-computed. External anchors:
- Wash-sale 30-day window: IRS Publication 550 ("30 days before or after").
- US 1-year holding rule for long-term treatment.
- FIFO/HIFO examples are the standard textbook lot-relief cases.
"""

import pandas as pd
import pytest

from portlab import taxlots as T


@pytest.fixture()
def ledger():
    led = T.LotLedger()
    led.buy("XYZ", "2024-01-15", 100, 50.0)
    led.buy("XYZ", "2024-03-15", 100, 70.0)
    return led


def test_fifo_hand_calc(ledger):
    """Textbook: FIFO sells the Jan $50 lot -> 100 x (60-50) = +$1,000 ST."""
    rec = ledger.sell("XYZ", "2024-06-15", 100, 60.0, method="fifo")
    assert rec["realized_gain"] == pytest.approx(1000.0)
    assert rec["st_gain"] == pytest.approx(1000.0)
    assert rec["lt_gain"] == pytest.approx(0.0)
    assert rec["lots"].iloc[0]["lot_date"] == pd.Timestamp("2024-01-15")
    assert ledger.position("XYZ") == pytest.approx(100.0)


def test_hifo_hand_calc(ledger):
    """HIFO sells the Mar $70 lot -> 100 x (60-70) = -$1,000 (a loss)."""
    rec = ledger.sell("XYZ", "2024-06-15", 100, 60.0, method="hifo")
    assert rec["realized_gain"] == pytest.approx(-1000.0)
    assert rec["lots"].iloc[0]["lot_date"] == pd.Timestamp("2024-03-15")


def test_lifo_and_lofo(ledger):
    rec = ledger.sell("XYZ", "2024-06-15", 100, 60.0, method="lifo")
    assert rec["lots"].iloc[0]["lot_date"] == pd.Timestamp("2024-03-15")
    led2 = T.LotLedger()
    led2.buy("XYZ", "2024-01-15", 100, 50.0)
    led2.buy("XYZ", "2024-03-15", 100, 70.0)
    rec2 = led2.sell("XYZ", "2024-06-15", 100, 60.0, method="lofo")
    assert rec2["lots"].iloc[0]["lot_date"] == pd.Timestamp("2024-01-15")


def test_min_tax_sells_losses_first():
    led = T.LotLedger()
    led.buy("XYZ", "2024-01-15", 100, 50.0)   # +$20/share unrealized
    led.buy("XYZ", "2024-03-15", 100, 90.0)   # -$20/share unrealized
    rec = led.sell("XYZ", "2024-06-15", 100, 70.0, method="min_tax")
    # loss lot first: 100 x (70-90) = -$2,000
    assert rec["realized_gain"] == pytest.approx(-2000.0)
    assert rec["lots"].iloc[0]["lot_date"] == pd.Timestamp("2024-03-15")


def test_min_tax_prefers_lt_gains_over_st():
    led = T.LotLedger()
    led.buy("XYZ", "2022-01-15", 100, 50.0)   # LT gain lot
    led.buy("XYZ", "2024-03-15", 100, 60.0)   # ST gain lot
    rec = led.sell("XYZ", "2024-06-15", 100, 70.0, method="min_tax")
    assert rec["lots"].iloc[0]["lot_date"] == pd.Timestamp("2022-01-15")
    assert rec["lt_gain"] == pytest.approx(2000.0)
    assert rec["st_gain"] == pytest.approx(0.0)


def test_holding_period_boundary():
    led = T.LotLedger()
    led.buy("XYZ", "2023-01-01", 100, 50.0)
    # exactly 365 days: still short-term (needs MORE than one year)
    rec = led.sell("XYZ", "2024-01-01", 50, 60.0, method="fifo")
    assert rec["lots"].iloc[0]["term"] == "ST"
    # 366 days: long-term
    rec = led.sell("XYZ", "2024-01-02", 50, 60.0, method="fifo")
    assert rec["lots"].iloc[0]["term"] == "LT"
    assert rec["lt_gain"] == pytest.approx(500.0)


def test_split_adjusts_each_lot():
    led = T.LotLedger()
    led.buy("XYZ", "2024-01-15", 10, 100.0)
    led.apply_split("XYZ", "2024-06-10", 2.0)
    assert ledger_position(led) == pytest.approx(20.0)
    rec = led.sell("XYZ", "2024-07-01", 20, 60.0, method="fifo")
    # 20 x (60 - 50) = +$200; basis invariant under the split
    assert rec["realized_gain"] == pytest.approx(200.0)


def ledger_position(led):
    return led.position("XYZ")


def test_wash_sale_disallowed_and_basis_adjusted():
    """IRS Pub. 550: loss sale + repurchase within 30 days -> loss deferred
    into the replacement shares' basis."""
    led = T.LotLedger()
    led.buy("XYZ", "2024-01-01", 100, 60.0)
    led.buy("XYZ", "2024-01-20", 100, 55.0)   # replacement, within 30d
    rec = led.sell("XYZ", "2024-02-01", 100, 50.0, method="fifo")
    assert rec["realized_gain"] == pytest.approx(-1000.0)  # gross
    wash = led.wash_sale_report()
    assert len(wash) == 1
    assert wash.iloc[0]["disallowed_loss"] == pytest.approx(1000.0)
    # replacement lot basis: 55 + 1000/100 = $65/share
    lot = next(l for l in led.lots if l.date == pd.Timestamp("2024-01-20"))
    assert lot.basis_per_share == pytest.approx(65.0)
    # adjusted realized gain nets to zero
    adj = led.realized_gains(adjusted=True)
    assert adj["realized_gain"].sum() == pytest.approx(0.0, abs=1e-9)


def test_wash_sale_defers_does_not_forgive():
    """The deferred loss reappears when the replacement shares are sold:
    lifetime recognized gain equals the true economic gain."""
    led = T.LotLedger()
    led.buy("XYZ", "2024-01-01", 100, 60.0)
    led.buy("XYZ", "2024-01-20", 100, 55.0)
    led.sell("XYZ", "2024-02-01", 100, 50.0, method="fifo")   # -$1,000 gross
    led.sell("XYZ", "2024-06-01", 100, 70.0, method="fifo")   # replacement
    adj = led.realized_gains(adjusted=True)
    # sale 1: -1000 + 1000 deferred = 0; sale 2: 100 x (70-65) = +500
    assert adj.iloc[0]["realized_gain"] == pytest.approx(0.0)
    assert adj.iloc[1]["realized_gain"] == pytest.approx(500.0)
    # true economics: cost 11,500, proceeds 12,000
    assert adj["realized_gain"].sum() == pytest.approx(500.0)


def test_wash_sale_replacement_bought_after_sale():
    """The 30-day window extends after the sale: a later purchase still
    triggers the deferral, and the replay bumps its basis before any
    subsequent sale consumes it."""
    led = T.LotLedger()
    led.buy("XYZ", "2024-01-01", 100, 60.0)
    led.sell("XYZ", "2024-02-01", 100, 50.0, method="fifo")  # -$1,000
    led.buy("XYZ", "2024-02-15", 100, 52.0)                  # replacement
    wash = led.wash_sale_report()
    assert len(wash) == 1
    assert wash.iloc[0]["disallowed_loss"] == pytest.approx(1000.0)
    lot = next(l for l in led.lots if l.date == pd.Timestamp("2024-02-15"))
    assert lot.basis_per_share == pytest.approx(62.0)  # 52 + 1000/100
    led.sell("XYZ", "2024-06-01", 100, 70.0, method="fifo")
    adj = led.realized_gains(adjusted=True)
    # true economics: cost 11,200, proceeds 12,000
    assert adj["realized_gain"].sum() == pytest.approx(800.0)


def test_wash_sale_boundary_day_31():
    led = T.LotLedger()
    led.buy("XYZ", "2024-01-01", 100, 60.0)
    led.buy("XYZ", "2024-03-03", 100, 55.0)   # 31 days after Feb 1 sale
    led.sell("XYZ", "2024-02-01", 100, 50.0, method="fifo")
    assert led.wash_sale_report().empty


def test_wash_sale_purchase_before_sale():
    # the 30-day window extends BEFORE the sale too
    led = T.LotLedger()
    led.buy("XYZ", "2024-01-01", 100, 60.0)
    led.buy("XYZ", "2024-01-25", 100, 58.0)
    led.sell("XYZ", "2024-02-01", 100, 50.0, method="fifo")
    assert len(led.wash_sale_report()) == 1


def test_independent_recomputation():
    """From-scratch lot relief (no module helpers) must match sell()."""
    led = T.LotLedger()
    buys = [("2024-01-10", 50, 40.0), ("2024-02-10", 70, 55.0),
            ("2024-04-10", 30, 45.0)]
    for d, q, p in buys:
        led.buy("XYZ", d, q, p)
    rec = led.sell("XYZ", "2024-08-01", 90, 52.0, method="fifo")
    # independent: consume oldest lots first
    want = 50 * (52 - 40) + 40 * (52 - 55)
    assert rec["realized_gain"] == pytest.approx(want)
    assert ledger_position(led) == pytest.approx(60.0)


def test_sale_plan_does_not_mutate_and_matches():
    led = T.LotLedger()
    led.buy("XYZ", "2024-01-15", 100, 50.0)
    led.buy("XYZ", "2024-03-15", 100, 90.0)
    plan = T.tax_aware_sale_plan(led, {"XYZ": 7000.0}, {"XYZ": 70.0})
    assert ledger_position(led) == pytest.approx(200.0)  # untouched
    assert plan.loc["XYZ", "sell_qty"] == pytest.approx(100.0)
    assert plan.loc["XYZ", "est_realized_gain"] == pytest.approx(-2000.0)
    rec = led.sell("XYZ", "2024-06-15", 100, 70.0, method="min_tax")
    assert rec["realized_gain"] == pytest.approx(
        plan.loc["XYZ", "est_realized_gain"])


def test_harvest_candidates():
    led = T.LotLedger()
    led.buy("XYZ", "2024-01-15", 100, 50.0)
    led.buy("XYZ", "2024-03-15", 100, 90.0)
    led.buy("ABC", "2024-02-01", 50, 20.0)
    cands = led.harvest_candidates({"XYZ": 70.0, "ABC": 25.0})
    assert set(cands["symbol"]) == {"XYZ"}
    assert (cands["unrealized_gain"] < 0).all()
    assert cands.iloc[0]["lot_date"] == pd.Timestamp("2024-03-15")


def test_oversell_rejected(ledger):
    with pytest.raises(ValueError, match="only"):
        ledger.sell("XYZ", "2024-06-15", 500, 60.0, method="fifo")
    with pytest.raises(ValueError, match="method"):
        ledger.sell("XYZ", "2024-06-15", 10, 60.0, method="yolo")
