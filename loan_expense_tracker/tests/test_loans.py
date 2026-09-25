from datetime import date

import pytest

import loans as L


def test_annuity_payment_matches_textbook():
    # 100k over 30 years at 6%: the classic 599.55 mortgage payment
    assert L.annuity_payment(100_000, 6, 360) == pytest.approx(599.55, abs=0.01)


def test_zero_rate_splits_evenly():
    rows = L.schedule(1200, 0, 12, date(2026, 1, 15))
    assert len(rows) == 12
    assert all(r.payment == 100 and r.interest == 0 for r in rows)
    assert rows[-1].balance == 0


def test_schedule_amortizes_to_zero_and_dates_roll_monthly():
    rows = L.schedule(10_000, 12, 24, date(2026, 1, 31))
    assert len(rows) == 24
    assert rows[-1].balance == 0
    assert rows[1].due == date(2026, 2, 28)   # clamped to month end
    assert rows[2].due == date(2026, 3, 31)   # and back to the 31st
    assert sum(r.principal for r in rows) == pytest.approx(10_000, abs=0.05)


def test_extra_payment_shortens_loan_and_cuts_interest():
    base = L.schedule(20_000, 10, 60, date(2026, 1, 1))
    fast = L.schedule(20_000, 10, 60, date(2026, 1, 1), extra=200)
    assert len(fast) < len(base)
    assert sum(r.interest for r in fast) < sum(r.interest for r in base)
    assert fast[-1].balance == 0


def test_payment_below_interest_does_not_loop_forever():
    rows = L.schedule(100_000, 24, 12, date(2026, 1, 1), payment=100)
    assert rows == []


def test_summarize_tracks_paid_periods():
    rows = L.schedule(1200, 0, 12, date(2026, 1, 1))
    s = L.summarize(rows, {1, 2, 3})
    assert s["paid_count"] == 3
    assert s["balance"] == 900
    assert s["next"].period == 4
    assert s["remaining"] == 900
    assert L.summarize(rows, set())["balance"] == 1200
