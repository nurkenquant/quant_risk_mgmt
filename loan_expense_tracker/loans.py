"""Loan amortization: projects every scheduled payment of a loan."""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date


@dataclass
class Installment:
    period: int
    due: date
    payment: float
    principal: float
    interest: float
    extra: float
    balance: float


def add_months(d: date, months: int, day: int | None = None) -> date:
    """Shift `d` by `months`, pinning to `day` (clamped to the month length)."""
    m = d.month - 1 + months
    y = d.year + m // 12
    m = m % 12 + 1
    target = day or d.day
    return date(y, m, min(target, calendar.monthrange(y, m)[1]))


def annuity_payment(principal: float, annual_rate_pct: float, months: int) -> float:
    """Fixed monthly payment that repays `principal` over `months`."""
    if months <= 0:
        return principal
    r = annual_rate_pct / 100 / 12
    if r == 0:
        return principal / months
    return principal * r / (1 - (1 + r) ** -months)


def schedule(principal: float, annual_rate_pct: float, term_months: int,
             first_due: date, payment: float | None = None,
             extra: float = 0.0) -> list[Installment]:
    """Full amortization table.

    `payment` overrides the computed annuity payment (e.g. the exact figure
    printed on the bank statement). `extra` is added to principal every month.
    """
    r = annual_rate_pct / 100 / 12
    pmt = payment if payment else annuity_payment(principal, annual_rate_pct, term_months)
    bal = principal
    rows: list[Installment] = []
    period = 0
    # Cap iterations so a payment below the monthly interest can't loop forever.
    while bal > 0.005 and period < max(term_months, 1) * 3 + 600:
        period += 1
        interest = bal * r
        principal_part = min(pmt - interest, bal)
        if principal_part <= 0 and extra <= 0:
            # Payment doesn't even cover interest: loan never amortizes.
            break
        extra_part = min(extra, bal - principal_part) if principal_part < bal else 0.0
        bal -= principal_part + extra_part
        rows.append(Installment(
            period=period,
            due=add_months(first_due, period - 1, first_due.day),
            payment=round(principal_part + interest, 2),
            principal=round(principal_part, 2),
            interest=round(interest, 2),
            extra=round(extra_part, 2),
            balance=round(max(bal, 0.0), 2),
        ))
    return rows


def summarize(rows: list[Installment], paid_periods: set[int]) -> dict:
    """Totals and next-due info for a schedule given which periods are paid."""
    unpaid = [r for r in rows if r.period not in paid_periods]
    paid = [r for r in rows if r.period in paid_periods]
    last_paid_balance = min((r.balance for r in paid), default=None)
    principal = rows[0].balance + rows[0].principal + rows[0].extra if rows else 0.0
    return {
        "total_interest": round(sum(r.interest for r in rows), 2),
        "total_paid": round(sum(r.payment + r.extra for r in paid), 2),
        "remaining": round(sum(r.payment + r.extra for r in unpaid), 2),
        "balance": round(last_paid_balance if last_paid_balance is not None else principal, 2),
        "payoff": rows[-1].due if rows else None,
        "next": unpaid[0] if unpaid else None,
        "paid_count": len(paid),
        "count": len(rows),
        "progress": round(100 * len(paid) / len(rows)) if rows else 100,
    }
