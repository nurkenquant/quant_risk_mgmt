"""Receipt reading: turn an uploaded photo/PDF into transaction fields.

Backends, tried in order:
  1. Claude vision (when ANTHROPIC_API_KEY is set) - reads any photo or PDF.
  2. Plain text + heuristics - text PDFs via pypdf, photos via Tesseract OCR
     when `pytesseract` and the `tesseract` binary are installed.
If nothing can be read the receipt is still attached and the user fills in
the amount by hand.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from datetime import date, datetime

log = logging.getLogger(__name__)

EXPENSE_CATEGORIES = [
    "Groceries", "Dining", "Transport", "Fuel", "Utilities", "Rent", "Shopping",
    "Health", "Entertainment", "Travel", "Education", "Business", "Loan payment",
    "Fees", "Other",
]
INCOME_CATEGORIES = ["Revenue", "Salary", "Sales", "Interest", "Refund", "Other income"]

IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
PDF_TYPE = "application/pdf"

MODEL = os.environ.get("RECEIPT_MODEL", "claude-opus-5")

_SCHEMA = {
    "type": "object",
    "properties": {
        "merchant": {"type": "string", "description": "Store / payee name"},
        "date": {"type": "string", "description": "Purchase date as YYYY-MM-DD, empty if unreadable"},
        "total": {"type": "number", "description": "Final amount paid, including tax and tip"},
        "currency": {"type": "string", "description": "ISO currency code if visible, else empty"},
        "kind": {"type": "string", "enum": ["expense", "income"]},
        "category": {"type": "string", "enum": EXPENSE_CATEGORIES + INCOME_CATEGORIES},
        "summary": {"type": "string", "description": "Short note, e.g. main items bought"},
        "loan_name": {"type": "string", "description": "Name of the matching loan if this is a loan payment, else empty"},
    },
    "required": ["merchant", "date", "total", "currency", "kind", "category", "summary", "loan_name"],
    "additionalProperties": False,
}


def extract(data: bytes, mime: str, loan_names: list[str] | None = None) -> dict | None:
    """Best-effort structured read of a receipt. Returns None if unreadable."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _extract_claude(data, mime, loan_names or [])
        except Exception:  # network/API issues must never block saving
            log.exception("Claude receipt extraction failed; falling back")
    text = _read_text(data, mime)
    return parse_text(text) if text else None


def _extract_claude(raw: bytes, mime: str, loan_names: list[str]) -> dict | None:
    import anthropic

    data = base64.standard_b64encode(raw).decode()
    if mime == PDF_TYPE:
        block = {"type": "document", "source": {"type": "base64", "media_type": mime, "data": data}}
    elif mime in IMAGE_TYPES:
        block = {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}
    else:
        return None

    loans = ", ".join(loan_names) if loan_names else "none"
    prompt = (
        "Read this receipt, invoice or payment confirmation and extract the transaction. "
        f"Today is {date.today().isoformat()}. "
        "Use kind=income only if the document shows money received by me (a sale, invoice paid to me, "
        "salary slip); otherwise expense. "
        f"My loans are: {loans}. If this document is a payment toward one of them, set loan_name "
        "to that exact name and category to 'Loan payment'; otherwise leave loan_name empty."
    )
    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": _SCHEMA}},
        messages=[{"role": "user", "content": [block, {"type": "text", "text": prompt}]}],
    )
    if resp.stop_reason == "refusal":
        return None
    text = next((b.text for b in resp.content if b.type == "text"), "")
    out = json.loads(text)
    out["date"] = _valid_date(out.get("date"))
    out["source"] = "claude"
    return out


def _read_text(data: bytes, mime: str) -> str:
    try:
        if mime == PDF_TYPE:
            from pypdf import PdfReader
            return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
        if mime in IMAGE_TYPES:
            import pytesseract
            from PIL import Image
            return pytesseract.image_to_string(Image.open(io.BytesIO(data)))
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # optional deps missing/broken (pyo3 panics aren't Exceptions)
        log.info("No local text extraction for %s receipt: %r", mime, exc)
    return ""


# ---------------------------------------------------------------- heuristics

_AMOUNT = r"(\d{1,3}(?:[ ,.\u00a0]\d{3})+(?:[.,]\d{2})?|\d+(?:[.,]\d{2})?)"
_TOTAL_WORDS = re.compile(
    r"(grand\s*total|total\s*due|amount\s*due|amount\s*paid|balance\s*due|total|"
    r"итого|к\s*оплате|всего|сумма|jami|барлығы|төлеуге)", re.I)
_KEYWORDS = {
    "Groceries": ["market", "grocery", "supermarket", "magnum", "walmart", "aldi", "lidl", "продукт"],
    "Dining": ["cafe", "coffee", "restaurant", "pizza", "burger", "starbucks", "кафе", "ресторан"],
    "Fuel": ["fuel", "petrol", "gas station", "shell", "helios", "азс", "бензин"],
    "Transport": ["taxi", "uber", "yandex go", "bolt", "metro", "bus", "такси"],
    "Utilities": ["electric", "water", "internet", "mobile", "kcell", "beeline", "tele2", "коммунал"],
    "Health": ["pharmacy", "apteka", "аптека", "clinic", "hospital"],
    "Loan payment": ["loan", "credit", "кредит", "installment", "погашение"],
}


def _to_float(s: str) -> float | None:
    s = s.replace(" ", " ").strip()
    # Treat the last separator followed by exactly two digits as the decimal point.
    m = re.match(r"^(.*?)[.,](\d{2})$", s)
    whole, cents = (m.group(1), m.group(2)) if m else (s, "00")
    whole = re.sub(r"[ ,.]", "", whole)
    try:
        return float(f"{whole}.{cents}")
    except ValueError:
        return None


def _valid_date(s) -> str:
    try:
        return datetime.strptime(str(s), "%Y-%m-%d").date().isoformat()
    except (TypeError, ValueError):
        return ""


def _find_date(text: str) -> str:
    patterns = [
        (r"\b(\d{4})-(\d{2})-(\d{2})\b", lambda m: (m[1], m[2], m[3])),
        (r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b", lambda m: (m[3], m[2], m[1])),
        (r"\b(\d{1,2})[./](\d{1,2})[./](\d{2})\b", lambda m: ("20" + m[3], m[2], m[1])),
    ]
    for pat, parts in patterns:
        for m in re.finditer(pat, text):
            y, mo, d = parts(m)
            for a, b in ((mo, d), (d, mo)):  # try day-first, then month-first
                try:
                    return date(int(y), int(a), int(b)).isoformat()
                except ValueError:
                    continue
    return ""


def parse_text(text: str) -> dict | None:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    total = None
    for ln in reversed(lines):  # totals sit at the bottom; prefer the last one
        if _TOTAL_WORDS.search(ln) and not re.search(r"sub\s*total|tax|vat|ндс", ln, re.I):
            nums = [_to_float(x) for x in re.findall(_AMOUNT, ln)]
            nums = [n for n in nums if n]
            if nums:
                total = nums[-1]
                break
    if total is None:
        nums = [_to_float(x) for x in re.findall(_AMOUNT, text)]
        nums = [n for n in nums if n and n < 10_000_000]
        total = max(nums) if nums else None
    merchant = next((ln for ln in lines if re.search(r"[A-Za-zА-Яа-я]{3}", ln)), "")[:60]
    low = text.lower()
    category = next((c for c, words in _KEYWORDS.items() if any(w in low for w in words)), "Other")
    return {
        "merchant": merchant,
        "date": _find_date(text),
        "total": total or 0.0,
        "currency": "",
        "kind": "expense",
        "category": category,
        "summary": "",
        "loan_name": "",
        "source": "ocr",
    }
