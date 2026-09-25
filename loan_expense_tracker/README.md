# Loan & Expense Tracker

A small, mobile-first web app for tracking day-to-day spending and revenue alongside
your loans. It works in any phone browser and can be added to the home screen like an app.

## What it does

- **Loan projection.** Add a loan (amount, rate, term, first payment date) and the app
  builds the full schedule: every payment date, the principal/interest split, the balance
  left, total interest and payoff date. You can type your bank's exact monthly payment and
  an optional extra monthly payment. Tap ✓ on a payment to mark it paid, which also records
  it as an expense.
- **Daily tracking.** Add expenses and revenue in a couple of taps. The home screen shows
  today's net, month totals, upcoming loan payments (with overdue ones highlighted), the
  last 14 days, and a 12-month cash-flow projection (scheduled loan payments plus your
  average revenue and spending over the last 3 months).
- **Receipts that apply themselves.** Tap **+ → Scan receipt** and take a photo or pick a
  PDF. The app reads it and fills in the amount, date, merchant and category for you. If
  the receipt is a loan payment (the lender or amount matches), it marks that loan's next
  installment as paid. You can also attach a receipt when adding or editing an entry;
  only the fields you left blank get filled in.
- CSV export, a currency symbol setting (₸, $, €, …), dark mode and an optional password.

### How receipts are read

| Setup | Works on |
|---|---|
| `ANTHROPIC_API_KEY` set (recommended) | Any photo or PDF, any language. Uses Claude vision (`claude-opus-5` by default; change it with `RECEIPT_MODEL`) |
| No key | Text-based PDFs (e-receipts, bank confirmations) via `pypdf`. Photos too if you install `pytesseract` + `pillow` and the `tesseract` binary |

If a receipt can't be read, it's still attached and you're taken to the entry so you
can type in the amount. Photos are shrunk in the browser before upload, so uploads stay
quick on mobile data.

## Run it

```bash
cd loan_expense_tracker
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # optional, for reading photo receipts
export APP_PASSWORD=choose-one        # recommended if others can reach the server
python app.py                         # http://localhost:8000
```

Open it on your phone at `http://<your-computer-ip>:8000` (same Wi-Fi), or deploy it.

### Deploy (Docker)

```bash
docker build -t ledger .
docker run -d -p 8000:8000 -v ledger-data:/data \
  -e APP_PASSWORD=choose-one -e ANTHROPIC_API_KEY=sk-ant-... ledger
```

Any host that runs a Docker image or a Python web process works (Fly.io, Render, Railway,
a VPS). Give it a persistent volume at `/data`, which holds the SQLite database and
the receipt files. Put it behind HTTPS and set `APP_PASSWORD`.

| Variable | Default | Purpose |
|---|---|---|
| `DATA_DIR` | `./data` | Where the database and receipts are stored |
| `APP_PASSWORD` | *(none)* | Requires a password to sign in |
| `ANTHROPIC_API_KEY` | *(none)* | Turns on reading receipts with Claude |
| `RECEIPT_MODEL` | `claude-opus-5` | Claude model used to read receipts |
| `SECRET_KEY` | auto-generated | Session signing key |
| `PORT` | `8000` | HTTP port |

## Tests

```bash
python -m pytest -q tests
```

## Layout

```
app.py        routes, dashboard totals, cash-flow projection, receipt → transaction logic
loans.py      amortization schedule and summaries
receipts.py   receipt reading (Claude vision, or PDF text/OCR with heuristics)
db.py         SQLite schema and helpers
templates/    Jinja pages (mobile-first)
static/       CSS, JS (photo downscaling, bottom sheet, chart readouts), PWA manifest
```
