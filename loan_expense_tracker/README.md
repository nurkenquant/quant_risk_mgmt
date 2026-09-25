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

### Deploy for free: Render + Supabase

Render runs the app on its free plan. Supabase (also free) keeps your data: the
transactions and loans in its Postgres database and the receipt files in its Storage.
Render's free plan has no disk, so nothing is kept on Render itself.

**1. Create the Supabase project** at https://supabase.com (free plan). Pick a strong
database password and save it.

**2. Copy three values from Supabase:**
- `DATABASE_URL`: click **Connect** at the top of the project, then copy the
  **Session pooler** connection string and put your database password in place of
  `[YOUR-PASSWORD]`. Use the pooler, not the "Direct connection": Render can't reach
  the direct address.
- `SUPABASE_URL`: **Project Settings → Data API**, the project URL
  (`https://<project>.supabase.co`).
- `SUPABASE_SERVICE_KEY`: **Project Settings → API Keys**, a **secret** key (or the legacy
  `service_role` key). It gives full access, so only paste it into Render.

The app creates its tables and a private `receipts` storage bucket on first start.

**3. Create the Render service:** in the [Render dashboard](https://dashboard.render.com),
choose **New → Blueprint** and connect this repo on the `main` branch. Render reads
`render.yaml` and asks for:
- `APP_PASSWORD`: the password you'll use to sign in
- the three Supabase values above
- `ANTHROPIC_API_KEY` (optional): reads photo receipts

Click **Apply**. When the deploy is done, open the `https://….onrender.com` URL on your
phone, sign in, and choose **Add to Home Screen**. On Render the app refuses to start if
the password or any Supabase setting is missing, so your data is never exposed or lost.

**What "free" means:**
- The Render service sleeps after about 15 minutes without visits; the next visit takes
  up to a minute to load.
- Supabase pauses a free project after a week with no activity. Using the app keeps it
  awake; if it does pause, un-pause it from the Supabase dashboard (data is kept).
- Free limits (500 MB database, 1 GB file storage) are far more than personal use needs.

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
| `DATA_DIR` | `./data` | Where the database and receipts are stored when Supabase isn't set |
| `DATABASE_URL` | *(none)* | Postgres connection string (Supabase); replaces the local SQLite file |
| `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` | *(none)* | Store receipts in a private Supabase Storage bucket |
| `SUPABASE_BUCKET` | `receipts` | Name of that bucket |
| `APP_PASSWORD` | *(none)* | Requires a password to sign in |
| `ANTHROPIC_API_KEY` | *(none)* | Turns on reading receipts with Claude |
| `RECEIPT_MODEL` | `claude-opus-5` | Claude model used to read receipts |
| `SECRET_KEY` | auto-generated | Session signing key |
| `PORT` | `8000` | HTTP port |

## Tests

```bash
python -m pytest -q tests
# also run the app tests against Postgres:
TEST_DATABASE_URL=postgresql://user:pass@localhost/test_db python -m pytest -q tests
```

## Layout

```
app.py        routes, dashboard totals, cash-flow projection, receipt → transaction logic
loans.py      amortization schedule and summaries
receipts.py   receipt reading (Claude vision, or PDF text/OCR with heuristics)
db.py         database schema and helpers (SQLite or Postgres)
storage.py    receipt files (local folder or Supabase Storage)
templates/    Jinja pages (mobile-first)
static/       CSS, JS (photo downscaling, bottom sheet, chart readouts), PWA manifest
```
