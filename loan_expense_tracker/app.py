"""Loan & expense tracker: a small, mobile-first personal finance web app."""
from __future__ import annotations

import csv
import io
import os
import secrets
import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta
from functools import wraps

from flask import (Flask, Response, abort, flash, redirect, render_template, request,
                   send_from_directory, session, url_for)

import db
import loans as L
import receipts
import storage

HERE = os.path.dirname(os.path.abspath(__file__))
ALLOWED_MIME = receipts.IMAGE_TYPES | {receipts.PDF_TYPE}
EXT_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".gif": "image/gif", ".webp": "image/webp", ".pdf": "application/pdf"}


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__)
    data_dir = os.environ.get("DATA_DIR", os.path.join(HERE, "data"))
    app.config.update(
        # Postgres (e.g. Supabase) when DATABASE_URL is set, else a local SQLite file.
        DATABASE=os.environ.get("DATABASE_URL") or os.path.join(data_dir, "tracker.db"),
        UPLOAD_DIR=os.path.join(data_dir, "receipts"),
        SUPABASE_URL=os.environ.get("SUPABASE_URL", ""),
        SUPABASE_SERVICE_KEY=os.environ.get("SUPABASE_SERVICE_KEY", ""),
        SUPABASE_BUCKET=os.environ.get("SUPABASE_BUCKET", "receipts"),
        SECRET_KEY=os.environ.get("SECRET_KEY") or _persistent_secret(data_dir),
        APP_PASSWORD=os.environ.get("APP_PASSWORD", ""),
        MAX_CONTENT_LENGTH=20 * 1024 * 1024,
    )
    if config:
        app.config.update(config)
    if os.environ.get("RENDER") and not app.config.get("TESTING"):
        check_hosted_config(app.config)
    app.extensions["receipts"] = storage.from_config(app.config)
    db.init_db(app.config["DATABASE"])
    app.teardown_appcontext(db.close_db)
    register(app)
    return app


def check_hosted_config(cfg):
    """On a host without a persistent disk, refuse to start in a way that
    would expose the data or silently lose it on the next restart."""
    missing = []
    if not cfg["APP_PASSWORD"]:
        missing.append("APP_PASSWORD (without it anyone with the URL can see your finances)")
    if not db.is_pg(cfg["DATABASE"]):
        missing.append("DATABASE_URL (your Supabase Postgres connection string)")
    if not (cfg["SUPABASE_URL"] and cfg["SUPABASE_SERVICE_KEY"]):
        missing.append("SUPABASE_URL and SUPABASE_SERVICE_KEY (for storing receipts)")
    if missing:
        raise RuntimeError("Missing settings: " + "; ".join(missing))


def _persistent_secret(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, ".secret_key")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
    with open(path) as f:
        return f.read().strip()


# ------------------------------------------------------------------ helpers

def parse_day(s: str | None, default: date | None = None) -> date:
    try:
        return datetime.strptime(s or "", "%Y-%m-%d").date()
    except ValueError:
        return default or date.today()


def parse_amount(s) -> float | None:
    if s is None or str(s).strip() == "":
        return None
    try:
        return round(float(str(s).replace(" ", "").replace(",", ".")), 2)
    except ValueError:
        return None


def loan_rows():
    return db.get_db().execute("SELECT * FROM loans ORDER BY name").fetchall()


def paid_periods(loan_id: int) -> set[int]:
    rows = db.get_db().execute(
        "SELECT loan_period FROM transactions WHERE loan_id = ?", (loan_id,)).fetchall()
    return {r["loan_period"] for r in rows}


def loan_schedule(loan) -> list[L.Installment]:
    return L.schedule(loan["principal"], loan["rate"], loan["term_months"],
                      parse_day(loan["first_due"]), loan["payment"], loan["extra"] or 0)


def loans_overview():
    out = []
    for loan in loan_rows():
        rows = loan_schedule(loan)
        out.append({"loan": loan, "rows": rows,
                    "summary": L.summarize(rows, paid_periods(loan["id"]))})
    return out


def save_upload(file) -> tuple[str, str, bytes] | None:
    """Store an uploaded receipt; returns (filename, mime, bytes) or None."""
    if not file or not file.filename:
        return None
    ext = os.path.splitext(file.filename)[1].lower()
    mime = file.mimetype if file.mimetype in ALLOWED_MIME else EXT_MIME.get(ext)
    if mime not in ALLOWED_MIME:
        flash("Receipt must be a photo (JPG, PNG, WEBP) or a PDF.", "error")
        return None
    ext = ext if ext in EXT_MIME else {v: k for k, v in EXT_MIME.items()}[mime]
    name = f"{date.today():%Y%m%d}-{uuid.uuid4().hex[:10]}{ext}"
    data = file.read()
    receipt_store().save(name, data, mime)
    return name, mime, data


def receipt_store():
    from flask import current_app
    return current_app.extensions["receipts"]


def match_loan(fields: dict, overview: list[dict]):
    """Find the loan a receipt pays: by name, by lender, then by amount."""
    name = (fields.get("loan_name") or "").strip().lower()
    merchant = (fields.get("merchant") or "").lower()
    total = fields.get("total") or 0
    candidates = [o for o in overview if o["summary"]["next"]]
    for o in candidates:
        if name and name == o["loan"]["name"].lower():
            return o
    for o in candidates:
        lender = (o["loan"]["lender"] or "").lower()
        if lender and len(lender) >= 3 and (lender in merchant or (merchant and merchant in lender)):
            return o
    for o in candidates:
        nxt = o["summary"]["next"]
        due = nxt.payment + nxt.extra
        if total and abs(total - due) <= max(0.01 * due, 0.5):
            return o
    return None


def apply_receipt(tx_id: int, data: bytes, mime: str) -> dict | None:
    """Read a receipt and fill the transaction with what it says.

    Only fields the user left blank (amount 0, default category, empty
    merchant/note) are overwritten. A receipt that matches a loan's next
    installment marks that installment paid.
    """
    conn = db.get_db()
    overview = loans_overview()
    fields = receipts.extract(data, mime, [o["loan"]["name"] for o in overview])
    if not fields:
        return None
    tx = conn.execute("SELECT * FROM transactions WHERE id = ?", (tx_id,)).fetchone()
    updates = {"source": fields.get("source", "ocr")}
    if not tx["amount"] and fields.get("total"):
        updates["amount"] = round(float(fields["total"]), 2)
    if not tx["merchant"] and fields.get("merchant"):
        updates["merchant"] = fields["merchant"][:80]
    if not tx["note"] and fields.get("summary"):
        updates["note"] = fields["summary"][:200]
    if tx["category"] in ("Other", "Other income") and fields.get("category"):
        updates["category"] = fields["category"]
        if fields.get("kind") in ("expense", "income"):
            updates["kind"] = fields["kind"]
    if fields.get("date") and tx["day"] == date.today().isoformat():
        updates["day"] = fields["date"]

    if tx["loan_id"] is None and updates.get("kind", tx["kind"]) == "expense":
        merged = dict(fields, total=updates.get("amount", tx["amount"]))
        o = match_loan(merged, overview)
        if o:
            updates.update(loan_id=o["loan"]["id"], loan_period=o["summary"]["next"].period,
                           category="Loan payment")
            fields["matched_loan"] = o["loan"]["name"]

    sets = ", ".join(f"{k} = ?" for k in updates)
    conn.execute(f"UPDATE transactions SET {sets} WHERE id = ?", (*updates.values(), tx_id))
    conn.commit()
    return fields


def month_bounds(d: date) -> tuple[date, date]:
    start = d.replace(day=1)
    return start, L.add_months(start, 1, 1)


def totals(start: date, end: date) -> dict:
    rows = db.get_db().execute(
        "SELECT kind, loan_id IS NOT NULL AS is_loan, SUM(amount) s FROM transactions "
        "WHERE day >= ? AND day < ? GROUP BY kind, is_loan",
        (start.isoformat(), end.isoformat())).fetchall()
    t = {"income": 0.0, "expense": 0.0, "loan": 0.0}
    for r in rows:
        if r["kind"] == "income":
            t["income"] += r["s"]
        elif r["is_loan"]:
            t["loan"] += r["s"]
        else:
            t["expense"] += r["s"]
    t["net"] = t["income"] - t["expense"] - t["loan"]
    return t


def projection(overview: list[dict], months: int = 12) -> list[dict]:
    """Month-by-month cash flow: scheduled loan payments + recent averages."""
    today = date.today()
    hist_start = L.add_months(today.replace(day=1), -3, 1)
    hist = totals(hist_start, today.replace(day=1))
    # Fall back to the current month when there is no full-month history yet.
    n = 3
    if hist["income"] == 0 and hist["expense"] == 0:
        hist, n = totals(*month_bounds(today)), 1
    avg_in, avg_out = hist["income"] / n, hist["expense"] / n

    due = defaultdict(float)
    for o in overview:
        paid = paid_periods(o["loan"]["id"])
        for r in o["rows"]:
            if r.period not in paid:
                key = max(r.due, today.replace(day=1)).strftime("%Y-%m")
                due[key] += r.payment + r.extra
    out = []
    for i in range(months):
        m = L.add_months(today.replace(day=1), i, 1)
        key = m.strftime("%Y-%m")
        loan_amt = round(due.get(key, 0.0), 2)
        out.append({"key": key, "label": m.strftime("%b"), "year": m.year,
                    "income": round(avg_in, 2), "expense": round(avg_out, 2),
                    "loan": loan_amt, "net": round(avg_in - avg_out - loan_amt, 2)})
    peak = max([max(p["income"], p["expense"] + p["loan"]) for p in out] + [1])
    for p in out:
        p["h_in"] = round(100 * p["income"] / peak, 1)
        p["h_exp"] = round(100 * p["expense"] / peak, 1)
        p["h_loan"] = round(100 * p["loan"] / peak, 1)
    return out


def daily_series(days: int = 14) -> list[dict]:
    today = date.today()
    start = today - timedelta(days=days - 1)
    rows = db.get_db().execute(
        "SELECT day, kind, SUM(amount) s FROM transactions WHERE day >= ? AND day <= ? "
        "GROUP BY day, kind", (start.isoformat(), today.isoformat())).fetchall()
    agg = defaultdict(lambda: {"income": 0.0, "expense": 0.0})
    for r in rows:
        agg[r["day"]][r["kind"]] += r["s"]
    out = []
    for i in range(days):
        d = start + timedelta(days=i)
        v = agg[d.isoformat()]
        out.append({"day": d, "income": v["income"], "expense": v["expense"]})
    peak = max([max(p["income"], p["expense"]) for p in out] + [1])
    for p in out:
        p["h_in"] = round(100 * p["income"] / peak, 1)
        p["h_exp"] = round(100 * p["expense"] / peak, 1)
    return out


# ------------------------------------------------------------------- routes

def register(app: Flask):

    @app.context_processor
    def inject():
        cur = db.get_setting("currency", os.environ.get("CURRENCY", "$"))

        def money(v, signed=False):
            v = v or 0
            sign = ("+" if v > 0 else "−" if v < 0 else "") if signed else ("−" if v < 0 else "")
            return f"{sign}{cur}{abs(v):,.2f}"
        return {"money": money, "currency": cur, "today": date.today(),
                "EXPENSE_CATEGORIES": receipts.EXPENSE_CATEGORIES,
                "INCOME_CATEGORIES": receipts.INCOME_CATEGORIES,
                "ai_enabled": bool(os.environ.get("ANTHROPIC_API_KEY"))}

    def login_required(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            if app.config["APP_PASSWORD"] and not session.get("ok"):
                return redirect(url_for("login", next=request.path))
            return fn(*a, **kw)
        return wrapper

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            if secrets.compare_digest(request.form.get("password", ""), app.config["APP_PASSWORD"]):
                session["ok"] = True
                session.permanent = True
                nxt = request.args.get("next", "/")
                return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else "/")
            flash("Wrong password.", "error")
        return render_template("login.html")

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    # ---- dashboard
    @app.route("/")
    @login_required
    def dashboard():
        today = date.today()
        overview = loans_overview()
        upcoming = sorted(
            [(o["loan"], o["summary"]["next"]) for o in overview if o["summary"]["next"]],
            key=lambda x: x[1].due)
        recent = db.get_db().execute(
            "SELECT * FROM transactions ORDER BY day DESC, id DESC LIMIT 8").fetchall()
        return render_template(
            "dashboard.html",
            day=totals(today, today + timedelta(days=1)),
            month=totals(*month_bounds(today)),
            debt=sum(o["summary"]["balance"] for o in overview),
            upcoming=upcoming, recent=recent,
            daily=daily_series(), proj=projection(overview))

    # ---- transactions
    @app.route("/transactions")
    @login_required
    def transactions():
        month = request.args.get("month") or date.today().strftime("%Y-%m")
        kind = request.args.get("kind", "")
        start = parse_day(month + "-01")
        _, end = month_bounds(start)
        q = "SELECT t.*, l.name AS loan_name FROM transactions t LEFT JOIN loans l ON l.id = t.loan_id " \
            "WHERE day >= ? AND day < ?"
        args = [start.isoformat(), end.isoformat()]
        if kind in ("income", "expense"):
            q += " AND kind = ?"
            args.append(kind)
        rows = db.get_db().execute(q + " ORDER BY day DESC, t.id DESC", args).fetchall()
        groups = defaultdict(list)
        for r in rows:
            groups[r["day"]].append(r)
        days = [{"day": parse_day(d), "entries": items,
                 "net": sum(i["amount"] if i["kind"] == "income" else -i["amount"] for i in items)}
                for d, items in groups.items()]
        return render_template(
            "transactions.html", days=days, month=month, kind=kind, tot=totals(start, end),
            prev=L.add_months(start, -1, 1).strftime("%Y-%m"),
            next=L.add_months(start, 1, 1).strftime("%Y-%m"),
            month_label=start.strftime("%B %Y"))

    def tx_from_form(form) -> dict:
        kind = form.get("kind") if form.get("kind") in ("income", "expense") else "expense"
        cats = receipts.INCOME_CATEGORIES if kind == "income" else receipts.EXPENSE_CATEGORIES
        cat = form.get("category")
        return {"kind": kind,
                "amount": parse_amount(form.get("amount")) or 0.0,
                "category": cat if cat in cats else cats[-1],
                "day": parse_day(form.get("day")).isoformat(),
                "merchant": form.get("merchant", "").strip()[:80],
                "note": form.get("note", "").strip()[:200]}

    @app.route("/add", methods=["GET", "POST"])
    @login_required
    def add():
        if request.method == "POST":
            data = tx_from_form(request.form)
            up = save_upload(request.files.get("receipt"))
            if not data["amount"] and not up:
                flash("Enter an amount or attach a receipt.", "error")
                return render_template("tx_form.html", tx=data, mode="add")
            tx_id = db.insert(
                "INSERT INTO transactions(kind, amount, category, day, merchant, note, receipt) "
                "VALUES(:kind, :amount, :category, :day, :merchant, :note, :receipt)",
                dict(data, receipt=up[0] if up else None))
            db.get_db().commit()
            if up:
                return finish_receipt(tx_id, up)
            flash(f"Saved {data['kind']} of {data['amount']:,.2f}.", "ok")
            return redirect(url_for("dashboard"))
        kind = request.args.get("kind", "expense")
        return render_template("tx_form.html", mode="add", tx={
            "kind": kind, "day": date.today().isoformat(),
            "category": "Revenue" if kind == "income" else ""})

    def finish_receipt(tx_id: int, up: tuple[str, str, bytes]):
        fields = apply_receipt(tx_id, up[2], up[1])
        tx = db.get_db().execute("SELECT * FROM transactions WHERE id = ?", (tx_id,)).fetchone()
        if not fields:
            flash("Receipt attached. Couldn't read it automatically — please check the amount.", "warn")
            return redirect(url_for("edit_tx", tx_id=tx_id))
        msg = f"Receipt applied: {tx['merchant'] or tx['category']} · {tx['amount']:,.2f}"
        if fields.get("matched_loan"):
            msg += f" · marked as payment for “{fields['matched_loan']}”"
        flash(msg, "ok")
        if not tx["amount"]:
            return redirect(url_for("edit_tx", tx_id=tx_id))
        return redirect(url_for("dashboard"))

    @app.route("/scan", methods=["POST"])
    @login_required
    def scan():
        """One-tap receipt: create an expense and fill it from the receipt."""
        up = save_upload(request.files.get("receipt"))
        if not up:
            if not request.files.get("receipt") or not request.files["receipt"].filename:
                flash("No receipt selected.", "error")
            return redirect(request.referrer or url_for("dashboard"))
        tx_id = db.insert(
            "INSERT INTO transactions(kind, amount, category, day, receipt) VALUES('expense', 0, 'Other', ?, ?)",
            (date.today().isoformat(), up[0]))
        db.get_db().commit()
        return finish_receipt(tx_id, up)

    @app.route("/tx/<int:tx_id>", methods=["GET", "POST"])
    @login_required
    def edit_tx(tx_id):
        conn = db.get_db()
        tx = conn.execute("SELECT t.*, l.name AS loan_name FROM transactions t "
                          "LEFT JOIN loans l ON l.id = t.loan_id WHERE t.id = ?", (tx_id,)).fetchone()
        if not tx:
            abort(404)
        if request.method == "POST":
            data = tx_from_form(request.form)
            up = save_upload(request.files.get("receipt"))
            conn.execute("UPDATE transactions SET kind=:kind, amount=:amount, category=:category, "
                         "day=:day, merchant=:merchant, note=:note WHERE id=:id", dict(data, id=tx_id))
            if up:
                conn.execute("UPDATE transactions SET receipt = ? WHERE id = ?", (up[0], tx_id))
            conn.commit()
            if up:
                if tx["receipt"]:
                    receipt_store().delete(tx["receipt"])
                return finish_receipt(tx_id, up)
            flash("Updated.", "ok")
            return redirect(url_for("transactions", month=data["day"][:7]))
        return render_template("tx_form.html", tx=tx, mode="edit")

    @app.route("/tx/<int:tx_id>/delete", methods=["POST"])
    @login_required
    def delete_tx(tx_id):
        conn = db.get_db()
        tx = conn.execute("SELECT receipt FROM transactions WHERE id = ?", (tx_id,)).fetchone()
        conn.execute("DELETE FROM transactions WHERE id = ?", (tx_id,))
        conn.commit()
        if tx and tx["receipt"]:
            receipt_store().delete(tx["receipt"])
        flash("Deleted.", "ok")
        return redirect(request.form.get("next") or url_for("transactions"))

    @app.route("/receipts/<path:name>")
    @login_required
    def receipt_file(name):
        try:
            data = receipt_store().load(name)
        except FileNotFoundError:
            abort(404)
        mime = EXT_MIME.get(os.path.splitext(name)[1].lower(), "application/octet-stream")
        return Response(data, mimetype=mime, headers={"Cache-Control": "private, max-age=86400"})

    # ---- loans
    @app.route("/loans")
    @login_required
    def loans():
        return render_template("loans.html", overview=loans_overview())

    def loan_from_form(form) -> dict | None:
        principal = parse_amount(form.get("principal"))
        rate = parse_amount(form.get("rate"))
        try:
            term = int(form.get("term_months") or 0)
        except ValueError:
            term = 0
        if not principal or principal <= 0 or rate is None or rate < 0 or term <= 0:
            return None
        return {"name": form.get("name", "").strip()[:60] or "Loan",
                "lender": form.get("lender", "").strip()[:60],
                "principal": principal, "rate": rate, "term_months": term,
                "first_due": parse_day(form.get("first_due")).isoformat(),
                "payment": parse_amount(form.get("payment")),
                "extra": parse_amount(form.get("extra")) or 0.0}

    @app.route("/loans/new", methods=["GET", "POST"])
    @login_required
    def new_loan():
        if request.method == "POST":
            data = loan_from_form(request.form)
            if not data:
                flash("Amount, rate and term are required.", "error")
                return render_template("loan_form.html", loan=request.form, mode="add")
            conn = db.get_db()
            loan_id = db.insert(
                "INSERT INTO loans(name, lender, principal, rate, term_months, first_due, payment, extra) "
                "VALUES(:name, :lender, :principal, :rate, :term_months, :first_due, :payment, :extra)", data)
            if request.form.get("mark_past"):
                loan = conn.execute("SELECT * FROM loans WHERE id = ?", (loan_id,)).fetchone()
                for r in loan_schedule(loan):
                    if r.due < date.today():
                        record_payment(loan, r, create_tx=False, commit=False)
            conn.commit()
            flash(f"Loan “{data['name']}” added.", "ok")
            return redirect(url_for("loan_detail", loan_id=loan_id))
        return render_template("loan_form.html", loan={"first_due": L.add_months(date.today(), 1).isoformat()},
                               mode="add")

    @app.route("/loans/<int:loan_id>/edit", methods=["GET", "POST"])
    @login_required
    def edit_loan(loan_id):
        conn = db.get_db()
        loan = conn.execute("SELECT * FROM loans WHERE id = ?", (loan_id,)).fetchone() or abort(404)
        if request.method == "POST":
            data = loan_from_form(request.form)
            if not data:
                flash("Amount, rate and term are required.", "error")
                return render_template("loan_form.html", loan=request.form, mode="edit", loan_id=loan_id)
            conn.execute("UPDATE loans SET name=:name, lender=:lender, principal=:principal, rate=:rate, "
                         "term_months=:term_months, first_due=:first_due, payment=:payment, extra=:extra "
                         "WHERE id=:id", dict(data, id=loan_id))
            conn.commit()
            flash("Loan updated.", "ok")
            return redirect(url_for("loan_detail", loan_id=loan_id))
        return render_template("loan_form.html", loan=loan, mode="edit", loan_id=loan_id)

    @app.route("/loans/<int:loan_id>/delete", methods=["POST"])
    @login_required
    def delete_loan(loan_id):
        conn = db.get_db()
        # Keep real money movements, drop the placeholder "already paid" markers.
        conn.execute("DELETE FROM transactions WHERE loan_id = ? AND source = 'history'", (loan_id,))
        conn.execute("UPDATE transactions SET loan_id = NULL, loan_period = NULL WHERE loan_id = ?", (loan_id,))
        conn.execute("DELETE FROM loans WHERE id = ?", (loan_id,))
        conn.commit()
        flash("Loan deleted.", "ok")
        return redirect(url_for("loans"))

    @app.route("/loans/<int:loan_id>")
    @login_required
    def loan_detail(loan_id):
        loan = db.get_db().execute("SELECT * FROM loans WHERE id = ?", (loan_id,)).fetchone() or abort(404)
        rows = loan_schedule(loan)
        paid = paid_periods(loan_id)
        return render_template("loan_detail.html", loan=loan, rows=rows, paid=paid,
                               s=L.summarize(rows, paid))

    def record_payment(loan, inst: L.Installment, create_tx=True, day: date | None = None, commit=True):
        """Mark an installment paid. Pre-app history is stored with source='history'
        and amount 0 so it doesn't distort spending totals."""
        amount = inst.payment + inst.extra if create_tx else 0.0
        db.get_db().execute(
            "INSERT INTO transactions(kind, amount, category, day, merchant, note, source, loan_id, loan_period) "
            "VALUES('expense', ?, 'Loan payment', ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
            (round(amount, 2), (day or inst.due).isoformat(), loan["lender"] or loan["name"],
             f"{loan['name']} #{inst.period}", "loan" if create_tx else "history",
             loan["id"], inst.period))
        if commit:
            db.get_db().commit()

    @app.route("/loans/<int:loan_id>/pay/<int:period>", methods=["POST"])
    @login_required
    def toggle_payment(loan_id, period):
        conn = db.get_db()
        loan = conn.execute("SELECT * FROM loans WHERE id = ?", (loan_id,)).fetchone() or abort(404)
        existing = conn.execute("SELECT id FROM transactions WHERE loan_id = ? AND loan_period = ?",
                                (loan_id, period)).fetchone()
        if existing:
            conn.execute("DELETE FROM transactions WHERE id = ?", (existing["id"],))
            conn.commit()
            flash(f"Payment #{period} marked unpaid.", "ok")
        else:
            inst = next((r for r in loan_schedule(loan) if r.period == period), None) or abort(404)
            record_payment(loan, inst, day=date.today())
            flash(f"Payment #{period} recorded as an expense today.", "ok")
        return redirect(request.form.get("next") or url_for("loan_detail", loan_id=loan_id))

    # ---- settings / export
    @app.route("/settings", methods=["GET", "POST"])
    @login_required
    def settings():
        if request.method == "POST":
            db.set_setting("currency", request.form.get("currency", "$").strip()[:4] or "$")
            flash("Settings saved.", "ok")
            return redirect(url_for("settings"))
        return render_template("settings.html")

    @app.route("/export.csv")
    @login_required
    def export_csv():
        rows = db.get_db().execute(
            "SELECT t.day, t.kind, t.category, t.amount, t.merchant, t.note, l.name AS loan, "
            "t.loan_period, t.receipt FROM transactions t LEFT JOIN loans l ON l.id = t.loan_id "
            "WHERE t.source != 'history' ORDER BY t.day").fetchall()
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["date", "type", "category", "amount", "merchant", "note", "loan", "installment", "receipt"])
        w.writerows([[r[k] for k in ("day", "kind", "category", "amount", "merchant", "note",
                                     "loan", "loan_period", "receipt")] for r in rows])
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=transactions.csv"})

    @app.route("/manifest.webmanifest")
    def manifest():
        return send_from_directory(os.path.join(HERE, "static"), "manifest.webmanifest",
                                   mimetype="application/manifest+json")

    @app.route("/healthz")
    def healthz():
        db.get_db().execute("SELECT 1")
        return "ok"


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=bool(os.environ.get("DEBUG")))
